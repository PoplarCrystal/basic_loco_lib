from __future__ import annotations

import torch

import warp as wp
from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_newton.assets.kernels import get_link_vel_from_root_com_vel_func

from basic_loco_lib.tasks.mimic.mdp_warp.warp_utils import quat_conj_xyzw, quat_mul_xyzw, quat_rotate_inv_xyzw


def _body_ids(cfg: SceneEntityCfg, key: str, device: str) -> wp.array:
    cache = getattr(_body_ids, "_cache", None)
    if cache is None:
        cache = _body_ids._cache = {}
    ids = tuple(int(i) for i in cfg.body_ids)
    cache_key = (key, ids, device)
    if cache_key not in cache:
        cache[cache_key] = wp.array(ids, dtype=wp.int32, device=device)
    return cache[cache_key]


@wp.kernel
def _body_pos_kernel(
    pose: wp.array2d(dtype=wp.transformf),
    anchor_id: int,
    ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    anchor_pos = wp.transform_get_translation(pose[env_id, anchor_id])
    anchor_quat_raw = wp.transform_get_rotation(pose[env_id, anchor_id])
    anchor_quat = wp.vec4f(anchor_quat_raw[0], anchor_quat_raw[1], anchor_quat_raw[2], anchor_quat_raw[3])
    for j in range(ids.shape[0]):
        body_id = ids[j]
        delta = wp.transform_get_translation(pose[env_id, body_id]) - anchor_pos
        local = quat_rotate_inv_xyzw(anchor_quat, delta)
        out[env_id, j * 3 + 0] = local[0]
        out[env_id, j * 3 + 1] = local[1]
        out[env_id, j * 3 + 2] = local[2]


@wp.kernel
def _body_ori_kernel(
    pose: wp.array2d(dtype=wp.transformf),
    anchor_id: int,
    ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    anchor_raw = wp.transform_get_rotation(pose[env_id, anchor_id])
    anchor = wp.vec4f(anchor_raw[0], anchor_raw[1], anchor_raw[2], anchor_raw[3])
    for j in range(ids.shape[0]):
        body_id = ids[j]
        body_raw = wp.transform_get_rotation(pose[env_id, body_id])
        body = wp.vec4f(body_raw[0], body_raw[1], body_raw[2], body_raw[3])
        q = quat_mul_xyzw(quat_conj_xyzw(anchor), body)
        x, y, z, w = q[0], q[1], q[2], q[3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        base = j * 6
        out[env_id, base + 0] = 1.0 - 2.0 * (yy + zz)
        out[env_id, base + 1] = 2.0 * (xy - wz)
        out[env_id, base + 2] = 2.0 * (xy + wz)
        out[env_id, base + 3] = 1.0 - 2.0 * (xx + zz)
        out[env_id, base + 4] = 2.0 * (xz - wy)
        out[env_id, base + 5] = 2.0 * (yz + wx)


@wp.kernel
def _body_vel_kernel(
    pose: wp.array2d(dtype=wp.transformf),
    com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    com_pos_b: wp.array2d(dtype=wp.vec3f),
    ids: wp.array(dtype=wp.int32),
    linear: bool,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    for j in range(ids.shape[0]):
        body_id = ids[j]
        body_pose = pose[env_id, body_id]
        q_raw = wp.transform_get_rotation(body_pose)
        q = wp.vec4f(q_raw[0], q_raw[1], q_raw[2], q_raw[3])
        link_vel = get_link_vel_from_root_com_vel_func(com_vel[env_id, body_id], body_pose, com_pos_b[env_id, body_id])
        v = wp.spatial_bottom(link_vel)
        if linear:
            v = wp.spatial_top(link_vel)
        local = quat_rotate_inv_xyzw(q, v)
        out[env_id, j * 3 + 0] = local[0]
        out[env_id, j * 3 + 1] = local[1]
        out[env_id, j * 3 + 2] = local[2]


def _asset_arrays(env, anchor_cfg, body_cfg):
    asset = env.scene[anchor_cfg.name]
    data = asset.data
    body_ids = _body_ids(body_cfg, "amp_body", env.device)
    anchor_id = int(anchor_cfg.body_ids[0])
    return (
        data.body_link_pose_w.warp,
        body_ids,
        anchor_id,
    )


@generic_io_descriptor_warp(out_dim=39, dtype=torch.float32)
def robot_body_pos_b(env, out, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg) -> None:
    pose, ids, anchor_id = _asset_arrays(env, anchor_cfg, body_cfg)
    wp.launch(_body_pos_kernel, dim=env.num_envs, inputs=[pose, anchor_id, ids, out], device=env.device)


@generic_io_descriptor_warp(out_dim=78, dtype=torch.float32)
def robot_body_ori_b(env, out, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg) -> None:
    pose, ids, anchor_id = _asset_arrays(env, anchor_cfg, body_cfg)
    wp.launch(_body_ori_kernel, dim=env.num_envs, inputs=[pose, anchor_id, ids, out], device=env.device)


@generic_io_descriptor_warp(out_dim=39, dtype=torch.float32)
def robot_body_lin_vel_b(env, out, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg) -> None:
    pose, ids, _ = _asset_arrays(env, anchor_cfg, body_cfg)
    data = env.scene[anchor_cfg.name].data
    wp.launch(
        _body_vel_kernel,
        dim=env.num_envs,
        inputs=[pose, data.body_com_vel_w.warp, data.body_com_pos_b.warp, ids, True, out],
        device=env.device,
    )


@generic_io_descriptor_warp(out_dim=39, dtype=torch.float32)
def robot_body_ang_vel_b(env, out, anchor_cfg: SceneEntityCfg, body_cfg: SceneEntityCfg) -> None:
    pose, ids, _ = _asset_arrays(env, anchor_cfg, body_cfg)
    data = env.scene[anchor_cfg.name].data
    wp.launch(
        _body_vel_kernel,
        dim=env.num_envs,
        inputs=[pose, data.body_com_vel_w.warp, data.body_com_pos_b.warp, ids, False, out],
        device=env.device,
    )
