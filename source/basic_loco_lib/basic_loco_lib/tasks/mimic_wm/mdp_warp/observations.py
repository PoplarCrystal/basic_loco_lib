"""Warp-first observation terms for the mimic task (mirror of ``../mdp/observations.py``).

Reads the persistent Warp views exposed by :class:`MotionCommandWarp`; the
outputs are numerically identical to the torch terms (rotation matrices are the
first two columns of ``matrix_from_quat``, flattened row-major).
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp

from basic_loco_lib.tasks.mimic_wm.mdp_warp.commands import MotionCommandWarp
from basic_loco_lib.tasks.mimic_wm.mdp_warp.warp_utils import quat_conj_xyzw, quat_mul_xyzw, quat_rotate_inv_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _motion_anchor_pos_b_kernel(
    robot_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    robot_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    motion_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    qr = wp.vec4f(robot_anchor_quat[i, 0], robot_anchor_quat[i, 1], robot_anchor_quat[i, 2], robot_anchor_quat[i, 3])
    d = wp.vec3f(
        motion_anchor_pos[i, 0] - robot_anchor_pos[i, 0],
        motion_anchor_pos[i, 1] - robot_anchor_pos[i, 1],
        motion_anchor_pos[i, 2] - robot_anchor_pos[i, 2],
    )
    p = quat_rotate_inv_xyzw(qr, d)
    out[i, 0] = p[0]
    out[i, 1] = p[1]
    out[i, 2] = p[2]


@wp.kernel
def _motion_anchor_ori_b_kernel(
    robot_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    motion_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    qr = wp.vec4f(robot_anchor_quat[i, 0], robot_anchor_quat[i, 1], robot_anchor_quat[i, 2], robot_anchor_quat[i, 3])
    qm = wp.vec4f(motion_anchor_quat[i, 0], motion_anchor_quat[i, 1], motion_anchor_quat[i, 2], motion_anchor_quat[i, 3])
    q = quat_mul_xyzw(quat_conj_xyzw(qr), qm)
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    # first two columns of the rotation matrix, flattened row-major
    out[i, 0] = 1.0 - 2.0 * (yy + zz)
    out[i, 1] = 2.0 * (xy - wz)
    out[i, 2] = 2.0 * (xy + wz)
    out[i, 3] = 1.0 - 2.0 * (xx + zz)
    out[i, 4] = 2.0 * (xz - wy)
    out[i, 5] = 2.0 * (yz + wx)


@wp.kernel
def _robot_body_pos_b_kernel(
    robot_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    robot_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    qa = wp.vec4f(robot_anchor_quat[i, 0], robot_anchor_quat[i, 1], robot_anchor_quat[i, 2], robot_anchor_quat[i, 3])
    pa = wp.vec3f(robot_anchor_pos[i, 0], robot_anchor_pos[i, 1], robot_anchor_pos[i, 2])
    for b in range(robot_body_pos.shape[1]):
        d = wp.vec3f(
            robot_body_pos[i, b, 0] - pa[0],
            robot_body_pos[i, b, 1] - pa[1],
            robot_body_pos[i, b, 2] - pa[2],
        )
        p = quat_rotate_inv_xyzw(qa, d)
        out[i, 3 * b + 0] = p[0]
        out[i, 3 * b + 1] = p[1]
        out[i, 3 * b + 2] = p[2]


@wp.kernel
def _robot_body_ori_b_kernel(
    robot_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    robot_body_quat: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    qa = wp.vec4f(robot_anchor_quat[i, 0], robot_anchor_quat[i, 1], robot_anchor_quat[i, 2], robot_anchor_quat[i, 3])
    for b in range(robot_body_quat.shape[1]):
        qb = wp.vec4f(
            robot_body_quat[i, b, 0], robot_body_quat[i, b, 1], robot_body_quat[i, b, 2], robot_body_quat[i, b, 3]
        )
        q = quat_mul_xyzw(quat_conj_xyzw(qa), qb)
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        out[i, 6 * b + 0] = 1.0 - 2.0 * (yy + zz)
        out[i, 6 * b + 1] = 2.0 * (xy - wz)
        out[i, 6 * b + 2] = 2.0 * (xy + wz)
        out[i, 6 * b + 3] = 1.0 - 2.0 * (xx + zz)
        out[i, 6 * b + 4] = 2.0 * (xz - wy)
        out[i, 6 * b + 5] = 2.0 * (yz + wx)


@generic_io_descriptor_warp(out_dim=3, dtype=torch.float32)
def motion_anchor_pos_b(env: ManagerBasedEnv, out, command_name: str) -> None:
    """Reference anchor position in the robot anchor frame. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_motion_anchor_pos_b_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_pos_w, command.wp_robot_anchor_quat_w, command.wp_anchor_pos_w, out],
        device=env.device,
    )


@generic_io_descriptor_warp(out_dim=6, dtype=torch.float32)
def motion_anchor_ori_b(env: ManagerBasedEnv, out, command_name: str) -> None:
    """Reference anchor orientation in the robot anchor frame. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_motion_anchor_ori_b_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_quat_w, command.wp_anchor_quat_w, out],
        device=env.device,
    )


# out_dim: 14 tracked bodies x 3 (the "body:N" sentinel resolves via asset_cfg, which
# these command-based terms do not have)
@generic_io_descriptor_warp(out_dim=42, dtype=torch.float32)
def robot_body_pos_b(env: ManagerBasedEnv, out, command_name: str) -> None:
    """Tracked body positions in the robot anchor frame. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_robot_body_pos_b_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_pos_w, command.wp_robot_anchor_quat_w, command.wp_robot_body_pos_w, out],
        device=env.device,
    )


@generic_io_descriptor_warp(out_dim=84, dtype=torch.float32)  # 14 tracked bodies x 6
def robot_body_ori_b(env: ManagerBasedEnv, out, command_name: str) -> None:
    """Tracked body orientations in the robot anchor frame. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_robot_body_ori_b_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_quat_w, command.wp_robot_body_quat_w, out],
        device=env.device,
    )
