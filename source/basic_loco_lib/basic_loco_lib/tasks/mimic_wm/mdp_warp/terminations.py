"""Warp-first termination terms for the mimic task (mirror of ``../mdp/terminations.py``)."""

from __future__ import annotations

import math

import warp as wp
from typing import TYPE_CHECKING

from isaaclab_experimental.managers import SceneEntityCfg

from basic_loco_lib.tasks.mimic_wm.mdp.rewards import _get_body_indexes
from basic_loco_lib.tasks.mimic_wm.mdp_warp.commands import MotionCommandWarp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _bad_anchor_pos_z_kernel(
    robot_pos: wp.array(dtype=wp.float32, ndim=2),
    motion_pos: wp.array(dtype=wp.float32, ndim=2),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    out[i] = wp.abs(robot_pos[i, 2] - motion_pos[i, 2]) > threshold


@wp.kernel
def _bad_anchor_ori_kernel(
    robot_quat: wp.array(dtype=wp.float32, ndim=2),
    motion_quat: wp.array(dtype=wp.float32, ndim=2),
    gravity_z: float,
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    qr = wp.vec4f(robot_quat[i, 0], robot_quat[i, 1], robot_quat[i, 2], robot_quat[i, 3])
    qm = wp.vec4f(motion_quat[i, 0], motion_quat[i, 1], motion_quat[i, 2], motion_quat[i, 3])
    # z of the gravity vector projected into the body frame equals gravity_z * R[2, 2],
    # with R the rotation matrix of the quaternion
    r22_robot = 1.0 - 2.0 * (qr[0] * qr[0] + qr[1] * qr[1])
    r22_motion = 1.0 - 2.0 * (qm[0] * qm[0] + qm[1] * qm[1])
    out[i] = wp.abs(gravity_z * (r22_motion - r22_robot)) > threshold


@wp.kernel
def _bad_motion_body_pos_z_kernel(
    robot_pos: wp.array(dtype=wp.float32, ndim=3),
    motion_pos: wp.array(dtype=wp.float32, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    violations = int(0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        if wp.abs(robot_pos[i, b, 2] - motion_pos[i, b, 2]) > threshold:
            violations += 1
    out[i] = violations > 0


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, out, command_name: str, threshold: float) -> None:
    """Terminate when the anchor height deviates from the reference. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_bad_anchor_pos_z_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_pos_w, command.wp_anchor_pos_w, threshold, out],
        device=env.device,
    )


def bad_anchor_ori(
    env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> None:
    """Terminate when the anchor orientation deviates from the reference. Writes into ``out``."""
    # the torch term compares the projected NORMALIZED gravity direction (Newton's
    # GRAVITY_VEC_W is normalize(gravity), e.g. (0, 0, -1)), so normalize the sim
    # gravity here too; host-only math, safe inside the CUDA-graph capture path
    gravity = env.cfg.sim.gravity
    gravity_norm = math.sqrt(gravity[0] ** 2 + gravity[1] ** 2 + gravity[2] ** 2)
    gravity_z = gravity[2] / gravity_norm

    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_bad_anchor_ori_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_quat_w, command.wp_anchor_quat_w, gravity_z, threshold, out],
        device=env.device,
    )


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, out, command_name: str, threshold: float, body_names: list[str] | None = None
) -> None:
    """Terminate when any selected tracked body height deviates from the reference. Writes into ``out``."""
    fn = bad_motion_body_pos_z_only
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    key = tuple(body_names) if body_names is not None else ()
    if getattr(fn, "_body_key", None) != key:
        # persistent warp array of tracked-body indices (stable across CUDA graph captures)
        body_indexes = _get_body_indexes(command, body_names)
        fn._body_ids_wp = wp.array(body_indexes, dtype=wp.int32, device=env.device)
        fn._body_key = key
    wp.launch(
        kernel=_bad_motion_body_pos_z_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_body_pos_w, command.wp_body_pos_rel_w, fn._body_ids_wp, threshold, out],
        device=env.device,
    )
