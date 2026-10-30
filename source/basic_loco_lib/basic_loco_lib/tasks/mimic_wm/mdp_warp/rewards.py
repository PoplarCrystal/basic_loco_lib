"""Warp-first tracking reward terms for the mimic task (mirror of ``../mdp/rewards.py``).

All kernels read the persistent Warp views exposed by :class:`MotionCommandWarp`
and evaluate the same ``exp(-error / std**2)`` expressions as the torch terms.
Body errors are averaged over all tracked bodies (the task never uses the
``body_names`` subset parameter).
"""

from __future__ import annotations

import warp as wp
from typing import TYPE_CHECKING

from basic_loco_lib.tasks.mimic_wm.mdp_warp.commands import MotionCommandWarp
from basic_loco_lib.tasks.mimic_wm.mdp_warp.warp_utils import quat_error_magnitude_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _anchor_position_error_kernel(
    robot_pos: wp.array(dtype=wp.float32, ndim=2),
    motion_pos: wp.array(dtype=wp.float32, ndim=2),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dx = robot_pos[i, 0] - motion_pos[i, 0]
    dy = robot_pos[i, 1] - motion_pos[i, 1]
    dz = robot_pos[i, 2] - motion_pos[i, 2]
    error = dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-error / (std * std))


@wp.kernel
def _anchor_orientation_error_kernel(
    robot_quat: wp.array(dtype=wp.float32, ndim=2),
    motion_quat: wp.array(dtype=wp.float32, ndim=2),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    qr = wp.vec4f(robot_quat[i, 0], robot_quat[i, 1], robot_quat[i, 2], robot_quat[i, 3])
    qm = wp.vec4f(motion_quat[i, 0], motion_quat[i, 1], motion_quat[i, 2], motion_quat[i, 3])
    error = quat_error_magnitude_xyzw(qm, qr)
    out[i] = wp.exp(-error * error / (std * std))


@wp.kernel
def _body_position_error_kernel(
    robot_pos: wp.array(dtype=wp.float32, ndim=3),
    motion_pos: wp.array(dtype=wp.float32, ndim=3),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = motion_pos.shape[1]
    error_sum = float(0.0)
    for b in range(num_bodies):
        dx = robot_pos[i, b, 0] - motion_pos[i, b, 0]
        dy = robot_pos[i, b, 1] - motion_pos[i, b, 1]
        dz = robot_pos[i, b, 2] - motion_pos[i, b, 2]
        error_sum += dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


@wp.kernel
def _body_orientation_error_kernel(
    robot_quat: wp.array(dtype=wp.float32, ndim=3),
    motion_quat: wp.array(dtype=wp.float32, ndim=3),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = motion_quat.shape[1]
    error_sum = float(0.0)
    for b in range(num_bodies):
        qr = wp.vec4f(robot_quat[i, b, 0], robot_quat[i, b, 1], robot_quat[i, b, 2], robot_quat[i, b, 3])
        qm = wp.vec4f(motion_quat[i, b, 0], motion_quat[i, b, 1], motion_quat[i, b, 2], motion_quat[i, b, 3])
        error = quat_error_magnitude_xyzw(qm, qr)
        error_sum += error * error
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


@wp.kernel
def _body_velocity_error_kernel(
    robot_vel: wp.array(dtype=wp.float32, ndim=3),
    motion_vel: wp.array(dtype=wp.float32, ndim=3),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = motion_vel.shape[1]
    error_sum = float(0.0)
    for b in range(num_bodies):
        dx = robot_vel[i, b, 0] - motion_vel[i, b, 0]
        dy = robot_vel[i, b, 1] - motion_vel[i, b, 1]
        dz = robot_vel[i, b, 2] - motion_vel[i, b, 2]
        error_sum += dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the anchor global position. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_anchor_position_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_pos_w, command.wp_anchor_pos_w, std, out],
        device=env.device,
    )


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the anchor global orientation. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_anchor_orientation_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_anchor_quat_w, command.wp_anchor_quat_w, std, out],
        device=env.device,
    )


def motion_relative_body_position_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the tracked body positions (anchor-aligned frame). Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_body_position_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_body_pos_w, command.wp_body_pos_rel_w, std, out],
        device=env.device,
    )


def motion_relative_body_orientation_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the tracked body orientations (anchor-aligned frame). Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_body_orientation_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_body_quat_w, command.wp_body_quat_rel_w, std, out],
        device=env.device,
    )


def motion_global_body_linear_velocity_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the tracked body linear velocities. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_body_velocity_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_body_lin_vel_w, command.wp_body_lin_vel_w, std, out],
        device=env.device,
    )


def motion_global_body_angular_velocity_error_exp(env: ManagerBasedRLEnv, out, command_name: str, std: float) -> None:
    """Tracking reward for the tracked body angular velocities. Writes into ``out``."""
    command: MotionCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_body_velocity_error_kernel,
        dim=env.num_envs,
        inputs=[command.wp_robot_body_ang_vel_w, command.wp_body_ang_vel_w, std, out],
        device=env.device,
    )
