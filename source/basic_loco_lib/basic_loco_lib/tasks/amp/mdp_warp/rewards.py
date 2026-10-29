"""Warp AMP rewards for captured Newton rollouts."""

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg

from basic_loco_lib.tasks.mimic.mdp_warp.warp_utils import (
    quat_rotate_inv_xyzw,
    quat_rotate_xyzw,
    yaw_quat_xyzw,
)

from .utils import _delay_state, _wp


@wp.kernel
def _track_lin_vel(
    vel: wp.array(dtype=wp.vec3f, ndim=2),
    quat: wp.array(dtype=wp.quatf, ndim=2),
    command: wp.array(dtype=wp.float32, ndim=2),
    anchor_id: int,
    std: float,
    delay_mask: wp.array(dtype=wp.bool),
    delay_counters: wp.array(dtype=wp.int64),
    mask_delay: bool,
    delay_env_rew_ratio: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    command_local = wp.vec3f(command[i, 0], command[i, 1], 0.0)
    anchor_quat = quat[i, anchor_id]
    quat_xyzw = wp.vec4f(anchor_quat[0], anchor_quat[1], anchor_quat[2], anchor_quat[3])
    command_world = quat_rotate_xyzw(yaw_quat_xyzw(quat_xyzw), command_local)
    error = command_world - vel[i, anchor_id]
    reward = wp.exp(-wp.dot(error, error) / (std * std))
    if mask_delay and delay_mask[i] and delay_counters[i] > 0:
        reward = reward * delay_env_rew_ratio
    out[i] = reward


@wp.kernel
def _track_ang_vel(
    vel: wp.array(dtype=wp.vec3f, ndim=2),
    quat: wp.array(dtype=wp.quatf, ndim=2),
    command: wp.array(dtype=wp.float32, ndim=2),
    anchor_id: int,
    std: float,
    delay_mask: wp.array(dtype=wp.bool),
    delay_counters: wp.array(dtype=wp.int64),
    mask_delay: bool,
    delay_env_rew_ratio: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    anchor_quat = quat[i, anchor_id]
    quat_xyzw = wp.vec4f(anchor_quat[0], anchor_quat[1], anchor_quat[2], anchor_quat[3])
    anchor_ang_vel_b = quat_rotate_inv_xyzw(quat_xyzw, vel[i, anchor_id])
    error_z = command[i, 2] - vel[i, anchor_id][2]
    error = error_z * error_z + anchor_ang_vel_b[0] * anchor_ang_vel_b[0] + anchor_ang_vel_b[1] * anchor_ang_vel_b[1]
    reward = wp.exp(-error / (std * std))
    if mask_delay and delay_mask[i] and delay_counters[i] > 0:
        reward = reward * delay_env_rew_ratio
    out[i] = reward


@wp.kernel
def _track_height(
    root_pos: wp.array(dtype=wp.vec3f),
    default_height: float,
    std: float,
    delay_mask: wp.array(dtype=wp.bool),
    delay_counters: wp.array(dtype=wp.int64),
    mask_delay: bool,
    delay_env_rew_ratio: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    error = root_pos[i][2] - default_height
    reward = wp.exp(-(error * error) / (std * std))
    if mask_delay:
        if delay_mask[i] and delay_counters[i] > 0:
            reward = reward * delay_env_rew_ratio
        else:
            reward = 0.0
    out[i] = reward


@wp.kernel
def _body_ang_vel_xy(
    vel: wp.array(dtype=wp.vec3f, ndim=2),
    quat: wp.array(dtype=wp.quatf, ndim=2),
    body_id: int,
    std: float,
    delay_mask: wp.array(dtype=wp.bool),
    delay_counters: wp.array(dtype=wp.int64),
    mask_delay: bool,
    delay_env_rew_ratio: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    body_quat = quat[i, body_id]
    quat_xyzw = wp.vec4f(body_quat[0], body_quat[1], body_quat[2], body_quat[3])
    ang_vel_b = quat_rotate_inv_xyzw(quat_xyzw, vel[i, body_id])
    reward = wp.exp(-(ang_vel_b[0] * ang_vel_b[0] + ang_vel_b[1] * ang_vel_b[1]) / (std * std))
    if mask_delay and delay_mask[i] and delay_counters[i] > 0:
        reward = reward * delay_env_rew_ratio
    out[i] = reward


def track_anchor_linear_velocity(
    env,
    out,
    command_name: str,
    std: float,
    anchor_cfg: SceneEntityCfg,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
):
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    wp.launch(
        _track_lin_vel,
        dim=env.num_envs,
        inputs=[
            _wp(asset.data.body_lin_vel_w),
            _wp(asset.data.body_quat_w),
            _wp(command),
            int(anchor_cfg.body_ids[0]),
            std,
            *_delay_state(env)[:2],
            mask_delay,
            delay_env_rew_ratio,
            out,
        ],
        device=env.device,
    )


def track_anchor_angular_velocity(
    env,
    out,
    command_name: str,
    std: float,
    anchor_cfg: SceneEntityCfg,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
):
    asset = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    wp.launch(
        _track_ang_vel,
        dim=env.num_envs,
        inputs=[
            _wp(asset.data.body_ang_vel_w),
            _wp(asset.data.body_quat_w),
            _wp(command),
            int(anchor_cfg.body_ids[0]),
            std,
            *_delay_state(env)[:2],
            mask_delay,
            delay_env_rew_ratio,
            out,
        ],
        device=env.device,
    )


def track_root_height(env, out, std: float, mask_delay: bool = True, delay_env_rew_ratio: float = 3.5):
    asset = env.scene["robot"]
    default_height = float(asset.cfg.init_state.pos[2])
    wp.launch(
        _track_height,
        dim=env.num_envs,
        inputs=[
            _wp(asset.data.root_pos_w),
            default_height,
            std,
            *_delay_state(env)[:2],
            mask_delay,
            delay_env_rew_ratio,
            out,
        ],
        device=env.device,
    )


def body_ang_vel_xy_l2(
    env, out, std: float, body_cfg: SceneEntityCfg, mask_delay: bool = True, delay_env_rew_ratio: float = 0.0
):
    asset = env.scene[body_cfg.name]
    wp.launch(
        _body_ang_vel_xy,
        dim=env.num_envs,
        inputs=[
            _wp(asset.data.body_ang_vel_w),
            _wp(asset.data.body_quat_w),
            int(body_cfg.body_ids[0]),
            std,
            *_delay_state(env)[:2],
            mask_delay,
            delay_env_rew_ratio,
            out,
        ],
        device=env.device,
    )
