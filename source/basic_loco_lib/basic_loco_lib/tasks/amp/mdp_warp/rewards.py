"""Warp AMP rewards for captured Newton rollouts."""

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg

from basic_loco_lib.tasks.amp.mdp_warp.warp_utils import (
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


@wp.kernel
def _feet_slip(
    body_lin_vel_w: wp.array(dtype=wp.vec3f, ndim=2),
    body_ang_vel_w: wp.array(dtype=wp.vec3f, ndim=2),
    body_quat_w: wp.array(dtype=wp.quatf, ndim=2),
    body_com_pos_b: wp.array(dtype=wp.vec3f, ndim=2),
    body_net_forces_w: wp.array(dtype=wp.vec3f, ndim=2),
    sensor_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    command: wp.array(dtype=wp.float32, ndim=2),
    command_threshold: float,
    foot_offset: wp.vec3f,
    out: wp.array(dtype=wp.float32),
):
    """Penalize foot sliding at a virtual sole point.

    URDF has no MuJoCo sites, so the sole point is reconstructed from the ankle
    body: ``v_site = v_com + omega x (R @ (foot_offset - com_local))``.
    """
    i = wp.tid()
    command_norm = wp.sqrt(command[i, 0] * command[i, 0] + command[i, 1] * command[i, 1])
    active = float(0.0)
    if command_norm + wp.abs(command[i, 2]) > command_threshold:
        active = 1.0

    cost = float(0.0)
    for k in range(sensor_ids.shape[0]):
        force = body_net_forces_w[i, sensor_ids[k]]
        if wp.dot(force, force) > 1.0:
            body_id = body_ids[k]
            q = body_quat_w[i, body_id]
            quat_xyzw = wp.vec4f(q[0], q[1], q[2], q[3])
            arm_body = foot_offset - body_com_pos_b[i, body_id]
            arm_world = quat_rotate_xyzw(quat_xyzw, arm_body)
            site_vel = body_lin_vel_w[i, body_id] + wp.cross(body_ang_vel_w[i, body_id], arm_world)
            cost += site_vel[0] * site_vel[0] + site_vel[1] * site_vel[1]
    out[i] = cost * active


def feet_slip(
    env,
    out,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    foot_offset: tuple[float, float, float] = (0.04, 0.0, -0.035),
):
    """AMP_mjlab's ``feet_slip`` at a virtual sole point (no URDF sites)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    wp.launch(
        _feet_slip,
        dim=env.num_envs,
        inputs=[
            _wp(asset.data.body_lin_vel_w),
            _wp(asset.data.body_ang_vel_w),
            _wp(asset.data.body_quat_w),
            _wp(asset.data.body_com_pos_b),
            _wp(sensor.data.net_forces_w),
            sensor_cfg.body_ids_wp,
            asset_cfg.body_ids_wp,
            _wp(command),
            command_threshold,
            wp.vec3f(*foot_offset),
            out,
        ],
        device=env.device,
    )


# @wp.kernel
# def _self_collision_matrix(
#     force_history: wp.array(dtype=wp.vec3f, ndim=4),
#     force_threshold: float,
#     out: wp.array(dtype=wp.float32),
# ):
#     """Count history frames with any robot-robot force above the threshold."""
#     i = wp.tid()
#     count = float(0.0)
#     for t in range(force_history.shape[1]):
#         hit = wp.bool(False)
#         for s in range(force_history.shape[2]):
#             for f in range(force_history.shape[3]):
#                 force = force_history[i, t, s, f]
#                 if wp.dot(force, force) > force_threshold * force_threshold:
#                     hit = True
#         if hit:
#             count += 1.0
#     out[i] = count
#
#
# @wp.kernel
# def _self_collision_net(
#     force_history: wp.array(dtype=wp.vec3f, ndim=3),
#     body_ids: wp.array(dtype=wp.int32),
#     force_threshold: float,
#     out: wp.array(dtype=wp.float32),
# ):
#     """Fallback: non-foot body net forces (PhysX has no shape-level filters)."""
#     i = wp.tid()
#     count = float(0.0)
#     for t in range(force_history.shape[1]):
#         hit = wp.bool(False)
#         for k in range(body_ids.shape[0]):
#             force = force_history[i, t, body_ids[k]]
#             if wp.dot(force, force) > force_threshold * force_threshold:
#                 hit = True
#         if hit:
#             count += 1.0
#     out[i] = count
#
#
# def self_collision_cost(
#     env,
#     out,
#     sensor_name: str,
#     force_threshold: float = 10.0,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("self_collision"),
# ):
#     sensor = env.scene.sensors[sensor_name]
#     matrix = sensor.data.force_matrix_w_history
#     if matrix is not None:
#         wp.launch(
#             _self_collision_matrix,
#             dim=env.num_envs,
#             inputs=[_wp(matrix), force_threshold, out],
#             device=env.device,
#         )
#         return
#
#     history = sensor.data.net_forces_w_history
#     if history is None:
#         raise RuntimeError("self_collision requires force_matrix_w_history or net_forces_w_history")
#     wp.launch(
#         _self_collision_net,
#         dim=env.num_envs,
#         inputs=[_wp(history), sensor_cfg.body_ids_wp, force_threshold, out],
#         device=env.device,
#     )
