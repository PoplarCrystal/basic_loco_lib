"""Warp equivalents of the velocity task's custom rewards."""
from functools import lru_cache

import torch
import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg


def _wp(value):
    return wp.from_torch(value if torch.is_tensor(value) else wp.to_torch(value))


@lru_cache(maxsize=None)
def _indices(indices, device):
    return wp.array(indices, dtype=wp.int32, device=device)


@lru_cache(maxsize=None)
def _offsets(offset, device):
    return wp.array(offset, dtype=wp.float32, device=device)


@wp.kernel
def _energy(velocity: wp.array2d(dtype=float), torque: wp.array2d(dtype=float),
            ids: wp.array(dtype=int), out: wp.array(dtype=float)):
    i = wp.tid()
    total = float(0.0)
    for k in range(ids.shape[0]):
        j = ids[k]
        total += wp.abs(velocity[i, j]) * wp.abs(torque[i, j])
    out[i] = total


def energy(env, out, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')):
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        ids = range(len(asset.joint_names))[ids]
    ids_wp = _indices(tuple(ids), str(env.device))
    wp.launch(_energy, dim=env.num_envs,
              inputs=[_wp(asset.data.joint_vel), _wp(asset.data.applied_torque), ids_wp, out], device=env.device)


@wp.kernel
def _height(pos: wp.array2d(dtype=float), target: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = wp.pow(pos[i, 2] - target, 2.0)


def base_height_l2(env, out, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')):
    wp.launch(_height, dim=env.num_envs,
              inputs=[_wp(env.scene[asset_cfg.name].data.root_pos_w), target_height, out], device=env.device)


@wp.kernel
def _clearance(pos: wp.array3d(dtype=float), velocity: wp.array3d(dtype=float),
               ids: wp.array(dtype=int), target: float, std: float, mult: float, out: wp.array(dtype=float)):
    i = wp.tid()
    total = float(0.0)
    for k in range(ids.shape[0]):
        j = ids[k]
        speed = wp.sqrt(velocity[i, j, 0] ** 2.0 + velocity[i, j, 1] ** 2.0)
        total += (pos[i, j, 2] - target) ** 2.0 * wp.tanh(mult * speed)
    out[i] = wp.exp(-total / std)


def foot_clearance_reward(env, out, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float):
    asset = env.scene[asset_cfg.name]
    wp.launch(_clearance, dim=env.num_envs,
              inputs=[_wp(asset.data.body_pos_w), _wp(asset.data.body_lin_vel_w), asset_cfg.body_ids_wp,
                      target_height, std, tanh_mult, out], device=env.device)


@wp.kernel
def _gait(steps: wp.array(dtype=wp.int64), dt: float, period: float,
          offsets: wp.array(dtype=float), contact_time: wp.array2d(dtype=float),
          ids: wp.array(dtype=int), threshold: float, command: wp.array2d(dtype=float),
          gate: bool, out: wp.array(dtype=float)):
    i = wp.tid()
    phase = (float(steps[i]) * dt % period) / period
    total = float(0.0)
    for k in range(ids.shape[0]):
        stance = (phase + offsets[k]) % 1.0 < threshold
        contact = contact_time[i, ids[k]] > 0.0
        if stance == contact:
            total += 1.0
    if gate:
        norm_sq = float(0.0)
        for j in range(command.shape[1]):
            norm_sq += command[i, j] * command[i, j]
        if norm_sq <= 0.01:
            total = 0.0
    out[i] = total


def feet_gait(env, out, period: float, offset: list[float], sensor_cfg: SceneEntityCfg,
              threshold: float = 0.5, command_name=None):
    sensor = env.scene.sensors[sensor_cfg.name]
    if len(offset) != len(sensor_cfg.body_ids):
        raise ValueError('Gait phase offsets must match the selected feet.')
    # The task always supplies base_velocity; retain optional gating semantics.
    command = env.command_manager.get_command(command_name or 'base_velocity')
    wp.launch(_gait, dim=env.num_envs,
              inputs=[env._episode_length_buf_wp, env.step_dt, period, _offsets(tuple(offset), str(env.device)),
                      _wp(sensor.data.current_contact_time), sensor_cfg.body_ids_wp, threshold,
                      _wp(command), command_name is not None, out], device=env.device)
