"""Torch AMP rewards matching the Warp task's frames, scales, and delay masks."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw


def track_anchor_linear_velocity(
    env,
    command_name: str,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]
    anchor_id = anchor_cfg.body_ids[0]
    command = env.command_manager.get_command(command_name)
    command_local = torch.zeros_like(command)
    command_local[:, :2] = command[:, :2]
    command_world = quat_apply_yaw(asset.data.body_quat_w.torch[:, anchor_id], command_local)
    error = command_world - asset.data.body_lin_vel_w.torch[:, anchor_id]
    reward = torch.exp(-error.square().sum(dim=-1) / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def track_anchor_angular_velocity(
    env,
    command_name: str,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]
    anchor_id = anchor_cfg.body_ids[0]
    velocity = asset.data.body_ang_vel_w.torch[:, anchor_id]
    local_velocity = quat_apply_inverse(asset.data.body_quat_w.torch[:, anchor_id], velocity)
    command = env.command_manager.get_command(command_name)
    error = (command[:, 2] - velocity[:, 2]).square() + local_velocity[:, :2].square().sum(dim=-1)
    reward = torch.exp(-error / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def track_root_height(env, std: float, mask_delay: bool = True, delay_env_rew_ratio: float = 3.5) -> torch.Tensor:
    asset = env.scene["robot"]
    error = asset.data.root_pos_w.torch[:, 2] - asset.cfg.init_state.pos[2]
    reward = torch.exp(-error.square() / std**2)
    if mask_delay:
        # Height tracking is a recovery-only reward in the Warp task.
        delay = _active_delay(env)
        reward = torch.where(delay, reward * delay_env_rew_ratio, 0.0)
    return reward


def body_ang_vel_xy_l2(
    env,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("pelvis",)),
) -> torch.Tensor:
    asset = env.scene[body_cfg.name]
    body_id = body_cfg.body_ids[0]
    local_velocity = quat_apply_inverse(
        asset.data.body_quat_w.torch[:, body_id], asset.data.body_ang_vel_w.torch[:, body_id]
    )
    reward = torch.exp(-local_velocity[:, :2].square().sum(dim=-1) / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def _active_delay(env) -> torch.Tensor:
    manager = env.termination_manager
    if hasattr(manager, "_delay_env_mask"):
        return manager._delay_env_mask & (manager._delay_counters > 0)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _mask_by_delay(env, reward: torch.Tensor, delay_rew_ratio: float) -> torch.Tensor:
    return torch.where(_active_delay(env), reward * delay_rew_ratio, reward)
