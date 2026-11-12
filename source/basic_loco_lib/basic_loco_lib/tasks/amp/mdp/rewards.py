"""Torch AMP rewards matching the Warp task's frames, scales, and delay masks."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_apply_yaw


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


def feet_slip(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_offset: tuple[float, float, float] = (0.04, 0.0, -0.035),
) -> torch.Tensor:
    """AMP_mjlab's ``feet_slip`` at a virtual sole point (URDF has no sites)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    sensor_ids = sensor_cfg.body_ids
    body_ids = asset_cfg.body_ids

    forces = _as_torch(sensor.data.net_forces_w)
    in_contact = (torch.linalg.vector_norm(forces[:, sensor_ids], dim=-1) > 1.0).float()

    quat = asset.data.body_quat_w.torch[:, body_ids]
    com = asset.data.body_com_pos_b.torch[:, body_ids]
    offset = torch.tensor(foot_offset, dtype=quat.dtype, device=quat.device)
    arm_world = quat_apply(quat, offset - com)
    site_vel = (
        asset.data.body_lin_vel_w.torch[:, body_ids]
        + torch.cross(asset.data.body_ang_vel_w.torch[:, body_ids], arm_world, dim=-1)
    )

    command = env.command_manager.get_command(command_name)
    active = (
        torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs() > command_threshold
    ).float()
    return (site_vel[..., :2].square().sum(dim=-1) * in_contact).sum(dim=-1) * active


def _as_torch(value):
    """Torch view of a data buffer (torch tensor or IsaacLab proxy)."""
    return getattr(value, "torch", value)


def self_collision_cost(
    env,
    sensor_name: str,
    force_threshold: float = 10.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("self_collision"),
) -> torch.Tensor:
    """Count history frames with any robot self-contact above the force threshold.

    Newton uses the shape-level force matrix (ground excluded). The PhysX
    fallback uses non-foot body net forces, which excludes ordinary foot-ground
    contact but can still see other body-ground contacts during a fall.
    """
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data

    matrix = data.force_matrix_w_history
    if matrix is not None:
        force = _as_torch(matrix)  # (N, T, S, F, 3)
        magnitude = torch.linalg.vector_norm(force, dim=-1)  # (N, T, S, F)
        hit = (magnitude > force_threshold).any(dim=-1).any(dim=-1)  # (N, T)
        return hit.float().sum(dim=-1)

    history = data.net_forces_w_history
    if history is None:
        raise RuntimeError("self_collision requires force_matrix_w_history or net_forces_w_history")
    force = _as_torch(history)  # (N, T, S, 3)
    ids = sensor_cfg.body_ids
    magnitude = torch.linalg.vector_norm(force[:, :, ids], dim=-1)  # (N, T, K)
    return (magnitude > force_threshold).any(dim=-1).sum(dim=-1).float()
