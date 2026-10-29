from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionResetManager:
    """Manages motion frame data and delayed-reset logic for AMP environments."""

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls, env) -> MotionResetManager:
        # Joint ordering and device belong to an environment, not a global singleton.
        if not hasattr(env, "_amp_motion_reset_manager"):
            env._amp_motion_reset_manager = cls()
        return env._amp_motion_reset_manager

    def init(
        self,
        env: ManagerBasedRLEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        if motion_dir in self.walk_run_frames:
            return

        joint_names = env.scene["robot"].joint_names
        motion_data = self._load_dir(motion_dir, env.device, joint_names)
        self.walk_run_frames[motion_dir] = self._concat_frames(motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(motion_data)} clips, {motion_count} frames from {motion_dir}")

        if recovery_dir is not None:
            recovery_data = self._load_dir(recovery_dir, env.device, joint_names)
            self.recovery_frames[motion_dir] = self._concat_frames(recovery_data)
            recovery_count = self.recovery_frames[motion_dir]["root_pos"].shape[0]
            print(
                f"[MotionResetManager] Loaded {len(recovery_data)} recovery clips, {recovery_count} frames from {recovery_dir}"
            )

    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]
            normal_ids = env_ids

        if len(normal_ids) > 0:
            self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg)

        if len(delay_ids) > 0:
            self._write_reset_state(env, delay_ids, self.recovery_frames[motion_dir], asset_cfg)

    def _get_delay_env_mask(self, env: ManagerBasedRLEnv) -> torch.Tensor | None:
        return getattr(env.termination_manager, "_delay_env_mask", None)

    def _write_reset_state(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()

        terrain_z = positions[:, 2].clone()
        positions[:, 2] = terrain_z + root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits.torch
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=asset_cfg.joint_ids,
        )

    @staticmethod
    def _load_dir(dir_path: str, device: str, joint_names: list[str]) -> list[dict]:
        import numpy as np
        import os

        assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
        result = []
        from pathlib import Path

        for filename in sorted(Path(dir_path).rglob("*.npz")):
            data = np.load(filename)
            file_joint_names = data["joint_names"].tolist()
            joint_ids = [file_joint_names.index(name) for name in joint_names]
            body_names = data["body_names"].tolist()
            root_name = data["root_name"].item() if "root_name" in data else body_names[0]
            root_id = body_names.index(root_name)
            result.append(
                {
                    "root_pos": torch.tensor(data["body_pos_w"][:, root_id], dtype=torch.float32, device=device),
                    "root_quat": torch.tensor(data["body_quat_w"][:, root_id], dtype=torch.float32, device=device),
                    "root_lin_vel": torch.tensor(
                        data["body_lin_vel_w"][:, root_id], dtype=torch.float32, device=device
                    ),
                    "root_ang_vel": torch.tensor(
                        data["body_ang_vel_w"][:, root_id], dtype=torch.float32, device=device
                    ),
                    "joint_pos": torch.tensor(data["joint_pos"][:, joint_ids], dtype=torch.float32, device=device),
                    "joint_vel": torch.tensor(data["joint_vel"][:, joint_ids], dtype=torch.float32, device=device),
                }
            )
        return result

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        return {name: torch.cat([motion[name] for motion in motions], dim=0) for name in motions[0]}


# ------------------------------------------------------------------
# Event callback wrappers
# ------------------------------------------------------------------


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
    max_delay_joint_velocity: float = 50.0,
    termination_manager_cls: type = DelayedTerminationManager,
) -> None:
    """Startup event: load motion data and optionally install delayed termination."""
    MotionResetManager.get(env).init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    num_delay = int(env.num_envs * delay_reset_env_ratio)
    # Install persistent delay buffers even when tiny batches round down to zero
    # recovery environments; the Warp reward path needs them during graph capture.
    if delay_reset_env_ratio > 0.0 and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
        delay_mask[delay_indices] = True
        env.termination_manager = termination_manager_cls(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
            joint_vel=env.scene["robot"].data.joint_vel.torch,
            max_joint_velocity=max_delay_joint_velocity,
        )
        print(
            "[init_motion_loader] DelayedTerminationManager installed: "
            f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}, "
            f"max_joint_velocity={max_delay_joint_velocity} rad/s"
        )


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random motion frames."""
    MotionResetManager.get(env).reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        asset_cfg=asset_cfg,
    )
