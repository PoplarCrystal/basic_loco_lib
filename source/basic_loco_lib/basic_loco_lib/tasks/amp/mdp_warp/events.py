"""Motion startup and reset callbacks for Warp managers."""

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.utils.warp import warp_capturable

from basic_loco_lib.tasks.amp.mdp.events import MotionResetManager, init_motion_loader

from .terminations import DelayedTerminationManagerWarp


@warp_capturable(False)
def reset_from_motion_data_warp(env, env_mask, motion_dir: str, asset_cfg: SceneEntityCfg):
    env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).flatten()
    MotionResetManager.get(env).reset(env=env, env_ids=env_ids, motion_dir=motion_dir, asset_cfg=asset_cfg)


def init_motion_loader_warp(
    env,
    env_mask,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
    max_delay_joint_velocity: float = 50.0,
):
    init_motion_loader(
        env,
        None,
        motion_dir,
        recovery_dir,
        delay_reset_env_ratio,
        max_delay_steps,
        max_delay_joint_velocity,
        termination_manager_cls=DelayedTerminationManagerWarp,
    )
