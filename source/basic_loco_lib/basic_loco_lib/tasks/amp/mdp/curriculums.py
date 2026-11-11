"""Command-range curricula for the AMP task.

Ported from AMP_mjlab's ``velocity.mdp.curriculums.commands_vel``: stage the
sampled velocity ranges by the environment's common step counter. The command
term reads its ranges on every compute, so mutating ``cfg.ranges`` in place is
enough; the curriculum manager runs on the stable (torch) implementation even
for the Warp task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def commands_vel(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str,
    velocity_stages: list[dict],
) -> dict:
    """Apply staged command ranges when ``common_step_counter`` passes a step.

    Each stage may set ``lin_vel_x``, ``lin_vel_y`` and/or ``ang_vel_z``.
    Stages are applied in order while ``common_step_counter > step``, so later
    stages override earlier ones. Fields omitted from a stage are left at their
    previous value (matching AMP_mjlab).
    """
    del env_ids  # The stage is global, not per-environment.

    command_term = env.command_manager.get_term(command_name)
    cfg = command_term.cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if stage.get("lin_vel_x") is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]
            if stage.get("lin_vel_y") is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]
            if stage.get("ang_vel_z") is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    return {}


__all__ = ["commands_vel"]
