"""Warp-convention startup events for the mimic Warp task.

The experimental EventManager calls terms as ``func(env, env_mask, **params)``
with a boolean Warp mask over envs, which the classic torch events do not
accept.  The startup events here adapt the mimic ones; the interval push uses
the official Warp-first ``isaaclab_experimental.envs.mdp.push_by_setting_velocity``.
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Literal
from types import MethodType

from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab_experimental.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _process_actions(
    actions: wp.array2d(dtype=wp.float32),
    action_offset: int,
    scale: wp.array(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    clip: wp.array2d(dtype=wp.float32),
    raw: wp.array2d(dtype=wp.float32),
    processed: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    value = actions[i, action_offset + j]
    raw[i, j] = value
    processed[i, j] = wp.clamp(value * scale[j] + default_joint_pos[i, joint_ids[j]], clip[j, 0], clip[j, 1])


def _process_per_env_actions(term, actions: wp.array, action_offset: int = 0):
    """Read the per-env default pose without changing the official action class."""
    wp.launch(
        _process_actions,
        dim=(term.num_envs, term.action_dim),
        inputs=[actions, int(action_offset), term._scale,
                term._event_default_joint_pos, term._default_joint_ids,
                term._clip, term._raw_actions, term._processed_actions],
        device=term.device,
    )


def _mask_to_env_ids(env_mask: wp.array) -> torch.Tensor:
    """Boolean Warp env mask -> torch index tensor."""
    return wp.to_torch(env_mask).nonzero(as_tuple=True)[0]


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_mask: wp.array,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize per-env defaults; the position action reads this buffer directly."""
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term("JointPositionAction")
    if action_term._asset is not asset or not action_term.cfg.use_default_offset:
        raise ValueError("Joint default randomization requires the same asset and use_default_offset=True.")
    default_pos = wp.to_torch(asset.data.default_joint_pos)
    asset.data.default_joint_pos_nominal = default_pos[0].clone()
    env_ids = _mask_to_env_ids(env_mask)
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        joint_ids = torch.tensor(list(joint_ids), dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    pos = _randomize_prop_by_op(
        default_pos.clone(), pos_distribution_params, env_ids, joint_ids,
        operation=operation, distribution=distribution,
    )[env_ids][:, joint_ids]
    selected_envs = env_ids if isinstance(joint_ids, slice) else env_ids[:, None]
    default_pos[selected_envs, joint_ids] = pos

    # Install once during startup, before graph capture. The official action
    # uses a shared 1-D offset; this instance instead reads each env's default.
    if not hasattr(action_term, "_default_joint_ids"):
        action_term._event_default_joint_pos = wp.from_torch(default_pos)
        action_term._default_joint_ids = action_term._joint_ids_wp
        if action_term._default_joint_ids is None:
            action_term._default_joint_ids = wp.array(
                range(action_term.action_dim), dtype=wp.int32, device=env.device
            )
        action_term.process_actions = MethodType(_process_per_env_actions, action_term)
    # Keep the official IO descriptor's env-0 offset consistent for export.
    wp.to_torch(action_term._offset)[:] = default_pos[0, action_term._joint_ids]
