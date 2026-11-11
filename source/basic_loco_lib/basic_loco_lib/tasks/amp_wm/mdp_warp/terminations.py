from __future__ import annotations

import torch

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers.termination_manager import TerminationManager

from .warp_utils import quat_rotate_inv_xyzw

from .utils import _wp


@wp.kernel
def _apply_delayed_termination(
    delay_env_mask: wp.array(dtype=wp.bool),
    delay_counters: wp.array(dtype=wp.int64),
    truncated: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    dones: wp.array(dtype=wp.bool),
    max_delay_steps: int,
):
    env_id = wp.tid()
    if delay_env_mask[env_id] and dones[env_id]:
        delay_counters[env_id] = delay_counters[env_id] + wp.int64(1)
        if delay_counters[env_id] < max_delay_steps:
            terminated[env_id] = False
            dones[env_id] = truncated[env_id]
        else:
            delay_counters[env_id] = wp.int64(0)
    elif delay_env_mask[env_id]:
        delay_counters[env_id] = wp.int64(0)


class DelayedTerminationManagerWarp(TerminationManager):
    """TerminationManager subclass that delays reset for a subset of envs.

    For delay envs, when a termination is triggered the reset signal is
    suppressed and a counter starts incrementing. Once the counter reaches
    ``max_delay_steps``, the reset signal is released and the counter resets.
    """

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # Steal all internal state from the base manager (avoid re-init).
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps
        self._delay_env_mask_wp = wp.from_torch(self._delay_env_mask)
        self._delay_counters_wp = wp.from_torch(self._delay_counters)

    def compute(self) -> torch.Tensor:
        dones = super().compute()

        if self._max_delay_steps <= 0:
            return dones

        # Keep all updates on Warp buffers so this remains CUDA-graph capturable.
        wp.launch(
            _apply_delayed_termination,
            dim=self.num_envs,
            inputs=[
                self._delay_env_mask_wp,
                self._delay_counters_wp,
                self._truncated_wp,
                self._terminated_wp,
                self._dones_wp,
                self._max_delay_steps,
            ],
            device=self.device,
        )
        return self._dones_tensor_view

    def reset(self, env_ids=None, *, env_mask=None):
        extras = super().reset(env_ids, env_mask=env_mask)
        mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        wp.launch(
            _reset_delay_counters,
            dim=self.num_envs,
            inputs=[mask, self._delay_counters_wp],
            device=self.device,
        )
        return extras


@wp.kernel
def _reset_delay_counters(mask: wp.array(dtype=wp.bool), counters: wp.array(dtype=wp.int64)):
    i = wp.tid()
    if mask[i]:
        counters[i] = wp.int64(0)


@wp.kernel
def _bad_orientation(
    root_pose_w: wp.array(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    limit_angle: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    quat = wp.transform_get_rotation(root_pose_w[i])
    quat_xyzw = wp.vec4f(quat[0], quat[1], quat[2], quat[3])
    projected_gravity = quat_rotate_inv_xyzw(quat_xyzw, wp.normalize(gravity_w[i]))
    out[i] = wp.acos(wp.clamp(-projected_gravity[2], -1.0, 1.0)) > limit_angle


def bad_orientation(env, out, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    wp.launch(
        _bad_orientation,
        dim=env.num_envs,
        inputs=[_wp(asset.data.root_link_pose_w), _wp(asset.data.GRAVITY_VEC_W), limit_angle, out],
        device=env.device,
    )
