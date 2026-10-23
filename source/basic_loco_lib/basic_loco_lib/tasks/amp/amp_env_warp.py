"""AMP environment that preserves observations before automatic resets."""

import torch

from isaaclab_experimental.envs import ManagerBasedRLEnvWarp


class AmpEnvWarp(ManagerBasedRLEnvWarp):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.extras["terminal_amp_observations"] = torch.empty(
            (self.num_envs, *self.observation_manager.group_obs_dim["amp"]), device=self.device
        )

    def _reset_idx(self, env_ids, *, env_mask=None):
        # compute_group does not advance actor/critic history; copy its persistent
        # output before reset and the next observation computation overwrite it.
        self.extras["terminal_amp_observations"].copy_(self.observation_manager.compute_group("amp"))
        super()._reset_idx(env_ids, env_mask=env_mask)
        # Publish body poses/velocities after all root and joint reset writes.
        self.scene["robot"].data._update_body_state()
