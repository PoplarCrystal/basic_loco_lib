"""Standard AMP environment supporting PhysX and Newton physics."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


class AmpEnv(ManagerBasedRLEnv):
    """Preserve AMP observations before automatic resets for terminal transitions."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.extras["terminal_amp_observations"] = torch.empty(
            (self.num_envs, *self.observation_manager.group_obs_dim["amp"]), device=self.device
        )

    def _reset_idx(self, env_ids):
        self.extras["terminal_amp_observations"].copy_(self.observation_manager.compute_group("amp"))
        super()._reset_idx(env_ids)
        # Resolve kinematics after all root/joint reset writes on either backend.
        self.sim.forward()
