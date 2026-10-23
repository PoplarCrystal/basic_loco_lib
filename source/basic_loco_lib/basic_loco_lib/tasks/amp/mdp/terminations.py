"""Torch delayed termination for standard PhysX and Newton AMP tasks."""

import torch

from isaaclab.managers import TerminationManager


class DelayedTerminationManager(TerminationManager):
    """Delay fall resets for recovery environments while preserving timeouts."""

    def __init__(self, base, delay_env_mask, max_delay_steps, joint_vel, max_joint_velocity=50.0):
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps
        self._joint_vel = joint_vel
        self._max_joint_velocity = max_joint_velocity

    def compute(self):
        dones = super().compute()
        if self._max_delay_steps <= 0:
            return dones
        unstable = self._delay_env_mask & (
            ~torch.isfinite(self._joint_vel).all(dim=-1)
            | (self._joint_vel.abs() > self._max_joint_velocity).any(dim=-1)
        )
        delayed = self._delay_env_mask & dones & ~unstable
        self._delay_counters.copy_(torch.where(delayed, self._delay_counters + 1, 0))
        suppress = delayed & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[suppress] = False
        self._terminated_buf[unstable] = True
        self._delay_counters.masked_fill_(~suppress, 0)
        return self.dones

    def reset(self, env_ids=None):
        extras = super().reset(env_ids)
        self._delay_counters[slice(None) if env_ids is None else env_ids] = 0
        return extras
