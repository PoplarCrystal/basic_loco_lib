"""Execution noise leaves the policy action and observation buffers clean."""

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class AmpWmJointPositionAction(JointPositionAction):
    """Joint position action whose executed setpoint carries world-model noise.

    ``execution_noise`` is filled per step by :class:`AmpWmBridge`; the
    processed action, the observation's last-action slot and ``a_prev`` stay
    clean, so the networks never see the noise.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.execution_noise = torch.zeros_like(self._raw_actions)

    def apply_actions(self):
        processed = self._processed_actions
        self._processed_actions = processed + self.execution_noise * self._scale
        try:
            super().apply_actions()
        finally:
            self._processed_actions = processed


__all__ = ["AmpWmJointPositionAction"]
