"""Execution noise leaves the original mimic action and observation buffers clean."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class MimicWmJointPositionAction(JointPositionAction):
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
