"""Warp-native execution noise; policy and processed-action buffers stay clean."""

import warp as wp
from isaaclab_experimental.envs.mdp.actions.joint_actions import JointPositionAction


@wp.kernel
def _execution_targets(
    processed: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    targets: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    targets[i, j] = processed[i, j] + noise[i, j] * scale[j]


class MimicWmJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._execution_noise = wp.zeros_like(self._raw_actions)
        self.execution_noise = wp.to_torch(self._execution_noise)
        self._execution_targets = wp.zeros_like(self._processed_actions)

    def apply_actions(self):
        wp.launch(
            _execution_targets,
            dim=self._processed_actions.shape,
            inputs=[self._processed_actions, self._execution_noise, self._scale, self._execution_targets],
            device=self.device,
        )
        self._asset.set_joint_position_target_index(
            target=self._execution_targets, joint_ids=self._joint_ids_wp
        )
