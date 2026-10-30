"""Original mimic Warp environment with world-model transition capture."""
from isaaclab_experimental.envs import ManagerBasedRLEnvWarp
from .bridge import MimicWmBridge


class MimicWmEnvWarp(MimicWmBridge, ManagerBasedRLEnvWarp):
    def _reset_idx(self, env_ids, *, env_mask=None):
        self._capture_wm_terminal(env_ids, env_mask)
        super()._reset_idx(env_ids, env_mask=env_mask)
        self.scene['robot'].data._update_body_state()
