"""Original mimic environment with world-model transition capture."""
from isaaclab.envs import ManagerBasedRLEnv
from .bridge import MimicWmBridge


class MimicWmEnv(MimicWmBridge, ManagerBasedRLEnv):
    def _reset_idx(self, env_ids):
        self._capture_wm_terminal(env_ids)
        super()._reset_idx(env_ids)
        self.sim.forward()
