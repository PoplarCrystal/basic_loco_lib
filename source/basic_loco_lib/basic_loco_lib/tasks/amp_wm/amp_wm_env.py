"""Standard (PhysX / Newton) AMP environment with world-model transition capture."""

from isaaclab.envs import ManagerBasedRLEnv

from .bridge import AmpWmBridge


class AmpWmEnv(AmpWmBridge, ManagerBasedRLEnv):
    def _reset_idx(self, env_ids):
        self._capture_wm_terminal(env_ids)
        super()._reset_idx(env_ids)
        # Publish body poses/velocities after all root and joint reset writes.
        self.sim.forward()


__all__ = ["AmpWmEnv"]
