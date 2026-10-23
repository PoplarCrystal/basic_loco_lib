"""MuJoCo debug visualizer for the sim2sim viewer (ghost rendering).

Trimmed from sonic's ``deploy_mujoco/tools/mjlab_utils`` (debug_visualizer.py +
visualizer.py): same class/method names and mjv_addGeoms technique, only the
ghost mesh is kept. The ghost is a deepcopy of the robot model with a
translucent geom_rgba; each call re-poses it and appends its geoms to the
passive viewer's user scene.
"""

from __future__ import annotations

import mujoco
import numpy as np


class MujocoNativeDebugVisualizer:
    """Draws debug geometry into the passive viewer's user scene."""

    def __init__(self, scn: mujoco.MjvScene, mj_model: mujoco.MjModel, env_idx: int = 0):
        self.scn = scn
        self.mj_model = mj_model
        self.env_idx = env_idx
        self._initial_geom_count = scn.ngeom

        self._vopt = mujoco.MjvOption()
        self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self._pert = mujoco.MjvPerturb()
        self._viz_data = mujoco.MjData(mj_model)

    def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
        """Add a semi-transparent copy of the robot geometry at the target pose.

        Transparency comes from the ghost model's geom_rgba (set at creation);
        alpha/label are kept for interface parity with sonic and are unused.
        """
        del alpha, label
        self._viz_data.qpos[:] = np.asarray(qpos, dtype=float)
        mujoco.mj_forward(model, self._viz_data)
        mujoco.mjv_addGeoms(
            model,
            self._viz_data,
            self._vopt,
            self._pert,
            mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
            self.scn,
        )

    def clear(self) -> None:
        """Drop all debug geoms added since construction."""
        self.scn.ngeom = self._initial_geom_count


class NullDebugVisualizer:
    """No-op visualizer for --no_viewer runs."""

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx

    def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
        pass

    def clear(self) -> None:
        pass
