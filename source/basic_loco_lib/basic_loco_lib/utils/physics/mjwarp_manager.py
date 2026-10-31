"""Shared MuJoCo Warp solver managers for AMP and mimic tasks."""

import logging

import mujoco_warp as mjw
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from newton import Model
from newton.solvers import SolverMuJoCo

logger = logging.getLogger(__name__)


class NewtonMJWarpQuietLSManager(NewtonMJWarpManager):
    """Suppress the line-search iteration-limit warning.

    Solver behavior is unchanged; all other warnings remain enabled.
    """

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverMuJoCo:
        solver = super()._create_solver(model, solver_cfg)
        solver.mjw_model.opt.warn_overflow &= ~int(mjw.OverflowType.LS_ITERATIONS)
        logger.info(
            "Quiet MJWarp Newton: LS_ITERATIONS print masked (warn_overflow=0x%x)",
            solver.mjw_model.opt.warn_overflow,
        )
        return solver
