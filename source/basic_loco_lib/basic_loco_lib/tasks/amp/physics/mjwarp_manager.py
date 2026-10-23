"""Project-specific MuJoCo Warp solver managers.

History: this package used to host NewtonMJWarpStableManager — the zsl port of
mujoco_warp PR #1488 (pivot-floored Cholesky + unchanged-Hessian reuse), hard-pinned
to mujoco-warp==3.11.0. The newton 1.7.0 stack supersedes it (stable numerics without
the patch); the code was removed from this repo and remains archived in
basic_loco_lib's tasks/amp/physics plus doc/newton.md there.
"""

import logging

import mujoco_warp as mjw
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from newton import Model
from newton.solvers import SolverMuJoCo

logger = logging.getLogger(__name__)


class NewtonMJWarpQuietLSManager(NewtonMJWarpManager):
    """Default MJWarp Newton manager with only the linesearch-exhaustion print masked.

    The URDF-imported G1 (capsule feet + self-collisions) legitimately exhausts the
    ls_iterations budget in a small fraction of envs each step (~0.015 warnings per
    env-iter at ls_iterations=15, regardless of tolerance/njmax — measured on the
    newton 1.7.0 stack). mujoco-warp handles the event (uses the best alpha found)
    and prints "linesearch iterations limit reached" for every occurrence.

    This masks just the LS_ITERATIONS bit of ``warn_overflow`` after solver
    creation -- solver behavior is untouched and every other overflow warning
    (NEFC, NJMAX_NNZ, ... which DO indicate physics capping) stays visible.
    MJWarpSolverCfg does not expose ``warn_overflow``, hence this subclass.
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
