"""Official Warp MDP terms plus velocity-specific reward kernels."""
from isaaclab_experimental.envs.mdp import *  # noqa: F401,F403
from isaaclab_tasks_experimental.core.velocity.mdp import *  # noqa: F401,F403
from isaaclab_tasks_experimental.core.locomotion.mdp import *  # noqa: F401,F403

from .terminations import bad_orientation
from .rewards import base_height_l2, energy, feet_gait, foot_clearance_reward

from isaaclab_experimental.envs.frontend import WarpFrontend as _WarpFrontend

if __name__ not in _WarpFrontend.WARP_ROOT_PREFIXES:
    _WarpFrontend.WARP_ROOT_PREFIXES += (__name__,)
