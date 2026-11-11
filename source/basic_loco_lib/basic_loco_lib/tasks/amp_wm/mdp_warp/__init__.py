"""Warp-native MDP terms owned by the AMP world-model task."""

from isaaclab_experimental.envs.frontend import WarpFrontend as _WarpFrontend

if __name__ not in _WarpFrontend.WARP_ROOT_PREFIXES:
    _WarpFrontend.WARP_ROOT_PREFIXES += (__name__,)

from .actions import AmpWmJointPositionAction
from .commands import *
from .events import *
from .observations import *
from .obs_history import make_history_obs, resolve_dones
from .rewards import *
from .terminations import *

__all__ = ["AmpWmJointPositionAction", "make_history_obs", "resolve_dones"]
