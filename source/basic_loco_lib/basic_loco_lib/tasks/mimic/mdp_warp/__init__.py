"""Warp-kernel MDP terms for the mimic task.

The terms mirror ``basic_loco_lib.tasks.mimic.mdp`` one-to-one; everything
that has an official Warp-first override in
:mod:`isaaclab_experimental.envs.mdp` is used from there, and only the
motion-command-dependent observations / rewards / terminations plus the
Warp-consumable :class:`MotionCommandWarp` live here.
"""

from .commands import MotionCommandWarp, MotionCommandWarpCfg
from .events import randomize_joint_default_pos
from .obs_history import make_history_obs, resolve_dones
from .observations import motion_anchor_ori_b, motion_anchor_pos_b, robot_body_ori_b, robot_body_pos_b
from .rewards import (
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from .terminations import bad_anchor_ori, bad_anchor_pos_z_only, bad_motion_body_pos_z_only

# Declare this package as Warp-native while preserving real import paths for
# config serialization. The frontend identifies native terms by package prefix.
from isaaclab_experimental.envs.frontend import WarpFrontend as _WarpFrontend

_native_prefix = __name__
if _native_prefix not in _WarpFrontend.WARP_ROOT_PREFIXES:
    _WarpFrontend.WARP_ROOT_PREFIXES += (_native_prefix,)
