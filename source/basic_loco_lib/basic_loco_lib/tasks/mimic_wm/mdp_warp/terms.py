"""Warp output-buffer adapters for the differentiable SuperMimic reward terms."""
import warp as wp
from isaaclab_experimental.utils.warp import warp_capturable
from ..diff.mdp import _reward_terms


def _copy(env, out, name):
    wp.to_torch(out).copy_(_reward_terms(env)[name])


@warp_capturable(False)
def wm_motion_global_anchor_pos(env, out):
    _copy(env, out, "motion_global_anchor_pos")


@warp_capturable(False)
def wm_motion_global_anchor_ori(env, out):
    _copy(env, out, "motion_global_anchor_ori")


@warp_capturable(False)
def wm_motion_body_pos(env, out):
    _copy(env, out, "motion_body_pos")


@warp_capturable(False)
def wm_motion_body_ori(env, out):
    _copy(env, out, "motion_body_ori")


@warp_capturable(False)
def wm_motion_body_lin_vel(env, out):
    _copy(env, out, "motion_body_lin_vel")


@warp_capturable(False)
def wm_motion_body_ang_vel(env, out):
    _copy(env, out, "motion_body_ang_vel")


@warp_capturable(False)
def wm_action_rate(env, out):
    _copy(env, out, "action_rate")


@warp_capturable(False)
def wm_joint_limit(env, out):
    _copy(env, out, "joint_limit")


__all__ = [
    "wm_action_rate",
    "wm_joint_limit",
    "wm_motion_body_ang_vel",
    "wm_motion_body_lin_vel",
    "wm_motion_body_ori",
    "wm_motion_body_pos",
    "wm_motion_global_anchor_ori",
    "wm_motion_global_anchor_pos",
]
