"""Reward terms exposing the differentiable SuperMimic objective to the env.

The environment reward must equal the reward the world-model learner optimises
in imagination, so these terms read the components of
:func:`basic_loco_lib.tasks.mimic_wm.diff.mdp._reward_terms`, which is shared
with the Warp task, instead of re-implementing the tracking formula.
``RewardManager`` applies each term's weight and ``dt``; the config weights
mirror ``RewardCfg`` in ``diff/mdp.py``, so their weighted sum reproduces the
training reward exactly.
"""
from ..diff.mdp import _reward_terms


def wm_motion_global_anchor_pos(env):
    return _reward_terms(env)["motion_global_anchor_pos"]


def wm_motion_global_anchor_ori(env):
    return _reward_terms(env)["motion_global_anchor_ori"]


def wm_motion_body_pos(env):
    return _reward_terms(env)["motion_body_pos"]


def wm_motion_body_ori(env):
    return _reward_terms(env)["motion_body_ori"]


def wm_motion_body_lin_vel(env):
    return _reward_terms(env)["motion_body_lin_vel"]


def wm_motion_body_ang_vel(env):
    return _reward_terms(env)["motion_body_ang_vel"]


def wm_action_rate(env):
    return _reward_terms(env)["action_rate"]


def wm_joint_limit(env):
    return _reward_terms(env)["joint_limit"]


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
