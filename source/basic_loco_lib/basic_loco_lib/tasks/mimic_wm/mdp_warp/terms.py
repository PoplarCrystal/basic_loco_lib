"""Warp output-buffer adapter for the differentiable reward."""
import warp as wp
from isaaclab_experimental.utils.warp import warp_capturable
from ..mdp.terms import mimic_wm_reward as reward


@warp_capturable(False)
def mimic_wm_reward(env, out):
    wp.to_torch(out).copy_(reward(env))
