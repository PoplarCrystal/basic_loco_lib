"""Array adapters and recovery state for AMP Warp terms."""

import torch

import warp as wp


def _wp(value):
    if isinstance(value, wp.array):
        return value
    if hasattr(value, "warp"):
        return value.warp
    if torch.is_tensor(value):
        if not value.is_contiguous():
            value = value.contiguous()
        return wp.from_torch(value)
    return wp.from_torch(wp.to_torch(value))


def _delay_state(env):
    """Return delay mask/counters for recovery environments, or zero state."""
    manager = env.termination_manager
    if hasattr(manager, "_delay_env_mask"):
        return _wp(manager._delay_env_mask), _wp(manager._delay_counters), True
    if not hasattr(env, "_amp_zero_delay_state"):
        env._amp_zero_delay_state = (
            wp.zeros(env.num_envs, dtype=wp.bool, device=env.device),
            wp.zeros(env.num_envs, dtype=wp.int64, device=env.device),
        )
    return *env._amp_zero_delay_state, False
