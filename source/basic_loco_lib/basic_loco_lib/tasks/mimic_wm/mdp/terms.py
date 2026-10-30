"""SuperMimic-style reward evaluated on the original mimic environment."""
import torch
import warp as wp


def as_torch(value):
    return value if torch.is_tensor(value) else wp.to_torch(value)


def mimic_wm_reward(env):
    mdp = env.mimic_wm_mdp
    frames = (env.command_manager.get_term('motion').time_steps + 1).clamp(max=mdp.motion.num_frames-1)
    # RewardManager applies dt; DiffMDP already returns a dt-weighted reward.
    return mdp.reward(env._wm_state(), frames, as_torch(env.action_manager.action),
                      as_torch(env.action_manager.prev_action)) / env.step_dt
