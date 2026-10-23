"""Register Warp velocity environments."""
import gymnasium as gym

for task_id, env_cfg, play_cfg in (
    ('Unitree-G1-Velocity-Warp', 'RobotEnvCfg', 'RobotPlayEnvCfg'),
    ('Unitree-G1-Velocity-Flat-Warp', 'RobotFlatEnvCfg', 'RobotFlatPlayEnvCfg'),
):
    gym.register(
        id=task_id,
        entry_point='isaaclab_experimental.envs:ManagerBasedRLEnvWarp',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.velocity_env_cfg:{env_cfg}',
            'play_env_cfg_entry_point': f'{__name__}.velocity_env_cfg:{play_cfg}',
            'rsl_rl_cfg_entry_point': 'basic_loco_lib.tasks.velocity.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
        },
    )
