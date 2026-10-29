import gymnasium as gym

gym.register(
    id="Unitree-G1-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.velocity.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.velocity.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
