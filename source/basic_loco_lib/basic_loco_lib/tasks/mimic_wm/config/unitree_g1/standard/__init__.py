import gymnasium as gym

gym.register(
    id="Unitree-G1-Mimic-WM",
    entry_point="basic_loco_lib.tasks.mimic_wm.mimic_wm_env:MimicWmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.mimic_wm.agents.rsl_rl_mimic_wm_cfg:MimicWmRunnerCfg",
    },
)
