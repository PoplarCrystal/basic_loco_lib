import gymnasium as gym

gym.register(
    id="Unitree-G1-AMP-WM-Warp",
    entry_point="basic_loco_lib.tasks.amp_wm.amp_wm_env_warp:AmpWmEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.amp_wm.agents.rsl_rl_amp_wm_cfg:g1_amp_wm_warp_runner_cfg",
    },
)
