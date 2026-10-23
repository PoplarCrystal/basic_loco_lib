"""Register standard AMP environments (PhysX / Newton)."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-AMP-Loco",
    entry_point="basic_loco_lib.tasks.amp.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.amp.agents.rsl_rl_ppo_cfg:g1_amp_ppo_runner_cfg",
    },
)
