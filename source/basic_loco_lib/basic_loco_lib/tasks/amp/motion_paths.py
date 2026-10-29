"""Configure AMP expert and reset data from one motion directory."""

from pathlib import Path


def set_motion_directory(env_cfg, agent_cfg, motion_dir):
    root = Path(motion_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"AMP --motion_file must be a directory: {root}")
    walk_run = root / "WalkandRun" if (root / "WalkandRun").is_dir() else root
    recovery = root / "Recovery"
    agent_cfg.amp_motion_files = str(root)
    env_cfg.events.init_motion_loader.params["motion_dir"] = str(walk_run)
    env_cfg.events.reset_from_motion.params["motion_dir"] = str(walk_run)
    env_cfg.events.init_motion_loader.params["recovery_dir"] = str(recovery) if recovery.is_dir() else None
    if not recovery.is_dir():
        env_cfg.events.init_motion_loader.params["delay_reset_env_ratio"] = 0.0
