"""Kinematic replay of a motion npz in MuJoCo (no policy, no physics).

Usage
    python play_motion.py unitree_g1       # interactive viewer, loops
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from deploy_mujoco import SCRIPT_DIR, find_robots, viewer_setup, xyzw_to_wxyz


def mj_joint_perm(config):
    """Convert the YAML-selected backend joint order into MuJoCo order."""
    backend = config["backend"]
    mj2lab = np.asarray(config["joint_orders"][backend], dtype=int)
    return np.argsort(mj2lab)


def set_state(d, npz, f, perm, root_body):
    d.qpos[:3] = npz["body_pos_w"][f, root_body]
    d.qpos[3:7] = xyzw_to_wxyz(npz["body_quat_w"][f, root_body])
    d.qpos[7:] = npz["joint_pos"][f][perm]
    d.qvel[:] = 0


def main():
    parser = argparse.ArgumentParser(description="Replay a mimic motion npz in MuJoCo (kinematic)")
    parser.add_argument("robot", help=f"robot dir, e.g. {find_robots()} -> loads <robot>/<robot>.yaml")
    parser.add_argument("--motion", type=Path, default=None, help="override the yaml motion path")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.robot / f"{args.robot}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] no config for robot '{args.robot}' (expected {config_path});"
                         f" known: {find_robots()}")
    config_dir = config_path.parent  # relative paths in the yaml resolve here
    with open(config_path) as f:
        config = yaml.safe_load(f)

    m = mujoco.MjModel.from_xml_path(str(config_dir / config["xml_path"]))
    d = mujoco.MjData(m)
    motion_path = args.motion if args.motion else config_dir / config["motion"]
    npz = np.load(motion_path)
    T = len(npz["joint_pos"])
    dt = 1.0 / float(npz["fps"])
    perm = mj_joint_perm(config)
    root_body = [str(s) for s in npz["body_names"]].index(config["anchor_body_name"])
    print(f"Replaying {motion_path.name}: {T} frames @ {1.0 / dt:.0f} Hz ({T * dt:.1f} s)")
    print(f"Motion backend (joint order): {config['backend']}")

    run_viewer(m, d, npz, T, dt, perm, root_body, config)


def run_viewer(m, d, npz, T, dt, perm, root_body, config):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(m, d) as v:
        viewer_setup(v, m, config["viewer"], config["anchor_body_name"])
        while v.is_running():
            for f in range(T):
                if not v.is_running():
                    return
                set_state(d, npz, f, perm, root_body)
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(dt)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
