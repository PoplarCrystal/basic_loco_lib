"""Sim2sim (MuJoCo) rollout for basic_loco_lib velocity policies (unitree_g1).

Robot-specific paths, PD parameters, joint orders and camera settings live in
``<robot>/<robot>.yaml``. Relative yaml paths resolve against that directory.

Usage
    python deploy_mujoco.py unitree_g1
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.velocity_command_panel import VelocityCommandPanel

SCRIPT_DIR = Path(__file__).resolve().parent


def quat_to_mat(q_wxyz):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(q_wxyz))
    return rot.reshape(3, 3)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def viewer_setup(viewer, model, cam, track_body_name):
    """Apply the yaml 'viewer:' camera block; track the root body."""
    viewer.cam.distance = cam["distance"]
    viewer.cam.azimuth = cam["azimuth"]
    viewer.cam.elevation = cam["elevation"]
    viewer.cam.lookat[:] = np.asarray(cam["lookat"], dtype=float)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)


def find_robots():
    """Robot dirs shipping a <name>/<name>.yaml next to this script."""
    return sorted(p.parent.name for p in SCRIPT_DIR.glob("*/*.yaml") if p.parent.name == p.stem)


def main():
    parser = argparse.ArgumentParser(description="Deploy basic_loco_lib velocity policy in MuJoCo (sim2sim)")
    parser.add_argument("robot", help=f"robot dir, e.g. {find_robots()} -> loads <robot>/<robot>.yaml")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.robot / f"{args.robot}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] no config for robot '{args.robot}' (expected {config_path});"
                         f" known: {find_robots()}")
    config_dir = config_path.parent
    with open(config_path) as f:
        config = yaml.safe_load(f)

    policy_path = config_dir / config["policy"]
    xml_path = config_dir / config["xml_path"]
    sim_dt = config["simulation_dt"]
    decimation = int(config["control_decimation"])
    obs_history_length = int(config.get("obs_history_length", 1))
    command = np.zeros(3, dtype=np.float32)

    mj_joint_names = [str(n) for n in config["mj_joint_names"]]
    root_body_name = str(config["root_body_name"])
    num_joints = len(mj_joint_names)
    backend = config["backend"]
    print(f"Policy backend (joint order): {backend}")
    MJ2LAB = np.asarray(config["joint_orders"][backend], dtype=int)
    LAB2MJ = np.argsort(MJ2LAB)

    kps = np.array(config["stiffness"], dtype=float)
    kds = np.array(config["damping"], dtype=float)
    default_dof = np.array(config["default_joint_pos"], dtype=float)
    action_scale = np.array(config["action_scale"], dtype=float)

    # One obs frame: command(3) | projected_gravity(3) | base_ang_vel(3)
    # | joint_pos_rel(n) | joint_vel_rel(n) | last_action(n).
    frame_dim = 3 * num_joints + 9
    obs_dim = obs_history_length * frame_dim

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.timestep = sim_dt
    d = mujoco.MjData(m)

    print(f"Loading ONNX policy: {policy_path}")
    ort_session = onnxruntime.InferenceSession(str(policy_path))

    # Spawn at the velocity task's default pose, without reset noise.
    d.qpos[0:3] = config["initial_pos"]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7 : 7 + num_joints] = default_dof
    mujoco.mj_forward(m, d)

    target_dof_pos = default_dof.copy()
    action = np.zeros(num_joints, dtype=np.float32)
    obs_buffer = deque(maxlen=obs_history_length)
    print(f"policy obs: {obs_dim} dims (history_length={obs_history_length}); command={command.tolist()}")
    timestep = 0

    with VelocityCommandPanel(command, config["command_ranges"]) as command_panel, \
            mujoco.viewer.launch_passive(m, d) as viewer:
        viewer_setup(viewer, m, config["viewer"], root_body_name)
        start = time.time()
        while viewer.is_running() and command_panel.running:
            step_start = time.time()

            if timestep % decimation == 0:
                command[:] = command_panel.read()
                if not command_panel.running:
                    break
                joint_pos = d.qpos[7 : 7 + num_joints].copy()
                joint_vel = d.qvel[6 : 6 + num_joints].copy()
                projected_gravity = quat_to_mat(d.qpos[3:7]).T @ np.array([0.0, 0.0, -1.0])
                obs = np.concatenate(
                    [
                        command,
                        projected_gravity,
                        d.qvel[3:6] * config["ang_vel_scale"],
                        (joint_pos - default_dof)[MJ2LAB],
                        joint_vel[MJ2LAB] * config["joint_vel_scale"],
                        action,
                    ]
                )
                if not obs_buffer:
                    for _ in range(obs_history_length):
                        obs_buffer.append(obs)
                else:
                    obs_buffer.append(obs)

                policy_input = np.concatenate(obs_buffer).astype(np.float32)
                outputs = ort_session.run(
                    output_names=["actions"],
                    input_feed={"obs_history": policy_input.reshape(1, -1)},
                )
                action = outputs[0][0]
                target_dof_pos = action[LAB2MJ] * action_scale + default_dof
                if timestep % 500 == 0:
                    print(f"t={timestep * sim_dt:6.1f}s  wall={time.time() - start:6.1f}s")

            # PD control at physics rate.
            tau = pd_control(
                target_dof_pos, d.qpos[7:], kps,
                np.zeros_like(d.qvel[6:]), d.qvel[6:], kds,
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            timestep += 1

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print(f"\nSim2sim rollout complete: {timestep} physics steps, {d.time:.3f} s")


if __name__ == "__main__":
    main()
