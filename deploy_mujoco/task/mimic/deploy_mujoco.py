"""Sim2sim (MuJoCo) rollout for basic_loco_lib mimic policies (unitree_g1).

Robot-specific inputs (mj joint names, anchor body, PD/deploy arrays,
joint-order maps, viewer camera) live in ``<robot>/<robot>.yaml``; this script
is robot-agnostic -- ``deploy_mujoco.py unitree_g1`` loads ``unitree_g1/unitree_g1.yaml``, adding a
robot is a new directory, no code change.  Relative yaml paths resolve against
the yaml's own directory (xml points into the packaged assets; policy / motion
sit next to the yaml).

The obs/action pipeline mirrors the training cfgs exactly (obs layout, joint
orders, quat conventions -- see the robot READMEs for the full contract).

Usage 

    python deploy_mujoco.py unitree_g1
"""

import argparse
import copy
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
import yaml

# Shared viewer debug tools remain at the deployment root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mujoco_visualizer import MujocoNativeDebugVisualizer, NullDebugVisualizer

SCRIPT_DIR = Path(__file__).resolve().parent


# quat convention: MuJoCo is (w, x, y, z); Isaac Lab 3.0 -- and the motion npz
# it writes -- is (x, y, z, w).  Convert at every npz -> MuJoCo boundary.
def xyzw_to_wxyz(q):
    return q[[3, 0, 1, 2]]


def quat_to_mat(q_wxyz):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(q_wxyz))
    return rot.reshape(3, 3)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def viewer_setup(viewer, model, cam, track_body_name):
    """Apply the yaml 'viewer:' camera block; the camera tracks the anchor body."""
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
    parser = argparse.ArgumentParser(description="Deploy basic_loco_lib mimic policy in MuJoCo (sim2sim)")
    parser.add_argument("robot", help=f"robot dir, e.g. {find_robots()} -> loads <robot>/<robot>.yaml")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.robot / f"{args.robot}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] no config for robot '{args.robot}' (expected {config_path});"
                         f" known: {find_robots()}")
    config_dir = config_path.parent  # relative paths in the yaml resolve here
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    policy_path = config_dir / config["policy"]
    motion_path = config_dir / config["motion"]
    xml_path = config_dir / config["xml_path"]
    sim_dt = config["simulation_dt"]
    decimation = config["control_decimation"]
    # newton-variant checkpoints set history_length=4 on the policy obs group
    # (whole obs frames, oldest first); the warp task has none
    obs_history_length = int(config.get("obs_history_length", 1))

    # joint names in mj order (== mjcf joint order == order of every deploy
    # array below), plus the anchor body of the training MotionCommand
    mj_joint_names = [str(n) for n in config["mj_joint_names"]]
    anchor_body_name = str(config["anchor_body_name"])
    num_joints = len(mj_joint_names)

    backend = config["backend"]
    print(f"Policy backend (joint order): {backend}")
    MJ2LAB = np.asarray(config["joint_orders"][backend], dtype=int)
    LAB2MJ = np.argsort(MJ2LAB)
    joint_seq = [mj_joint_names[m] for m in MJ2LAB]  # lab/policy joint order

    kps = np.array(config["stiffness"], dtype=float)
    kds = np.array(config["damping"], dtype=float)
    default_dof = np.array(config["default_joint_pos"], dtype=float)
    action_scale = np.array(config["action_scale"], dtype=float)

    # one obs frame: motion_command(2n) | motion_anchor_ori_b(6) | base_ang_vel(3)
    # | joint_pos_rel(n) | joint_vel_rel(n) | last_action(n)
    frame_dim = 5 * num_joints + 9
    obs_dim = obs_history_length * frame_dim

    # motion npz (self-describing: joint_names/body_names -> align by name)
    print(f"Loading motion npz: {motion_path}")
    motion = np.load(motion_path)
    fps = float(np.asarray(motion["fps"]).item())
    npz_joint_names = [str(s) for s in motion["joint_names"]]
    perm_npz = [npz_joint_names.index(n) for n in joint_seq]
    ref_joint_pos = motion["joint_pos"][:, perm_npz]  # (T, n) lab order
    ref_joint_vel = motion["joint_vel"][:, perm_npz]
    root_body = [str(s) for s in motion["body_names"]].index(anchor_body_name)
    root_pos_w = motion["body_pos_w"][:, root_body, :]
    root_quat_w_xyzw = motion["body_quat_w"][:, root_body, :]
    root_lin_vel_w = motion["body_lin_vel_w"][:, root_body, :]
    root_ang_vel_w = motion["body_ang_vel_w"][:, root_body, :]
    num_frames = ref_joint_pos.shape[0]
    print(f"Motion: {num_frames} frames @ {fps:.0f} Hz ({num_frames / fps:.1f} s)")

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    print(f"Loading ONNX policy: {policy_path}")
    ort_session = onnxruntime.InferenceSession(str(policy_path))

    # spawn the robot exactly on reference motion frame 0 (training _resample
    # writes the robot onto the motion; here without the noise)
    d.qpos[0:3] = root_pos_w[0]
    d.qpos[3:7] = xyzw_to_wxyz(root_quat_w_xyzw[0])
    d.qpos[7 : 7 + num_joints] = ref_joint_pos[0][LAB2MJ]
    d.qvel[0:3] = root_lin_vel_w[0]  # free-joint linear vel is global
    d.qvel[3:6] = quat_to_mat(d.qpos[3:7]).T @ root_ang_vel_w[0]  # world -> body frame
    d.qvel[6 : 6 + num_joints] = ref_joint_vel[0][LAB2MJ]
    mujoco.mj_forward(m, d)

    target_dof_pos = d.qpos[7:].copy()
    action = np.zeros(num_joints, dtype=np.float32)
    # history of whole obs frames (oldest first), pre-filled with the first obs
    # like IsaacLab's CircularBuffer on reset (is_first_push fills every slot)
    obs_buffer = deque(maxlen=max(obs_history_length, 1))
    print(f"policy obs: {obs_dim} dims (history_length={obs_history_length})")

    timestep = 0
    frame = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer_setup(viewer, m, config["viewer"], anchor_body_name)
        # ghost of the reference pose
        visualizer = NullDebugVisualizer()
        ghost_model = None
        if getattr(viewer, "user_scn", None) is not None:
            visualizer = MujocoNativeDebugVisualizer(viewer.user_scn, m)
            ghost_model = copy.deepcopy(m)
            ghost_model.geom_rgba[:] = np.array([0.9, 0.1, 0.1, 0.5], dtype=np.float32)
            qpos_ghost = d.qpos.copy()

        start = time.time()
        while viewer.is_running() and frame < num_frames:
            step_start = time.time()

            if timestep % decimation == 0:
                joint_pos = d.qpos[7 : 7 + num_joints].copy()
                joint_vel = d.qvel[6 : 6 + num_joints].copy()

                # anchor = anchor_body_name on both sides (training cfg): ref
                # quat is that npz body's quat, robot quat is the base quat
                # qpos[3:7].  6D rep = first two columns of the relative
                # rotation, flattened row-major.
                rot_ref = quat_to_mat(xyzw_to_wxyz(root_quat_w_xyzw[frame]))
                rot_rel = quat_to_mat(d.qpos[3:7]).T @ rot_ref
                anchor_ori_6d = rot_rel[:, :2].ravel()

                obs = np.concatenate(
                    [
                        ref_joint_pos[frame],
                        ref_joint_vel[frame],
                        anchor_ori_6d,
                        d.qvel[3:6],  # body-frame angular velocity
                        (joint_pos - default_dof)[MJ2LAB],
                        joint_vel[MJ2LAB],
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

                # target = action * scale + default (lab -> mj order)
                target_dof_pos = action[LAB2MJ] * action_scale + default_dof

                if timestep % 500 == 0:
                    print(f"t={timestep * sim_dt:6.1f}s  frame={frame:5d}  wall={time.time() - start:6.1f}s")

                # ghost of the reference pose for the current frame
                if ghost_model is not None:
                    qpos_ghost[0:3] = root_pos_w[frame]
                    qpos_ghost[3:7] = xyzw_to_wxyz(root_quat_w_xyzw[frame])
                    qpos_ghost[7 : 7 + num_joints] = ref_joint_pos[frame][LAB2MJ]
                    visualizer.clear()
                    visualizer.add_ghost_mesh(qpos_ghost, model=ghost_model)
                frame += 1

            # PD control at physics rate
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

    print("\nSim2sim rollout complete!")


if __name__ == "__main__":
    main()
