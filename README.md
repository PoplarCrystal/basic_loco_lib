# Basic Loco Lib

English | [中文](README_zh.md)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://github.com/isaac-sim/IsaacLab)

## Overview

`basic_loco_lib` is a basic robot locomotion library built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). It includes **Velocity (velocity tracking), Mimic (motion imitation), and AMP (Adversarial Motion Priors)** tasks for training, playing back, and deploying locomotion policies. The library currently focuses on the **Unitree G1 (29 DoF)**, with Newton (MuJoCo-Warp) as the primary physics backend and PhysX support for selected tasks.

The project uses `rsl_rl` for PPO training and provides a complete workflow from motion data preparation and policy training to ONNX export and MuJoCo sim2sim validation, making it easier to reproduce basic tasks and develop new gaits.

| Task | Task IDs | Supported backends | Description |
|---|---|---|---|
| Velocity | `Unitree-G1-Velocity`, `Unitree-G1-Velocity-Flat` | Newton / PhysX | Track forward, lateral, and yaw velocity commands (vx, vy, wz), with rough-terrain and flat-plane variants |
| Velocity Warp | `Unitree-G1-Velocity-Warp`, `Unitree-G1-Velocity-Flat-Warp` | Newton | Velocity tracking tasks optimized with Warp |
| Mimic | `Unitree-G1-Mimic` | Newton / PhysX | Track reference motions stored as NPZ files converted from CSV |
| Mimic Warp | `Unitree-G1-Mimic-Warp` | Newton | Motion tracking optimized with Warp |
| AMP | `Unitree-G1-AMP-Loco` | Newton / PhysX | Learn locomotion styles through adversarial motion priors using velocity commands and a motion dataset |
| AMP Warp | `Unitree-G1-AMP-Loco-Warp` | Newton | AMP locomotion optimized with Warp |

## Environment setup

### Miniconda

This project targets **Isaac Lab `v3.0.0`, Isaac Sim `6.2.0.0`, and Python `3.12`**. Pin the Isaac Lab tag for a reproducible environment.

```bash
conda create -n basic-loco-lib python=3.12 -y
conda activate basic-loco-lib

# Install the Isaac Sim runtime matching Isaac Lab v3.0.0
pip install "isaacsim[all,extscache]==6.2.0.0" \
    --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab and its dependencies
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v3.0.0
./isaaclab.sh -i

# Return to this project's root and install the task library
cd /path/to/basic_loco_lib
pip install -e source/basic_loco_lib/
```

The default Newton `1.6.1` stack can run the tasks in this repository. Long training runs on rough terrain require upgrading to Newton `1.7.0` to fix NaN issues. The upgrade also requires an Isaac Lab compatibility patch; see the [Newton environment guide](doc/newton.md) for instructions.

### Physics backend selection

For standard tasks, use `presets=newton` or `presets=physx` to select the backend for training, playback, and export; Newton is the default. Warp tasks use a fixed Newton backend and do not need a `presets` argument.

Run the following commands from the project root with the Conda environment activated, except where the deployment section specifies another directory. Standard examples set the backend explicitly; Warp examples omit it. Keep data preparation, training, and playback on the same backend.

## 1. Data preparation: CSV → NPZ (Mimic)

Velocity does not require motion data. Mimic requires a reference motion NPZ matching the training backend. Store CSV and NPZ files under `data/<robot>/`; converted files are saved beside the input CSV by default.

```bash
# Generate reference motions for Newton / Warp tasks
python scripts/mimic/csv_to_npz.py -f data/unitree_g1/custom_motion.csv \
    --input_fps 30 --robot unitree_g1 --backend newton --viz none
```

`custom_motion.csv` is a placeholder for your own motion file; replace it with the actual path. Output filenames include the backend, for example `custom_motion.newton.npz` or `custom_motion.physx.npz`.

**Note:** `--backend` determines the joint and rigid-body ordering in the NPZ and must match the training task's backend. If a standard task uses `presets=physx`, generate its data with `--backend physx` and load the corresponding `*.physx.npz`. Newton training uses `*.newton.npz` generated with `--backend newton`. Incorrect ordering can silently break training, for example with episode length stuck at 1 and reward at 0. NPZ files include `ordering`, `joint_names`, and `body_names` fields for inspection.

Each row of a custom CSV must contain 36 columns: `base_pos(3) + base_quat xyzw(4) + 29 joint angles`. Joint angles must follow the SDK ordering in [robots/unitree.py](source/basic_loco_lib/basic_loco_lib/robots/unitree.py). Use `--motion_file` to select the converted NPZ for training. To customize the task configuration, use an existing task directory as a reference, edit `tracking_env_cfg.py`, and register a new task ID.

AMP motion data lives under `data/amp/`: `WalkandRun/` supplies the style reward, and `Recovery/` supports delayed-recovery resets. When using Newton, the accompanying dataset does not require the CSV conversion above. Use `--motion_file <motion_directory>` to select another dataset.

Motion data sources and license information: [data/amp/README.md](data/amp/README.md).

## 2. Training

All examples below use Newton. **For standard tasks (IDs without the `-Warp` suffix), replace `presets=newton` with `presets=physx` to use PhysX. Warp tasks support only Newton.** When switching the Mimic backend, also update the data generation `--backend` and the NPZ loaded for training.

### Velocity: velocity tracking

```bash
# Newton: rough terrain with terrain and velocity command curricula
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity --viz none presets=newton

# Newton: flat plane
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Flat --viz none presets=newton

# Newton / Warp: corresponding optimized variants
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Warp --viz none
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Flat-Warp --viz none
```

Flat-plane variants do not use height scanning or a terrain curriculum.

### Mimic: motion imitation

```bash
# Newton: standard variant
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic --viz none presets=newton \
    --motion_file /path/to/custom_motion.newton.npz

# Newton / Warp: optimized variant
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic-Warp --viz none \
    --motion_file /path/to/custom_motion.newton.npz
```

### AMP: Adversarial Motion Priors

```bash
# Newton: standard variant, using WalkandRun and Recovery under data/amp/ by default
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco --viz none presets=newton

# Newton / Warp: optimized variant
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco-Warp --viz none

# Select a custom motion dataset directory
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco --viz none presets=newton \
    --motion_file /path/to/motion_dir
```

**Note:** The default datasets in `data/amp/WalkandRun/` and `data/amp/Recovery/` are generated for Newton. To use the PhysX backend, regenerate the motion data for PhysX.

## 3. Playback and ONNX export

Use `play.py` to load a trained checkpoint. **Keep the task, physics backend, and Mimic reference motion consistent with training.** The examples below use Newton:

```bash
# Mimic: playback / export only
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/ckpt.pt --export_only --num_envs 1

# Velocity: playback / export only
python scripts/rsl_rl/play.py --task Unitree-G1-Velocity-Flat-Warp \
    --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-Velocity-Flat-Warp \
    --checkpoint /path/to/ckpt.pt --export_only --num_envs 1

# AMP: playback / export only
python scripts/rsl_rl/play.py --task Unitree-G1-AMP-Loco presets=newton \
    --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-AMP-Loco presets=newton \
    --checkpoint /path/to/ckpt.pt --export_only --num_envs 1
```

- `--viz viser` opens a lightweight 3D viewer; use `--viz none` explicitly for headless runs.
- `--export_only` exports the policy and exits without entering the playback loop. Normal playback also exports the policy.
- Without `--checkpoint`, a checkpoint is selected automatically from the latest run under `logs/rsl_rl/<experiment_name>/`.
- To play back a PhysX policy, use the standard task ID from training and `presets=physx`. For Mimic, also select the NPZ generated for PhysX.
- If AMP was trained with a custom dataset, also pass `--motion_file /path/to/motion_dir` during playback or export.

Exported files are saved in `<checkpoint_directory>/exported/`, including `policy.pt` and a single self-contained `policy.onnx`. The ONNX input is `obs_history`, ready for the MuJoCo deployment workflow below.

## 4. Deployment: MuJoCo sim2sim

Deployment scripts live under `deploy_mujoco/task/<task>/`. Robot configuration files are stored as `<task>/<robot>/<robot>.yaml` and define PD gains, joint-order mappings, observation layouts, and policy paths. Place the exported `policy.onnx` beside the corresponding YAML. Mimic also requires a matching reference motion NPZ; check the motion path in the YAML. The deployment configuration's `backend` must match the policy's training backend (`newton` or `physx`) to select the correct joint mapping.

Install deployment and export dependencies from the project root:

```bash
pip install -e "source/basic_loco_lib[deploy]"
```

Each command group below starts from the project root and enters the corresponding task directory.

Velocity: control vx / vy / wz commands through a Tk slider panel.

```bash
cd deploy_mujoco/task/velocity
python deploy_mujoco.py unitree_g1
```

AMP: use the same velocity command panel, with command ranges matching the AMP training configuration.

```bash
cd deploy_mujoco/task/amp
python deploy_mujoco.py unitree_g1
```

Mimic: initialize the robot from frame 0 of the reference motion and track it.

```bash
cd deploy_mujoco/task/mimic
python deploy_mujoco.py unitree_g1
```

After retraining, export the policy with `--export_only` and update `policy.onnx` in the corresponding deployment directory. Observation order, history length, scaling, and joint mappings must match the training configuration; see each task's `deploy_mujoco.py` docstring for details. To add another robot, use the existing directories as a reference and provide its YAML configuration, model, and policy files.

## Acknowledgments

This project builds on ideas and implementations from the following open-source projects. We thank their authors and communities for their contributions:

- [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Unitree RL Lab (isaaclab3.0 branch)](https://github.com/unitreerobotics/unitree_rl_lab/tree/isaaclab3.0)
- [Whole-Body Tracking](https://github.com/HybridRobotics/whole_body_tracking)

Motion data acknowledgments:

- [BONES-SEED / Bones Studio](https://bones.studio/datasets/seed): source of WalkandRun motions. Training data includes Motion Data by Bones Studio. Dataset use is governed by the [BONES-SEED license](https://bones.studio/info/seed-license).
- [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset): source of Recovery motions; see Harvey et al., *Robust Motion In-Betweening*, SIGGRAPH 2020.
