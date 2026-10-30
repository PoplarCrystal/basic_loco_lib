# Basic Loco Lib（中文文档）

[English](README.md) | 中文

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://github.com/isaac-sim/IsaacLab)

## 概述

`basic_loco_lib` 是一个基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的机器人基础步态库，包含 **Velocity（速度跟踪）、Mimic（动作模仿）、AMP（对抗式运动先验）、Mimic-WM（世界模型动作模仿）** 等基础任务，用于步态策略的训练、回放与部署。目前主要支持 **Unitree G1（29 自由度）**，以 Newton（MuJoCo-Warp）为主要物理后端，部分任务也支持 PhysX。

项目使用 `rsl_rl` 进行 PPO 训练，Mimic-WM 使用独立的世界模型学习器，提供从动作数据准备、策略训练到 ONNX 导出和 MuJoCo sim2sim 验证的完整流程，方便复现基础任务和扩展新的步态。

| 任务 | 任务 ID | 支持后端 | 说明 |
|---|---|---|---|
| Velocity | `Unitree-G1-Velocity`、`Unitree-G1-Velocity-Flat` | Newton / PhysX | 跟踪前向、侧向及转向速度指令（vx、vy、wz），包含起伏地形和平地版本 |
| Velocity Warp | `Unitree-G1-Velocity-Warp`、`Unitree-G1-Velocity-Flat-Warp` | Newton | 基于 Warp 优化的速度跟踪任务 |
| Mimic | `Unitree-G1-Mimic` | Newton / PhysX | 跟踪由 CSV 转换得到的 NPZ 参考动作 |
| Mimic Warp | `Unitree-G1-Mimic-Warp` | Newton | 基于 Warp 优化的动作跟踪任务 |
| Mimic-WM | `Unitree-G1-Mimic-WM` | Newton | 独立动作跟踪任务，通过世界模型和想象 rollout 训练策略 |
| Mimic-WM Warp | `Unitree-G1-Mimic-WM-Warp` | Newton | 使用相同 WM 学习器的 Warp 版本 |
| AMP | `Unitree-G1-AMP-Loco` | Newton / PhysX | 结合速度指令与动作数据集，通过对抗式运动先验学习步态风格 |
| AMP Warp | `Unitree-G1-AMP-Loco-Warp` | Newton | 基于 Warp 优化的 AMP 任务 |

## 环境准备

### Miniconda

本项目基于 **Isaac Lab `v3.0.0` 正式版、Isaac Sim `6.2.0.0` 和 Python `3.12`**。建议固定 Isaac Lab tag，以便复现环境。

```bash
conda create -n basic-loco-lib python=3.12 -y
conda activate basic-loco-lib

# 安装与 Isaac Lab v3.0.0 配套的 Isaac Sim 运行时
pip install "isaacsim[all,extscache]==6.2.0.0" \
    --extra-index-url https://pypi.nvidia.com

# 安装 Isaac Lab 及其依赖
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v3.0.0
./isaaclab.sh -i

# 回到本项目根目录，安装任务库
cd /path/to/basic_loco_lib
pip install -e source/basic_loco_lib/
```

默认 Newton `1.6.1` 栈可运行本仓库任务；起伏地形长时间训练需要升级至 Newton `1.7.0` 以修复 NaN 问题。升级时还需应用 Isaac Lab 兼容补丁，具体步骤见 [Newton 环境说明](doc/newton.md)。

### 物理后端选择

standard 任务在训练、回放和导出时可通过 `presets=newton` 或 `presets=physx` 选择后端，默认使用 Newton。Warp 任务固定使用 Newton，无需指定 `presets`。

以下命令均在项目根目录、激活上述 Conda 环境后执行（部署部分另有目录说明）。standard 示例显式指定后端；Warp 示例省略后端参数。数据、训练和回放应保持使用同一后端。

## 1. 数据准备：CSV → NPZ（Mimic）

Velocity 无需动作数据。Mimic 需要与训练后端匹配的参考动作 NPZ；CSV 与 NPZ 文件放在 `data/<robot>/` 下，转换结果默认保存在输入 CSV 的同一目录。

```bash
# 为 Newton / Warp 任务生成参考动作
python scripts/mimic/csv_to_npz.py -f data/unitree_g1/custom_motion.csv \
    --input_fps 30 --robot unitree_g1 --backend newton --viz none
```

`custom_motion.csv` 为自定义动作文件占位名，请替换为实际路径。输出文件名会包含后端标记，例如 `custom_motion.newton.npz` 或 `custom_motion.physx.npz`。

**注意：** `--backend` 决定 NPZ 中关节与刚体的排序，必须与训练任务的后端一致。若 standard 任务使用 `presets=physx`，生成数据时也应改为 `--backend physx`，并加载对应的 `*.physx.npz`；Newton 训练则使用 `--backend newton` 生成的 `*.newton.npz`。错误排序可能不会报错，却会导致训练失败，例如 episode 长度停在 1、reward 为 0。NPZ 中包含 `ordering`、`joint_names`、`body_names` 等字段，可用于检查数据。

自定义 CSV 每行需包含 36 列：`base_pos(3) + base_quat xyzw(4) + 29 个关节角`。关节角按 [robots/unitree.py](source/basic_loco_lib/basic_loco_lib/robots/unitree.py) 中的 SDK 顺序排列。训练时通过 `--motion_file` 指定转换后的 NPZ；需要自定义任务配置时，可参考现有任务目录，修改 `tracking_env_cfg.py` 并注册新的任务 ID。

AMP 的动作数据位于 `data/amp/`：`WalkandRun/` 用于风格奖励，`Recovery/` 用于延迟起身重置。使用 Newton 后端及仓库配套数据时无需执行上述 CSV 转换；更换数据集可通过 `--motion_file <动作数据目录>` 指定。

动作数据来源与许可说明见 [data/amp/README.md](data/amp/README.md)。

## 2. 训练

以下统一以 Newton 为例。**standard 任务（不带 `-Warp` 后缀）可将 `presets=newton` 替换为 `presets=physx`；Warp 任务仅支持 Newton。** 切换 Mimic 后端时，数据生成的 `--backend` 和训练加载的 NPZ 也必须同步切换。

### Velocity：速度跟踪

```bash
# Newton：起伏地形任务，包含地形与速度指令课程
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity --viz none presets=newton

# Newton：平地任务
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Flat --viz none presets=newton

# Newton / Warp：对应的优化版本
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Warp --viz none
python scripts/rsl_rl/train.py --task Unitree-G1-Velocity-Flat-Warp --viz none
```

平地版本不使用高度扫描和地形课程。

### Mimic：动作模仿

```bash
# Newton：标准版
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic --viz none presets=newton \
    --motion_file /path/to/custom_motion.newton.npz

# Newton / Warp：优化版
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic-Warp --viz none \
    --motion_file /path/to/custom_motion.newton.npz
```

### Mimic-WM：世界模型动作模仿

Mimic-WM 的任务配置和 MDP 独立维护，策略使用 mimic 风格的观测和四帧历史，
奖励参考 SuperMimic 的可微跟踪奖励。训练通过动力学模型中的想象 rollout 更新策略，
不使用 PPO。push 事件保留为注释；执行噪声不进入策略的上一帧动作观测。

```bash
# Newton：标准版
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic-WM --viz none presets=newton \
    --motion_file data/unitree_g1/G1_Take_102.bvh_60hz.newton.npz

# Newton / Warp 版
python scripts/rsl_rl/train.py --task Unitree-G1-Mimic-WM-Warp --viz none \
    --motion_file data/unitree_g1/G1_Take_102.bvh_60hz.newton.npz
```

### AMP：对抗式运动先验

```bash
# Newton：标准版，默认使用 data/amp/ 下的 WalkandRun 与 Recovery 数据
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco --viz none presets=newton

# Newton / Warp：优化版
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco-Warp --viz none

# 指定自定义动作数据集目录
python scripts/rsl_rl/train.py --task Unitree-G1-AMP-Loco --viz none presets=newton \
    --motion_file /path/to/motion_dir
```

**注意：** 默认使用的 `data/amp/WalkandRun/` 与 `data/amp/Recovery/` 数据为 Newton 版本；如需使用 PhysX 后端，请重新生成对应的 PhysX 动作数据。

## 3. 回放与 ONNX 导出

使用 `play.py` 加载训练后的 checkpoint。**任务、物理后端及 Mimic 参考动作应与训练时保持一致。** 以下为 Newton 后端示例：

```bash
# Mimic：回放 / 仅导出
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/ckpt.pt --export_only --num_envs 1

# Mimic-WM：回放 / 仅导出，任务 ID 与训练保持一致
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-WM-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/last.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-Mimic-WM-Warp \
    --motion_file /path/to/custom_motion.newton.npz --checkpoint /path/to/last.pt \
    --export_only --num_envs 1 --viz none

# Velocity：回放 / 仅导出
python scripts/rsl_rl/play.py --task Unitree-G1-Velocity-Flat-Warp \
    --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-Velocity-Flat-Warp \
    --checkpoint /path/to/ckpt.pt --export_only --num_envs 1

# AMP：回放 / 仅导出
python scripts/rsl_rl/play.py --task Unitree-G1-AMP-Loco presets=newton \
    --checkpoint /path/to/ckpt.pt --viz viser
python scripts/rsl_rl/play.py --task Unitree-G1-AMP-Loco presets=newton \
    --checkpoint /path/to/ckpt.pt --export_only --num_envs 1
```

- `--viz viser` 打开轻量 3D 查看器；无头运行可显式使用 `--viz none`。
- `--export_only` 导出策略后退出，不进入回放循环；普通回放也会导出策略。
- 不指定 `--checkpoint` 时，自动选取 `logs/rsl_rl/<experiment_name>/` 下最新一次运行的 checkpoint。
- 回放 PhysX 策略时，使用训练时的标准版任务 ID 和 `presets=physx`；Mimic 同时指定 PhysX 格式的 NPZ。
- 若 AMP 训练时使用了自定义数据集，回放或导出时同样添加 `--motion_file /path/to/motion_dir`。

导出文件保存在 `<checkpoint目录>/exported/`，包括 `policy.pt` 和单文件自包含的 `policy.onnx`。ONNX 的输入为 `obs_history`，可用于下述 MuJoCo 部署流程。

## 4. 部署：MuJoCo sim2sim

部署脚本位于 `deploy_mujoco/task/<task>/`。机器人配置文件为 `<task>/<robot>/<robot>.yaml`，其中包含 PD 增益、关节顺序映射、观测布局和策略路径。将导出的 `policy.onnx` 放在对应 YAML 旁边；Mimic 还需要放置匹配的参考动作 NPZ，并检查 YAML 中的动作文件路径。部署配置中的 `backend` 必须与策略训练后端一致（`newton` 或 `physx`），以便选择正确的关节映射。

在项目根目录安装部署及导出依赖：

```bash
pip install -e "source/basic_loco_lib[deploy]"
```

以下三组命令分别从项目根目录进入对应任务目录执行。

Velocity：通过 Tk 滑条面板控制 vx / vy / wz 速度指令。

```bash
cd deploy_mujoco/task/velocity
python deploy_mujoco.py unitree_g1
```

AMP：使用同样的速度指令面板，指令范围与 AMP 训练配置一致。

```bash
cd deploy_mujoco/task/amp
python deploy_mujoco.py unitree_g1
```

Mimic：从参考动作第 0 帧初始化机器人状态并跟踪动作。

```bash
cd deploy_mujoco/task/mimic
python deploy_mujoco.py unitree_g1
```

Mimic-WM **复用同一个** `deploy_mujoco/task/mimic/deploy_mujoco.py`。
先使用训练时的 Mimic-WM 任务 ID 导出策略；部署只需要 actor，不需要 WM 和 replay buffer。
导出的 FP32 ONNX 输入为 `obs_history`（4 × 154 = 616 维），输出为 `actions`（29 维）。
在现有 mimic YAML 中设置训练后端（上面示例为 `backend: newton`），
把 `motion` 改为匹配的 `.newton.npz`，把 `policy` 指向 WM actor 的 ONNX。
保持 `obs_history_length: 4`、PD 增益、动作缩放和关节映射与训练一致。
仓库中的 mimic YAML 默认使用 PhysX，使用 Newton 策略前需要改好后端和动作路径。
这里验证的是部署接口兼容，实际跟踪效果仍取决于 checkpoint 的训练质量。

重新训练后，使用 `--export_only` 导出策略，并更新对应部署目录下的 `policy.onnx`。部署时的观测顺序、历史长度、缩放和关节映射必须与训练配置一致；具体约定见各任务 `deploy_mujoco.py` 的文件说明。扩展机器人时，可参考现有目录添加机器人 YAML、模型和策略文件。

## 致谢

本项目的开发参考了以下论文和开源项目，感谢各作者与社区的贡献：

- [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Unitree RL Lab（isaaclab3.0 分支）](https://github.com/unitreerobotics/unitree_rl_lab/tree/isaaclab3.0)
- [Whole-Body Tracking](https://github.com/HybridRobotics/whole_body_tracking)
- [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)：世界模型策略训练方法的参考。
- [Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)：基于 UniLab 的 SuperMimic 独立复现工程，为本项目 Mimic-WM 的实现和可微奖励提供参考。

动作数据致谢：

- [BONES-SEED / Bones Studio](https://bones.studio/datasets/seed)：WalkandRun 动作来源。Motion Data by Bones Studio；使用数据须遵循其[数据集许可](https://bones.studio/info/seed-license)。
- [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)：Recovery 动作来源；相关论文为 Harvey 等人的 *Robust Motion In-Betweening*（SIGGRAPH 2020）。
