"""Newton-backend AMP locomotion task for Unitree G1."""

from __future__ import annotations

import math

from isaaclab_experimental.envs import mdp as warp_mdp
from isaaclab_experimental.managers import EventTermCfg as EventTerm
from isaaclab_experimental.managers import ObservationTermCfg as ObsTerm
from isaaclab_experimental.managers import RewardTermCfg as RewTerm
from isaaclab_experimental.managers import (
    SceneEntityCfg,
)
from isaaclab_experimental.managers import TerminationTermCfg as DoneTerm
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.utils import PresetCfg

from basic_loco_lib import PROJECT_DIR
from basic_loco_lib.robots.unitree_amp import (
    UNITREE_G1_29DOF_AMP_ACTION_SCALE,
)
from basic_loco_lib.robots.unitree_amp import UNITREE_G1_29DOF_AMP_CFG as ROBOT_CFG
from basic_loco_lib.tasks.amp.mdp_warp import events as amp_events
from basic_loco_lib.tasks.amp.mdp_warp import observations as amp_obs
from basic_loco_lib.tasks.amp.mdp_warp import obs_history as amp_history
from basic_loco_lib.tasks.amp.mdp_warp import rewards as amp_rewards
from basic_loco_lib.tasks.amp.mdp_warp import terminations as amp_terminations
from basic_loco_lib.tasks.amp.mdp_warp.commands import WarpUniformVelocityCommand

##
# Physics presets
##


@configclass
class PhysicsCfg(PresetCfg):
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Suppress line-search limit warnings; keep other warnings enabled.
            class_type="basic_loco_lib.utils.physics.mjwarp_manager:NewtonMJWarpQuietLSManager",
            # Extra contact capacity for capsule feet and self-collisions.
            njmax=256,
            nconmax=64,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            ls_parallel=True,
            ls_iterations=15,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton


##
# Scene
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the AMP locomotion scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# Commands
##


@configclass
class CommandsCfg:
    twist = UniformVelocityCommandCfg(
        class_type=WarpUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-3.14 / 2, 3.14 / 2),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    )


##
# Actions
##


@configclass
class ActionsCfg:
    joint_pos = warp_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_29DOF_AMP_ACTION_SCALE,
        use_default_offset=True,
    )


##
# Observations
##

_AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
_ANCHOR_NAME = "torso_link"


@configclass
class ObservationsCfg:
    """Observation specifications for AMP."""

    @configclass
    class ActorCfg(ObsGroup):
        # term order mirrors the velocity task's policy group
        command = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.generated_commands, 4, frame_dim=3),
            params={"command_name": "twist"},
            history_length=0,
        )
        projected_gravity = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.projected_gravity, 4, frame_dim=3, noise=Unoise(n_min=-0.05, n_max=0.05)
            ),
            history_length=0,
        )
        base_ang_vel = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.base_ang_vel, 4, frame_dim=3, noise=Unoise(n_min=-0.2, n_max=0.2)),
            history_length=0,
        )
        joint_pos = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.joint_pos_rel, 4, frame_dim=29, noise=Unoise(n_min=-0.01, n_max=0.01)
            ),
            history_length=0,
        )
        joint_vel = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.joint_vel_rel, 4, frame_dim=29, noise=Unoise(n_min=-0.5, n_max=0.5)
            ),
            history_length=0,
        )
        actions = ObsTerm(func=amp_history.make_history_obs(warp_mdp.last_action, 4, frame_dim=29), history_length=0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # term order mirrors the velocity task's critic group
        command = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.generated_commands, 4, frame_dim=3),
            params={"command_name": "twist"},
            history_length=0,
        )
        projected_gravity = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.projected_gravity, 4, frame_dim=3), history_length=0
        )
        base_lin_vel = ObsTerm(func=amp_history.make_history_obs(warp_mdp.base_lin_vel, 4, frame_dim=3), history_length=0)
        base_ang_vel = ObsTerm(func=amp_history.make_history_obs(warp_mdp.base_ang_vel, 4, frame_dim=3), history_length=0)
        joint_pos = ObsTerm(func=amp_history.make_history_obs(warp_mdp.joint_pos_rel, 4, frame_dim=29), history_length=0)
        joint_vel = ObsTerm(func=amp_history.make_history_obs(warp_mdp.joint_vel_rel, 4, frame_dim=29), history_length=0)
        actions = ObsTerm(func=amp_history.make_history_obs(warp_mdp.last_action, 4, frame_dim=29), history_length=0)
        body_pos_b = ObsTerm(
            func=amp_history.make_history_obs(amp_obs.robot_body_pos_b, 4, frame_dim=39),
            history_length=0,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )
        body_ori_b = ObsTerm(
            func=amp_history.make_history_obs(amp_obs.robot_body_ori_b, 4, frame_dim=78),
            history_length=0,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=amp_obs.robot_body_pos_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )
        body_ori_b = ObsTerm(
            func=amp_obs.robot_body_ori_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )
        body_lin_vel_b = ObsTerm(
            func=amp_obs.robot_body_lin_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )
        body_ang_vel_b = ObsTerm(
            func=amp_obs.robot_body_ang_vel_b,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


##
# Events
##

_MOTION_BASE = PROJECT_DIR / "data" / "amp"
_MOTION_DIR = str(_MOTION_BASE / "WalkandRun")
_RECOVERY_DIR = str(_MOTION_BASE / "Recovery")


@configclass
class EventCfg:
    init_motion_loader = EventTerm(
        func=amp_events.init_motion_loader_warp,
        mode="startup",
        params={
            "motion_dir": _MOTION_DIR,
            "recovery_dir": _RECOVERY_DIR,
            "delay_reset_env_ratio": 0.4,
            "max_delay_steps": 250,
            # Abort a delayed recovery early if physics enters a runaway state.
            "max_delay_joint_velocity": 50.0,
        },
    )
    reset_from_motion = EventTerm(
        func=amp_events.reset_from_motion_data_warp,
        mode="reset",
        params={
            "motion_dir": _MOTION_DIR,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    push_robot = EventTerm(
        func=warp_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    track_anchor_lin_vel = RewTerm(
        func=amp_rewards.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_anchor_ang_vel = RewTerm(
        func=amp_rewards.track_anchor_angular_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 3.14,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_root_height = RewTerm(
        func=amp_rewards.track_root_height,
        weight=1.0,
        params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    )
    body_ang_vel_xy = RewTerm(
        func=amp_rewards.body_ang_vel_xy_l2,
        weight=0.5,
        params={
            "std": 3.14,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "body_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )
    joint_acc = RewTerm(func=warp_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limit = RewTerm(func=warp_mdp.joint_pos_limits, weight=-10.0)
    action_rate = RewTerm(func=warp_mdp.action_rate_l2, weight=-0.01)
    is_terminated = RewTerm(func=warp_mdp.is_terminated, weight=-200.0)


##
# Terminations
##


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=warp_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=amp_terminations.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_root_height = DoneTerm(
        func=warp_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 AMP locomotion environment."""

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg(), use_newton_actuators=False)
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

    def play_mode(self):
        super().play_mode()
        self.events.push_robot = None
