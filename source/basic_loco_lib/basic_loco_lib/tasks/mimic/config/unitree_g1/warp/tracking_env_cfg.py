"""Warp-backend G1 mimic tracking configuration."""

from __future__ import annotations


from basic_loco_lib import PROJECT_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_experimental.envs.mdp as mdp
from isaaclab_experimental.managers import ObservationTermCfg as ObsTerm
from isaaclab_experimental.managers import RewardTermCfg as RewTerm
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers import TerminationTermCfg as DoneTerm
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab.sensors import ContactSensorCfg

import basic_loco_lib.tasks.mimic.mdp_warp as mdp_warp
from basic_loco_lib.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from basic_loco_lib.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG
from basic_loco_lib.tasks.mimic.terrain import ROUGH_TERRAINS_CFG

##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Observation history length H (frames), applied to BOTH obs groups below.
HISTORY_LENGTH = 4


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        # terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp_warp.MotionCommandWarpCfg(
        asset_name="robot",
        # reuse the Newton-ordered npz generated next to the Newton task (this task's
        motion_file=f"{PROJECT_DIR}/data/unitree_g1/G1_Take_102.bvh_60hz.newton.npz",
        anchor_body_name="pelvis",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observations specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations with per-term history (154 dimensions per frame).

        Wrappers infer dimensions and apply noise once per incoming frame.
        Source parameters are passed through the observation terms.
        """

        # observation terms (order preserved); dims: per-frame -> with history
        motion_command = ObsTerm(
            func=mdp_warp.make_history_obs(mdp.generated_commands, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp_warp.make_history_obs(
                mdp_warp.motion_anchor_ori_b, HISTORY_LENGTH, noise=Unoise(n_min=-0.05, n_max=0.05)
            ),
            params={"command_name": "motion"},
        )
        base_ang_vel = ObsTerm(
            func=mdp_warp.make_history_obs(mdp.base_ang_vel, HISTORY_LENGTH, noise=Unoise(n_min=-0.2, n_max=0.2))
        )
        joint_pos_rel = ObsTerm(
            func=mdp_warp.make_history_obs(mdp.joint_pos_rel, HISTORY_LENGTH, noise=Unoise(n_min=-0.01, n_max=0.01))
        )
        joint_vel_rel = ObsTerm(
            func=mdp_warp.make_history_obs(mdp.joint_vel_rel, HISTORY_LENGTH, noise=Unoise(n_min=-0.5, n_max=0.5))
        )
        last_action = ObsTerm(func=mdp_warp.make_history_obs(mdp.last_action, HISTORY_LENGTH))  # 29 -> 116

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # NOTE: no group-level history_length here -- the experimental

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for critic group (per-frame 286 -> 1144 with history).

        The critic needs the same temporal context as the policy, so every
        term is wrapped with HISTORY_LENGTH frames too (uniform H across the
        group keeps the layout identical to the torch version's group
        history).  Privileged observations are corruption-free: no
        ``noise=`` on any wrapper.
        """

        command = ObsTerm(
            func=mdp_warp.make_history_obs(mdp.generated_commands, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        motion_anchor_pos_b = ObsTerm(
            func=mdp_warp.make_history_obs(mdp_warp.motion_anchor_pos_b, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp_warp.make_history_obs(mdp_warp.motion_anchor_ori_b, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        body_pos = ObsTerm(
            func=mdp_warp.make_history_obs(mdp_warp.robot_body_pos_b, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        body_ori = ObsTerm(
            func=mdp_warp.make_history_obs(mdp_warp.robot_body_ori_b, HISTORY_LENGTH),
            params={"command_name": "motion"},
        )
        base_lin_vel = ObsTerm(func=mdp_warp.make_history_obs(mdp.base_lin_vel, HISTORY_LENGTH))
        base_ang_vel = ObsTerm(func=mdp_warp.make_history_obs(mdp.base_ang_vel, HISTORY_LENGTH))
        joint_pos = ObsTerm(func=mdp_warp.make_history_obs(mdp.joint_pos_rel, HISTORY_LENGTH))
        joint_vel = ObsTerm(func=mdp_warp.make_history_obs(mdp.joint_vel_rel, HISTORY_LENGTH))
        actions = ObsTerm(func=mdp_warp.make_history_obs(mdp.last_action, HISTORY_LENGTH))

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Events (Newton backend, same randomization as the Newton task's events).

    Startup terms use the Warp-convention adapters in ``mdp_warp.events`` (the
    experimental EventManager passes a boolean Warp env mask instead of torch
    env ids); the interval push uses the official Warp-first
    ``isaaclab_experimental.envs.mdp.push_by_setting_velocity``.
    """

    # startup
    add_joint_default_pos = EventTerm(
        func=mdp_warp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp_warp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp_warp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp_warp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp_warp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp_warp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp_warp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp_warp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp_warp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp_warp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Newton physics, fixed (same solver settings as the Newton mimic task)
    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # 95 (copied from the velocity task) overflows in fallen poses during
                njmax=128,
                nconmax=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ls_parallel=True,
                ls_iterations=15,
                # Use Newton collision detection for generated mesh terrain.
                use_mujoco_contacts=False,
            ),
            # collision_cfg=NewtonCollisionPipelineCfg(
            #     # Scene-wide triangle-pair capacity; increase if overflow warnings appear.
            #     max_triangle_pairs=2_500_000,
            # ),
            # Nonzero collision margin for stable triangle-mesh terrain contacts.
            # default_shape_cfg=NewtonShapeCfg(margin=0.01),
            num_substeps=1,
            debug_mode=False,
        )
    )
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # tick contact sensors with the physics step
        self.scene.contact_forces.update_period = self.sim.dt


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
