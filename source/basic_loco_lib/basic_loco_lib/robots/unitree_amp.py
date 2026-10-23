# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP-only Unitree G1 configuration and Unitree torque-speed actuator model."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from basic_loco_lib import ASSETS_DIR


@configclass
class UnitreeAMPArticulationCfg(ArticulationCfg):
    """Articulation config kept separate from the shared locomotion robot config."""

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeAMPUrdfFileCfg(sim_utils.UrdfFileCfg):
    """G1 URDF importer settings for AMP."""

    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir: str, urdf_path: str) -> None:
        """Use an AMP-specific temporary asset link so conversion stays isolated."""
        asset_dir = "/tmp/IsaacLab/basic_loco_lib_amp"
        mesh_link = os.path.join(asset_dir, "meshes")
        urdf_link = os.path.join(asset_dir, "robot.urdf")
        os.makedirs(asset_dir, exist_ok=True)
        for link in (mesh_link, urdf_link):
            if os.path.lexists(link):
                os.remove(link)
        os.symlink(meshes_dir, mesh_link)
        os.symlink(urdf_path, urdf_link)
        self.asset_path = urdf_link


@configclass
class UnitreeAMPActuatorCfg(IdealPDActuatorCfg):
    """Parameters for the torque-speed and friction model used by AMP in MJLab."""

    class_type: type = "{DIR}.unitree_amp:UnitreeAMPActuator"
    velocity_full_effort: float = 1.0
    velocity_no_load: float = 1.0
    effort_same_direction: float = 1.0
    effort_opposite_direction: float = 1.0
    actuator_effort_limit: float = 1.0
    friction_static: float = 0.0
    friction_dynamic: float = 0.0
    friction_activation_velocity: float = 0.01


class UnitreeAMPActuator(IdealPDActuator):
    """Explicit PD actuator matching AMP_mjlab's UnitreeActuator clipping law."""

    cfg: UnitreeAMPActuatorCfg

    def __init__(self, cfg: UnitreeAMPActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        friction = self.cfg.friction_static * torch.tanh(
            joint_vel / self.cfg.friction_activation_velocity
        ) + self.cfg.friction_dynamic * joint_vel
        self.applied_effort = self.applied_effort - friction
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        # GA passes joint_vel directly (EA's _clip_effort(effort) had no velocity arg,
        # hence the old self._joint_vel cache)
        same_direction = joint_vel * effort > 0.0
        peak_effort = torch.where(
            same_direction,
            self.cfg.effort_same_direction,
            self.cfg.effort_opposite_direction,
        )
        peak_effort = torch.minimum(peak_effort, self.actuator_effort_limit)
        speed = joint_vel.abs()
        speed_limited_effort = peak_effort * (
            (self.cfg.velocity_no_load - speed)
            / (self.cfg.velocity_no_load - self.cfg.velocity_full_effort)
        )
        max_effort = torch.where(
            speed < self.cfg.velocity_full_effort,
            peak_effort,
            speed_limited_effort.clamp_min(0.0),
        )
        return torch.maximum(torch.minimum(effort, max_effort), -max_effort)


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_5010_16 = 0.0021812
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _motor_gains(armature: float) -> tuple[float, float]:
    stiffness = armature * NATURAL_FREQ**2
    damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    return stiffness, damping


def _amp_motor(
    *,
    joints: list[str],
    armature: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    effort_limit: float,
    friction_static: float = 0.0,
    friction_dynamic: float = 0.0,
) -> UnitreeAMPActuatorCfg:
    stiffness, damping = _motor_gains(armature)
    return UnitreeAMPActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        actuator_effort_limit=effort_limit,
        joint_effort_limit=effort_limit,
        actuator_velocity_limit=x2,
        velocity_full_effort=x1,
        velocity_no_load=x2,
        effort_same_direction=y1,
        effort_opposite_direction=y2,
        friction_static=friction_static,
        friction_dynamic=friction_dynamic,
    )


UNITREE_G1_29DOF_AMP_CFG = UnitreeAMPArticulationCfg(
    spawn=UnitreeAMPUrdfFileCfg(asset_path=f"{ASSETS_DIR}/unitree_g1/urdf/g1.urdf"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_yaw_pitch": _amp_motor(
            joints=[".*_hip_yaw_joint", ".*_hip_pitch_joint", "waist_yaw_joint"], armature=ARMATURE_7520_14,
            x1=22.63, x2=35.52, y1=71.0, y2=83.3, effort_limit=88.0,
            friction_static=1.6, friction_dynamic=0.16,
        ),
        "hip_roll_knee": _amp_motor(
            joints=[".*_hip_roll_joint", ".*_knee_joint"],
            armature=ARMATURE_7520_22, x1=14.5, x2=22.7, y1=111.0, y2=131.0,
            effort_limit=139.0, friction_static=2.4, friction_dynamic=0.24,
        ),
        "ankles_waist": _amp_motor(
            joints=[".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint"],
            armature=2.0 * ARMATURE_5020, x1=30.86, x2=40.13, y1=49.6, y2=63.8,
            effort_limit=50.0, friction_static=1.2, friction_dynamic=0.12,
        ),
        "arms_5020": _amp_motor(
            joints=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                    ".*_elbow_joint", ".*_wrist_roll_joint"],
            armature=ARMATURE_5020, x1=30.86, x2=40.13, y1=24.8, y2=31.9,
            effort_limit=25.0, friction_static=0.6, friction_dynamic=0.06,
        ),
        "wrists_4010": _amp_motor(
            joints=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], armature=ARMATURE_4010,
            x1=15.3, x2=24.76, y1=4.8, y2=8.6, effort_limit=5.0,
            friction_static=0.6, friction_dynamic=0.06,
        ),
    },
)


_ACTION_LIMITS = {
    ".*_hip_yaw_joint": (88.0, _motor_gains(ARMATURE_7520_14)[0]),
    "waist_yaw_joint": (88.0, _motor_gains(ARMATURE_7520_14)[0]),
    ".*_hip_pitch_joint": (88.0, _motor_gains(ARMATURE_7520_14)[0]),
    ".*_hip_roll_joint": (139.0, _motor_gains(ARMATURE_7520_22)[0]),
    ".*_knee_joint": (139.0, _motor_gains(ARMATURE_7520_22)[0]),
    ".*_ankle_pitch_joint": (50.0, _motor_gains(2.0 * ARMATURE_5020)[0]),
    ".*_ankle_roll_joint": (50.0, _motor_gains(2.0 * ARMATURE_5020)[0]),
    "waist_roll_joint": (50.0, _motor_gains(2.0 * ARMATURE_5020)[0]),
    "waist_pitch_joint": (50.0, _motor_gains(2.0 * ARMATURE_5020)[0]),
    ".*_shoulder_pitch_joint": (25.0, _motor_gains(ARMATURE_5020)[0]),
    ".*_shoulder_roll_joint": (25.0, _motor_gains(ARMATURE_5020)[0]),
    ".*_shoulder_yaw_joint": (25.0, _motor_gains(ARMATURE_5020)[0]),
    ".*_elbow_joint": (25.0, _motor_gains(ARMATURE_5020)[0]),
    ".*_wrist_roll_joint": (25.0, _motor_gains(ARMATURE_5020)[0]),
    ".*_wrist_pitch_joint": (5.0, _motor_gains(ARMATURE_4010)[0]),
    ".*_wrist_yaw_joint": (5.0, _motor_gains(ARMATURE_4010)[0]),
}
UNITREE_G1_29DOF_AMP_ACTION_SCALE = {
    joint_pattern: 0.25 * effort / stiffness
    for joint_pattern, (effort, stiffness) in _ACTION_LIMITS.items()
}
