"""Differentiable G1 FK for imagined mimic observations and tracking rewards."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import torch

from basic_loco_lib.algo.mimic_wm.geometry import (
    axis_angle_to_mat,
    matmul,
    matvec,
    quat_to_mat,
)

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos_tensor(joint_names: Sequence[str], device) -> torch.Tensor:
    """Resolve the regex-style default pose to a vector in ``joint_names`` order."""
    import re

    positions = []
    for name in joint_names:
        value = 0.0
        for pattern, position in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                value = position
        positions.append(value)
    return torch.tensor(positions, dtype=torch.float32, device=device)


def _parse_vec(text: str, default: Sequence[float]) -> list[float]:
    if text is None:
        return list(default)
    return [float(part) for part in text.split()]


def _parse_joint_defaults(root: ET.Element) -> dict[str, dict[str, str]]:
    """Flatten the MJCF ``<default>`` class tree into per-class joint attributes.

    Nested classes inherit from their enclosing class (MuJoCo semantics), so
    each class maps to the merged ``<joint>`` attributes of its ancestor
    chain.  Joint elements reference a class through their ``class``
    attribute.
    """

    def flatten(elem: ET.Element, inherited: dict[str, str]) -> dict[str, dict[str, str]]:
        classes: dict[str, dict[str, str]] = {}
        for child in elem.findall("default"):
            merged = dict(inherited)
            joint = child.find("joint")
            if joint is not None:
                merged.update(joint.attrib)
            name = child.get("class")
            if name is not None:
                classes[name] = merged
            classes.update(flatten(child, merged))
        return classes

    return flatten(root, {})


class G1Kinematics:
    """Kinematic tree with per-body joint axis and fixed parent offset."""

    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        joint_names: Sequence[str] = G1_JOINT_NAMES,
        tracked: Sequence[str] = TRACKED_BODIES,
    ):
        if mjcf_path is None:
            # The shared repo asset: the same MJCF Newton simulates, and the
            # one the motion NPZ files were retargeted against (their body
            # order matches this file's document order).
            mjcf_path = Path(__file__).resolve().parents[3] / "assets" / "unitree_g1" / "mjcf" / "g1.xml"
        self.device = torch.device(device)
        joint_name_to_idx = {name: index for index, name in enumerate(joint_names)}

        root = ET.parse(str(mjcf_path)).getroot()
        worldbody = root.find("worldbody")
        joint_defaults = _parse_joint_defaults(root)

        bodies: list[str] = []
        parent: list[int] = []
        joint_idx: list[int] = []
        axes: list[list[float]] = []
        offsets_p: list[list[float]] = []
        offsets_q: list[list[float]] = []  # MJCF wxyz

        def resolve_joint(joint: ET.Element, attr: str, default: Sequence[float]) -> list[float]:
            """Inline attribute first, then the joint's default class chain."""
            text = joint.get(attr)
            if text is None:
                class_name = joint.get("class")
                text = joint_defaults.get(class_name, {}).get(attr) if class_name else None
            return _parse_vec(text, default)

        def walk(elem: ET.Element, parent_local: int) -> None:
            for body in elem.findall("body"):
                name = body.get("name")
                free = body.find("freejoint")
                joint = body.find("joint")
                if free is None and joint is None:
                    # Fixed body (collision shell / logo / head): skip, children
                    # keep scanning under the same jointed parent.
                    walk(body, parent_local)
                    continue
                mj_to_local[name] = len(bodies)
                bodies.append(name)
                if free is not None:
                    parent.append(-1)
                    joint_idx.append(-1)
                    axes.append([0.0, 0.0, 1.0])
                else:
                    parent.append(parent_local)
                    joint_name = joint.get("name")
                    if joint_name not in joint_name_to_idx:
                        raise ValueError(f"MJCF joint {joint_name!r} is not a tracked robot joint")
                    joint_idx.append(joint_name_to_idx[joint_name])
                    axes.append(resolve_joint(joint, "axis", (0.0, 0.0, 1.0)))
                    ranges = resolve_joint(joint, "range", (0.0, 0.0))
                    jnt_lo[joint_name_to_idx[joint_name]] = ranges[0]
                    jnt_hi[joint_name_to_idx[joint_name]] = ranges[1]
                offsets_p.append(_parse_vec(body.get("pos"), (0.0, 0.0, 0.0)))
                offsets_q.append(_parse_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
                walk(body, len(bodies) - 1)

        mj_to_local: dict[str, int] = {}
        jnt_lo = [0.0] * len(joint_names)
        jnt_hi = [0.0] * len(joint_names)
        walk(worldbody, -1)

        if len(bodies) != len(joint_names) + 1:
            raise ValueError(
                f"expected {len(joint_names) + 1} jointed bodies in {mjcf_path}, got {len(bodies)}"
            )
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.tensor(offsets_p, dtype=torch.float32, device=self.device)
        # MJCF body quats are wxyz; roll to xyzw before quat_to_mat.
        offsets_q_xyzw = torch.roll(
            torch.tensor(offsets_q, dtype=torch.float32, device=self.device), -1, -1
        )
        self.off_R = quat_to_mat(offsets_q_xyzw)
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)
        self.q_lo = torch.tensor(jnt_lo, dtype=torch.float32, device=self.device)
        self.q_hi = torch.tensor(jnt_hi, dtype=torch.float32, device=self.device)

    def to(self, device: str | torch.device) -> G1Kinematics:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        return new

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        all_bodies: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked-body world poses and link-origin linear/angular velocities."""
        positions = [p]
        rotations = [R]
        linear_velocities = [v]
        angular_velocities = [w]
        for index in range(1, self.num_bodies):
            parent_index = self.parent[index]
            joint_index = self.joint_idx[index]
            parent_rotation = rotations[parent_index]
            position = positions[parent_index] + matvec(parent_rotation, self.off_p[index])
            offset_rotation = matmul(parent_rotation, self.off_R[index])
            rotation = matmul(
                offset_rotation, axis_angle_to_mat(self.axes[index], q[..., joint_index])
            )
            axis_world = matvec(offset_rotation, self.axes[index])
            angular_velocity = (
                angular_velocities[parent_index]
                + axis_world * qd[..., joint_index : joint_index + 1]
            )
            linear_velocity = linear_velocities[parent_index] + torch.cross(
                angular_velocities[parent_index],
                position - positions[parent_index],
                dim=-1,
            )
            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)
        indices = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[index] for index in indices], -2),
            torch.stack([rotations[index] for index in indices], -3),
            torch.stack([linear_velocities[index] for index in indices], -2),
            torch.stack([angular_velocities[index] for index in indices], -2),
        )


__all__ = [
    "DEFAULT_JOINT_POS",
    "G1_JOINT_NAMES",
    "G1Kinematics",
    "TRACKED_BODIES",
    "default_joint_pos_tensor",
]
