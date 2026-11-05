# Copyright (c) 2025-2026, The IsaacNewton Project Developers.
# AMP motion data loader - loads npz files and precomputes body-relative features.

from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from tqdm import tqdm

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)


class AMPLoader:
    def __init__(self, motion_file: str,
                 body_names: Sequence[str],
                 anchor_name: str,
                 device: str = "cuda:0"):
        """Load AMP motion data.

        Args:
            motion_file: Path to a single .npz file or a directory of .npz files.
            body_names: Names of the target bodies to track.
            anchor_name: Name of the anchor (root) body.
            device: Torch device.
        """
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

        self._num_bodies = len(body_names)

        # Resolve a motion file or directory.
        if os.path.isfile(motion_file):
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith('.npz'):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

        self.motion_names = motion_names
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []

        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path)
            file_body_names = data["body_names"].tolist()
            body_indexes = [file_body_names.index(name) for name in body_names]
            anchor_index = file_body_names.index(anchor_name)

            if motion_idx == 0:
                self.fps = data["fps"]

            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            time_step_total = _dof_pos.shape[0]

            _body_pos_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_quat_b = torch.zeros((time_step_total, self._num_bodies, 4), dtype=torch.float32, device=device)
            _body_ori_b = torch.zeros((time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device)
            _body_lin_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_ang_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)

            for frame_idx in tqdm(range(time_step_total), desc=f"Preloading AMP data for {motion_name}"):
                tgt_anchor_pos_w = _body_pos_w[frame_idx, anchor_index, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_anchor_quat_w = _body_quat_w[frame_idx, anchor_index, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_body_pos_w = _body_pos_w[frame_idx, body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, body_indexes, :]

                tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                    subtract_frame_transforms(
                        tgt_anchor_pos_w,
                        tgt_anchor_quat_w,
                        tgt_body_pos_w,
                        tgt_body_quat_w,
                    )
                )

                mat = matrix_from_quat(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                tgt_body_lin_vel_b = quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_lin_vel_w,
                )

                tgt_body_ang_vel_b = quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_ang_vel_w,
                )

                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_quat_b[frame_idx] = tgt_robot_body_quat_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b

            self._body_pos_b_list.append(_body_pos_b)
            self._body_quat_b_list.append(_body_quat_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)

        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

        # Pack clips for per-sample random selection without host/device indexing.
        self._motion_lengths = torch.tensor(
            [frames.shape[0] for frames in self._body_pos_b_list], device=device
        )
        self._motion_offsets = self._motion_lengths.cumsum(0) - self._motion_lengths
        self._observations = torch.cat(
            [
                torch.cat([feature.flatten(1) for feature in features], dim=-1)
                for features in zip(
                    self._body_pos_b_list,
                    self._body_ori_b_list,
                    self._body_lin_vel_b_list,
                    self._body_ang_vel_b_list,
                )
            ]
        )

    @property
    def observation_dim(self) -> int:
        obs_dim = (3 + 6 + 3 + 3) * self._num_bodies
        return obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        device = self._observations.device
        for _ in range(num_mini_batch):
            motion_ids = torch.randint(len(self.motion_names), (mini_batch_size,), device=device)
            # Each clip has equal probability; its final frame cannot start a pair.
            frame_ids = (
                torch.rand(mini_batch_size, device=device) * (self._motion_lengths[motion_ids] - 1)
            ).long()
            idxs = self._motion_offsets[motion_ids] + frame_ids
            yield self._observations[idxs], self._observations[idxs + 1]
