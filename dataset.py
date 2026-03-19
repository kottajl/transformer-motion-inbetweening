import numpy as np
import torch
import os

from torch.utils.data import Dataset
from bvh_parser import load_anim_from_npz
from utils import to_tensor
from typing import Literal


def get_data_subset_paths(
    data_dir: str, 
    subset_type: Literal['all', 'selected-subjects', 'selected-moves', 'selected-subjects-and-moves', 'selected-files'] = 'all', 
    **kwargs
) -> list:
    """
    Get subset of data (following the subset_type and additional parameters in **kwargs).
    """

    if subset_type == 'all':
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    paths = []

    if subset_type == 'selected-subjects':
        try:
            subjects_indices = kwargs['subjects_indices']
        except KeyError:
            raise ValueError("subjects_indices must be provided when subset_type is 'selected-subjects'")
        for f in os.listdir(data_dir):
            for sub_idx in subjects_indices:
                if f"subject{sub_idx}" in f:
                    paths.append(os.path.join(data_dir, f))

    elif subset_type == 'selected-moves':
        try:
            moves_names = kwargs['moves_names']
        except KeyError:
            raise ValueError("moves_names must be provided when subset_type is 'selected-moves'")
        for f in os.listdir(data_dir):
            for move_name in moves_names:
                if move_name in f:
                    paths.append(os.path.join(data_dir, f))
    
    elif subset_type == 'selected-subjects-and-moves':
        try:
            subjects_indices = kwargs['subjects_indices']
            moves_names = kwargs['moves_names']
        except KeyError:
            raise ValueError("subjects_indices and moves_names must be provided when subset_type is 'selected-subjects-and-moves'")
        for f in os.listdir(data_dir):
            for sub_idx in subjects_indices:
                for move_name in moves_names:
                    if (f"subject{sub_idx}" in f) and (move_name in f):
                        paths.append(os.path.join(data_dir, f))

    elif subset_type == 'selected-files':
        try:
            files_names = kwargs['files_names']
        except KeyError:
            raise ValueError("files_names must be provided when subset_type is 'selected-files'")
        for f in os.listdir(data_dir):
            if f in files_names:
                paths.append(os.path.join(data_dir, f))
    
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")

    return paths
#get_data_subset_paths


class BvhDataset(Dataset):
    def __init__(
        self,
        data_dir,
        window: int,
        step: int,
        device: str,
        interpolate_missing: bool = False,
        subset_type: str = 'all',
        **subset_kwargs
    ):
        self.window = window
        self.step = step

        self.positions = []
        self.rotations_6d = []
        self.rotations_quat = [] if interpolate_missing else None
        self.n_frames = []
        self.parents = None
        self.offsets = None

        data_paths = get_data_subset_paths(data_dir, subset_type, **subset_kwargs)
        print(f"DATASET: Loading {len(data_paths)} animations from {data_dir}...")

        for path in data_paths:
            animation = load_anim_from_npz(path)
            self.positions.append(to_tensor(animation.positions, device=device))
            self.rotations_6d.append(to_tensor(animation.rotations_6d, device=device))
            self.n_frames.append(animation.rotations_6d.shape[0])
            if self.parents is None:
                self.parents = animation.parents
            if self.offsets is None:
                self.offsets = animation.offsets
            if interpolate_missing:
                self.rotations_quat.append(to_tensor(animation.rotations_quat, device=device))
        
        self.window_indices = []
        for anim_idx, anim_n_frames in enumerate(self.n_frames):
            if anim_n_frames < self.window:     # animation is too short - skip
                continue
            n_windows = (anim_n_frames - self.window) // self.step + 1
            for window_idx in range(n_windows):
                start = window_idx * self.step
                end = start + self.window
                self.window_indices.append((anim_idx, start, end))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        """
        Get one motion window from animation.
        """
        anim_idx, start, end = self.window_indices[idx]
        positions = self.positions[anim_idx][start:end]     # [T, 3]
        rotations = self.rotations_6d[anim_idx][start:end]  # [T, J, 6]

        assert rotations is not None, f"rotations is None for anim {anim_idx}"
        assert rotations.ndim == 3, f"rotations ndim != 3 for anim {anim_idx}: {rotations.shape}"

        if self.rotations_quat is not None:
            rotations_quat = self.rotations_quat[anim_idx][start:end] # [T, J, 4]
            assert rotations_quat is not None, f"rotations_quat is None for anim {anim_idx}"
            assert rotations_quat.ndim == 3, f"rotations_quat ndim != 3 for anim {anim_idx}: {rotations_quat.shape}"
            return {
                "rotations": rotations,     # (rotations 6d)
                "rotations_quat": rotations_quat,
                "positions": positions
            }
        else:
            return {
                "rotations": rotations,
                "positions": positions
            }