from typing import Literal
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
from animation import Animation
from utils import euler_to_6d, euler_to_quat

import numpy as np
import os


def load_bvh(bvh_path) -> Animation:

    with open(bvh_path, 'r') as f:
        mocap = Bvh(f.read())
    
    frames = np.array(mocap.frames, dtype=np.float32)
    num_frames = len(frames)
    joint_names = mocap.get_joints_names()
    num_joints = len(joint_names)
    # rotations in 6D/quat for every joint and frame
    rotations_6d = np.zeros((num_frames, num_joints, 6), dtype=np.float32)
    rotations_quat = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

    # Positions (root)
    root_name = joint_names[0]
    root_channels = mocap.joint_channels(root_name)
    pos_channels = [ch for ch in root_channels if 'position' in ch.lower()] # filter to *position* channels
    if pos_channels:
        root_idx_start = mocap.get_joint_channels_index(root_name) 
        joint_idx_end = root_idx_start + len(root_channels)
        pos_indices = list(range(root_idx_start, joint_idx_end))[:3]    # first 3 channels should be positions
        positions = frames[:, pos_indices]
    else:
        positions = np.zeros((num_frames, 3), dtype=np.float32)

    # Offsets
    offsets = np.zeros((num_joints, 3), dtype=np.float32)
    parents = np.full(num_joints, -1, dtype=np.int32)
    for j, joint_name in enumerate(joint_names):
        offsets[j] = np.array(mocap.joint_offset(joint_name), dtype=np.float32)
        parent_name = mocap.joint_parent(joint_name)
        if parent_name is None:
            continue    # parents[j] = -1
        parent_name = parent_name.name if hasattr(parent_name, 'name') else str(parent_name)
        parent_name = parent_name.replace("ROOT ", "").replace("JOINT ", "").strip()
        if parent_name in joint_names:
            parents[j] = joint_names.index(parent_name)

    # Rotations
    for j, joint_name in enumerate(joint_names):

        # Get list of channels for this joint (e.g. channels = ["Zrotation", "Yrotation", "Xrotation"])
        channels = mocap.joint_channels(joint_name)
        n_channels = len(channels)
        if n_channels < 3:      # skip end-site joints (without rotation)
            continue
        rot_channels = [ch for ch in channels if 'rotation' in ch.lower()] # filter to *rotation* channels
        
        # Extract euler order (e.g.: ["Zrotation", "Yrotation", "Xrotation"] -> "zyx")
        euler_order = "".join([rot_ch[0].lower() for rot_ch in rot_channels])

        # Get indices of rot_channels columns 
        joint_idx_start = mocap.get_joint_channels_index(joint_name) 
        joint_idx_end = joint_idx_start + n_channels
        rot_indices = list(range(joint_idx_start, joint_idx_end))[-3:]      # last 3 channels should always be rotations

        # Get rotations of this joint for every frame
        eulers = frames[:, rot_indices]     # (-> [num_frames, 3])
        rotations_quat[:, j, :] = euler_to_quat(eulers, euler_order)
        rotations_6d[:, j, :] = euler_to_6d(eulers, euler_order)

    return Animation(
        rotations_6d=rotations_6d,      # [N, J, 6]
        rotations_quat=rotations_quat,  # [N, J, 4]
        positions=positions,    # [N, 3]
        offsets=offsets,        # [J, 3]
        parents=parents,        # [J]
        names=joint_names
    )


def save_anim_to_npz(anim: Animation, out_path: str):
    names = np.array(anim.names, dtype=object)

    np.savez_compressed(
        out_path,
        rotations_6d=anim.rotations_6d,
        rotations_quat=anim.rotations_quat,
        positions=anim.positions,
        offsets=anim.offsets,
        parents=anim.parents,
        names=names,
    )
#save_anim_to_npz


def load_anim_from_npz(npz_path: str) -> Animation:
    data = np.load(npz_path, allow_pickle=True)

    rotations_6d = data['rotations_6d']
    rotations_quat = data["rotations_quat"]
    positions = data['positions']
    offsets = data['offsets']
    parents = data['parents']
    names = data['names'].tolist()

    return Animation(
        rotations_6d=rotations_6d,
        rotations_quat=rotations_quat,
        positions=positions,
        offsets=offsets,
        parents=parents,
        names=names,
    )


def parse_all_bvh_to_npz(bvh_dir, npz_dir, mode: Literal["all", "train", "test"] = "all", clear_npzs: bool = True):
    """
    Convert all .bvh files under `bvh_dir` to .npz in `npz_dir`.
    """
    bvh_dir = os.path.abspath(bvh_dir)
    npz_dir = os.path.abspath(npz_dir)
    os.makedirs(npz_dir, exist_ok=True)

    # Clear npz_dir
    if clear_npzs:
        for fname in os.listdir(npz_dir):
            if fname.lower().endswith('.npz'):
                os.remove(os.path.join(npz_dir, fname))

    saved = []
    errors = []

    for fname in os.listdir(bvh_dir):
        if not fname.lower().endswith('.bvh'):
            continue
        try:
            subject_id = int(fname.split('.')[0].split('_')[-1][-1])
        except ValueError:
            print(f"Warning: cannot determine subject ID from filename '{fname}', skipping.")
            continue
        if mode == "train" and subject_id == 5:
            continue
        if mode == "test" and subject_id != 5:
            continue

        in_bvh_path = os.path.join(bvh_dir, fname)

        out_npz_fname = os.path.splitext(fname)[0] + '.npz'
        out_npz_path = os.path.join(npz_dir, out_npz_fname)

        try:
            anim = load_bvh(in_bvh_path)
            save_anim_to_npz(anim, out_npz_path)
            saved.append(out_npz_path)
        except Exception as e:
            errors.append((in_bvh_path, repr(e)))

    return saved, errors
