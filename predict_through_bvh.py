from typing import List
from scipy.spatial.transform import Rotation as R
from bvh import Bvh
from bvh_parser import load_bvh
from interpolation import interpolate_positions, interpolate_rotations
from utils import load_params_from_json
from model.model import MotionTransformer
from predict_bvh import sixd_to_matrix, stable_euler_from_matrix

import numpy as np
import torch
import argparse


def get_bvh_frames(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Frames:"):
                return int(line.split(":")[1].strip())
    raise ValueError("Frames not found in BVH file.")


def predict_bvh_loop(
    model: torch.nn.Module,
    bvh_path_in: str,
    bvh_path_out: str,
    window_size: int,
    period: int,
    device: torch.device = None,
    context_frames: int = 10,
    target_frames: int = 1,
    preinterpolate: bool = False
):
    """
    Run inference on a BVH file and write result to another BVH.
    """
    # print(f"Predicting BVH hole from frame {start_hole} to {end_hole}...")
    # print(f"Window size: {window_size}, Context frames: {context_frames}, Target frames: {target_frames}")
    # assert (end_hole - start_hole) + context_frames + target_frames == window_size, "Window data doesn't match up..."

    if device is None:
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu')
    model.eval()

    with open(bvh_path_in, 'r') as f:
        raw_text = f.read()
    mocap = Bvh(raw_text)
    anim = load_bvh(bvh_path_in)

    rot6d = anim.rotations_6d   # (F, J, 6)
    positions = anim.positions  # (F, 3)

    F, J, _ = rot6d.shape

    hole_size = window_size - context_frames - target_frames
    print(f"Total frames: {F}, Joints: {J}, Hole size: {hole_size}")
    assert hole_size > 0, "Hole size must be positive"

    # START LOOP
    print()
    for start_hole in range(50, F, period):
        end_hole = start_hole + hole_size
        if end_hole + target_frames >= F:
            break

        assert 0 <= start_hole < end_hole <= F, "Invalid hole range"

        hole_len = end_hole - start_hole
        T_win = context_frames + hole_len + target_frames
        start_window = start_hole - context_frames
        end_window = start_window + T_win
        assert start_window >= 0 and end_window <= F, "Window exceeds sequence bounds"

        print(f"Predicting hole frames {start_hole} to {end_hole} (window {start_window} to {end_window})...", flush=True)

        win_rot = rot6d[start_window:end_window]        # (T_win, J, 6)
        win_pos = positions[start_window:end_window]    # (T_win, 3)
        assert win_rot.shape[0] == T_win, f"Window length mismatch: got {win_rot.shape[0]} expected {T_win}"

        rot = torch.from_numpy(win_rot).unsqueeze(0).to(device)  # (1, T_win, J, 6)
        pos = torch.from_numpy(win_pos).unsqueeze(0).to(device)  # (1, T_win, 3)

        # Center positions around root joint
        root_offset = pos[:, 0:1, :].clone()   # (B, T, 1, 3)
        pos -= root_offset

        with torch.no_grad():
            src_rot = rot.clone()
            src_pos = pos.clone()
            hole_start_in_win = start_hole - start_window
            hole_end_in_win = end_hole - start_window
            src_rot[:, hole_start_in_win:hole_end_in_win, :, :] = 0.0
            src_pos[:, hole_start_in_win:hole_end_in_win, :] = 0.0

            if preinterpolate:
                win_rot_q = anim.rotations_quat[start_window:end_window]        # (T_win, J, 4)
                src_rot_q = torch.from_numpy(win_rot_q).unsqueeze(0).to(device) # (1, T_win, J, 4)
                src_rot_q[:, hole_start_in_win:hole_end_in_win, :, :] = 0.0
                src_rot, _ = interpolate_rotations(
                    src_rot,
                    src_rot_q,
                    hole_start_in_win,
                    hole_end_in_win
                )
                src_pos = interpolate_positions(
                    src_pos,
                    hole_start_in_win,
                    hole_end_in_win
                )

            # Fixed points (context_frames and target_frames indices)
            fixed_points: List[int] = list(range(0, context_frames)) + list(range(T_win - target_frames, T_win))
            
            # pred_rot, pred_pos = model(
            #     src_rot, src_pos,
            #     src_rot.clone(), src_pos.clone(),
            #     fixed_points=fixed_points
            # )
            pred_rot, pred_pos = model(
                src_rot, src_pos
            )

            # Return to original position space
            pred_pos += root_offset

        pred_rot = pred_rot.cpu().numpy()[0]    # (T_win, J, 6)
        pred_pos = pred_pos.cpu().numpy()[0]    # (T_win, 3)

        # final_rot6d = rot6d.copy()
        # final_pos = positions.copy()
        # final_rot6d[start_hole:end_hole, :, :] = pred_rot[hole_start_in_win:hole_end_in_win, :, :]
        # final_pos[start_hole:end_hole, :] = pred_pos[hole_start_in_win:hole_end_in_win, :]

        # Ensure the rotations are correctly normalized
        for t_idx in range(hole_start_in_win, hole_end_in_win):
            m = sixd_to_matrix(pred_rot[t_idx])  # (J, 3, 3)
            pred_rot[t_idx, :, :3] = m[:, :, 0]
            pred_rot[t_idx, :, 3:] = m[:, :, 1]

        rot6d[start_hole:end_hole, :, :] = pred_rot[hole_start_in_win:hole_end_in_win, :, :]
        positions[start_hole:end_hole, :] = pred_pos[hole_start_in_win:hole_end_in_win, :]

        print(".", end="", flush=True)
    # END LOOP

    # Rebuild BVH frames

    total_channels = 0
    joint_names = mocap.get_joints_names()
    for name in joint_names:
        ch = mocap.joint_channels(name)
        total_channels += len(ch)

    new_frames = np.zeros((F, total_channels), dtype=np.float32)

    # Fill channel columns joint by joint
    for j, joint_name in enumerate(joint_names):
        channels = mocap.joint_channels(joint_name)
        start_idx = mocap.get_joint_channels_index(joint_name)

        # Find rotation channels for this joint (order and positions)
        rot_channel_indices = [i for i, ch in enumerate(channels) if 'rotation' in ch.lower()]
        pos_channel_indices = [i for i, ch in enumerate(channels) if 'position' in ch.lower()]

        # Fill positions (for root)
        if pos_channel_indices:
            for idx in pos_channel_indices:
                ch = channels[idx].lower()
                if ch.startswith('x'):
                    axis = 0
                elif ch.startswith('y'):
                    axis = 1
                elif ch.startswith('z'):
                    axis = 2
                else:
                    raise ValueError(f"Unknown position channel name: {ch}")
                new_frames[:, start_idx + idx] = positions[:, axis]

        # Fill rotations
        if len(rot_channel_indices) == 3:
            # Euler order string (e.g. 'zyx') from channel names
            rot_ch_names = [channels[i] for i in rot_channel_indices]
            euler_order = ''.join([r[0].lower() for r in rot_ch_names])
            euler_order = euler_order[::-1]

            # Convert 6D -> matrices -> euler angles
            joint_sixd = rot6d[:, j, :]   # (F, 6)
            mats = sixd_to_matrix(joint_sixd)  # (F, 3, 3)
            eulers = stable_euler_from_matrix(mats, euler_order, degrees=True)  # (F, 3)
            eulers = eulers[:, ::-1]
            # eulers = R.from_matrix(mats).as_euler(euler_order, degrees=True)

            for local_rot_idx, ch_idx in enumerate(rot_channel_indices):
                new_frames[:, start_idx + ch_idx] = eulers[:, local_rot_idx]
        else:
            raise ValueError(f"Joint {joint_name} has unexpected number of rotation channels: {len(rot_channel_indices)}")

    # Rebuild BVH text
    
    # Copy header up to 'MOTION'
    lower_text = raw_text.lower()
    mot_idx = lower_text.find('\nmotion')
    if mot_idx == -1:
        mot_idx = lower_text.find('motion')
    header_text = raw_text[:mot_idx]

    # Extract original Frame Time line if present
    frame_time = 0.033333
    for line in raw_text.splitlines():
        if line.strip().lower().startswith('frame time'):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    frame_time = float(parts[1].strip())
                except Exception:
                    print("Warning: couldn't parse Frame Time line, using default 0.033333", flush=True)
                    pass
            break

    # Build motion block
    motion_lines = ["MOTION", f"Frames: {F}", f"Frame Time: {frame_time}"]
    for f in range(F):
        row = new_frames[f]
        line = ' '.join([f"{v:.6f}" for v in row.tolist()])
        motion_lines.append(line)
    out_text = header_text + '\n' + '\n'.join(motion_lines) + '\n'

    with open(bvh_path_out, 'w') as f:
        f.write(out_text)
        # print(f"Written predicted BVH to '{bvh_path_out}'")

#predict_bvh_loop


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh-in', type=str, default="eval/jumps1_subject5.bvh")
    parser.add_argument('--bvh-out', default="eval/jumps1_subject5_predicted_alpha14.bvh")
    parser.add_argument('--period', type=int, default=25)
    parser.add_argument('--weights', default="best_model.pt", help='Path to model weights (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    args = parser.parse_args()

    # Load model parameters from JSON
    try:
        params = load_params_from_json(args.config)
    except FileNotFoundError:
        print(f"Error: The file '{args.config}' was not found.")
        exit(1)

    joint_embedding_size = params["joint_embedding_size"]
    root_embedding_size = params["root_embedding_size"]
    num_encoder_layers = params["num_encoder_layers"]
    num_decoder_layers = params["num_decoder_layers"]
    num_heads = params["num_heads"]
    dropout = params["dropout"]
    context_frames = params["context_frames"]
    target_frames = params["target_frames"]
    WINDOW_SIZE = params["context_frames"] + params["hole_frames"] + params["target_frames"]
    max_len = max(64, WINDOW_SIZE)
    INTERPOLATE_BEFORE_PREDICTION = params.get("interpolate_before_prediction", False)

    # Load input BVH to get number of joints for model instantiation
    anim = load_bvh(args.bvh_in)
    num_joints = len(anim.names)

    model = MotionTransformer(
        num_joints=num_joints,
        joint_embedding_size=joint_embedding_size,
        root_embedding_size=root_embedding_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_len=max_len
    )

    ckpt = torch.load(args.weights, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # temp_file_in = args.bvh_in
    # total_frames = get_bvh_frames(temp_file_in)

    predict_bvh_loop(
        model=model,
        bvh_path_in=args.bvh_in,
        bvh_path_out=args.bvh_out,
        window_size=WINDOW_SIZE,
        period=args.period,
        context_frames=context_frames,
        target_frames=target_frames,
        preinterpolate=INTERPOLATE_BEFORE_PREDICTION
    )
    print("\nPrediction completed.")