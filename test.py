import argparse
import torch
import metrics as metrics
import numpy as np

from tqdm import tqdm
from dataset import BvhDataset
from interpolation import interpolate_positions, interpolate_rotations
from model.model import MotionTransformer
from utils import forward_kinematics, load_params_from_json
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from utils import rot6d_to_mat_torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fetch_model(model_file: str, params: dict, n_joints: int, window_size: int) -> MotionTransformer:

    try:
        model = MotionTransformer(
            num_joints=n_joints,
            joint_embedding_size=params['joint_embedding_size'],
            root_embedding_size=params['root_embedding_size'],
            num_encoder_layers=params['num_encoder_layers'],
            num_decoder_layers=params['num_decoder_layers'],
            num_heads=params['num_heads'],
            dropout=params['dropout'],
            velocity_included=params.get("velocity_included", False),
            pe_type=params.get("pe_type", "sinusoidal"),
            # max_len=max(64, window_size)
            max_len=256
        )
    except KeyError as e:
        print(f"Error: Missing key in parameters: {e}")
        exit(-1)

    ckpt = torch.load(model_file, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE)

    return model
#fetch_model


def test_and_get_scores(
    model: MotionTransformer,
    dataset: BvhDataset,
    params: dict,
    hole_frames: int,
    window_step: int,
    data_subset_type: str,
    # moves_names: list
) -> dict:
    scores = dict()
    
    # Get params
    try:
        CONTEXT_FRAMES = params["context_frames"]
        HOLE_FRAMES = hole_frames
        TARGET_FRAMES = params["target_frames"]
        BATCH_SIZE = params["batch_size"]
        INTERPOLATE_BEFORE_PREDICTION = params.get("interpolate_before_prediction", False)
        WINDOW_SIZE = CONTEXT_FRAMES + HOLE_FRAMES + TARGET_FRAMES
    
    except KeyError as e:
        print(f"Error: Missing key in parameters: {e}")
        exit(-1)
    
    hole_start = CONTEXT_FRAMES
    hole_end = CONTEXT_FRAMES + HOLE_FRAMES
    print(f"Operating on hole size: {hole_end - hole_start} frames (from {hole_start} to {hole_end-1} in the window)")
    
    # List of context + target frames indices
    fixed_points = list(range(0, CONTEXT_FRAMES))
    fixed_points.extend(list(range(WINDOW_SIZE - TARGET_FRAMES, WINDOW_SIZE)))
    
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    model.eval()

    total_l2p = 0.0
    total_l2q = 0.0
    total_npss = 0.0
    total_samples = 0

    loop = tqdm(
        data_loader,
        desc=f"Testing model", 
        leave=False
    )
    with torch.no_grad():
        for batch in loop:
            rot = batch["rotations"].to(DEVICE)
            pos = batch["positions"].to(DEVICE)
            B, T, J, D = rot.shape    # batches, frames(time), joints, data_dimension
            assert B == pos.size(0), "Batch size mismatch between rotations and positions"

            # Center root position to the first frame of the window
            root_offset = pos[:, 0:1, :].clone()   # (B, T, 1, 3)
            pos -= root_offset

            # Make a hole
            src_rot = rot.clone()
            src_pos = pos.clone()
            hole_start, hole_end = CONTEXT_FRAMES, T - TARGET_FRAMES
            assert hole_end > hole_start, f"Hole length is 0 or less [{hole_start}, {hole_end})"
            src_rot[:, hole_start:hole_end, :, :] = 0.0
            src_pos[:, hole_start:hole_end, :] = 0.0

            if INTERPOLATE_BEFORE_PREDICTION:
                src_rot_q = batch["rotations_quat"].to(DEVICE)
                src_rot_q[:, hole_start:hole_end, :, :] = 0.0
                with torch.no_grad():
                    src_rot, _ = interpolate_rotations(
                        src_rot,
                        src_rot_q,
                        hole_start,
                        hole_end
                    )
                    src_pos = interpolate_positions(
                        src_pos,
                        hole_start,
                        hole_end
                    )
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_rot, pred_pos = model(
                    src_rot, src_pos,
                    # src_rot.clone(), src_pos.clone(),
                    fixed_points=fixed_points
                )
            
            # Return to original position space
            pos += root_offset
            pred_pos += root_offset

            gt_rot_mats = rot6d_to_mat_torch(rot)
            pred_rot_mats = rot6d_to_mat_torch(pred_rot)

            # Compute forward kinematics to get joint positions
            pred_pos_fk, pred_rot_mats_fk = forward_kinematics(
                pred_rot_mats,
                pred_pos,
                dataset.parents,
                torch.tensor(dataset.offsets, device=DEVICE),
                return_rot_mats=True
            )
            gt_pos_fk, gt_rot_mats_fk = forward_kinematics(
                gt_rot_mats,
                pos,
                dataset.parents,
                torch.tensor(dataset.offsets, device=DEVICE),
                return_rot_mats=True
            )

            # Compute predicted global quaternions for L2Q metric
            gt_rot_mats_fk_np = gt_rot_mats_fk.cpu().numpy().reshape(-1, 3, 3)
            gt_rot_q_fk_np = R.from_matrix(gt_rot_mats_fk_np).as_quat()
            gt_rot_fk_q = torch.tensor(gt_rot_q_fk_np, device=DEVICE).view(B, T, J, 4)

            # Compute gt global quaternions for L2Q metric
            pred_rot_mats_fk_np = pred_rot_mats_fk.cpu().numpy().reshape(-1, 3, 3)
            pred_rot_q_fk_np = R.from_matrix(pred_rot_mats_fk_np).as_quat()
            pred_rot_fk_q = torch.tensor(pred_rot_q_fk_np, device=DEVICE).view(B, T, J, 4)

            # --- METRICS ---

            # - l2 position error (L2P)
            batch_l2p = metrics.l2p(pred_pos_fk[:, hole_start:hole_end, :, :], gt_pos_fk[:, hole_start:hole_end, :, :])
            
            # - l2 quaternion error (L2Q)
            batch_l2q = metrics.l2q(pred_rot_fk_q[:, hole_start:hole_end, :, :], gt_rot_fk_q[:, hole_start:hole_end, :, :])

            # - npss (Normalized Power Spectrum Similarity) - on global rotations
            batch_npss = metrics.npss(
                pred_rot_fk_q[:, hole_start:hole_end, :, :].reshape(B, hole_end - hole_start, n_joints * 4),
                gt_rot_fk_q[:, hole_start:hole_end, :, :].reshape(B, hole_end - hole_start, n_joints * 4)
            )

            # Aggregate scores
            total_l2p += batch_l2p * B
            total_l2q += batch_l2q * B
            total_npss += batch_npss * B
            total_samples += B
    
    scores['l2p'] = total_l2p / total_samples
    scores['l2q'] = total_l2q / total_samples
    scores['npss'] = total_npss / total_samples

    return scores
#test_and_get_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="best_model.pt", help='Path to model weights (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--window_step', type=int, default=-1, help='Step size for sliding window over the data')
    parser.add_argument('--hole_size', type=int, default=-1, help='Hole size to test (overrides config if set)')
    # parser.add_argument('--data_subset_type', type=str, default='all', help='Subset of data to use for training (e.g., "all", "selected-moves", etc.)')
    args = parser.parse_args()

    # v Literal['all', 'selected-subjects', 'selected-moves', 'selected-subjects-and-moves', 'selected-files'] v
    data_subset_type = 'all'
    # subjects_indices = [5]
    # moves_names: list = ['fallAndGetUp', 'jumps1']

    # Load model parameters from JSON
    try:
        params = load_params_from_json(args.config)
    except FileNotFoundError:
        print(f"Error: The file '{args.config}' was not found.")
        exit(1)
    
    window_step = args.window_step
    if window_step == -1:
        window_step = params["window_step"]
        print(f"Using default window_step: {window_step}")
    
    # Create dataset and extract number of joints for model initialization
    try:
        if args.hole_size != -1:
            HOLE_FRAMES = args.hole_size
        else:
            if isinstance(args.hole_size, int):
                HOLE_FRAMES = args.hole_size
            elif isinstance(args.hole_size, list) and len(args.hole_size) == 2:
                HOLE_FRAMES = args.hole_size[1]
            else:
                raise ValueError("Invalid hole_size argument. Must be an integer or a list of two integers.")
            print(f"'hole_size' parameter not specified. Using hole_frames: {HOLE_FRAMES}")
        
        WINDOW_SIZE = params["context_frames"] + HOLE_FRAMES + params["target_frames"]

        dataset = BvhDataset(
            "datasets/lafan1/test_processed/",
            window=WINDOW_SIZE,
            step=window_step,
            device=DEVICE,
            interpolate_missing=params["interpolate_before_prediction"],
            subset_type=data_subset_type,
            # subjects_indices=subjects_indices
        )
        n_joints = dataset.get_num_of_joints()
        print(f"Number of joints: {n_joints}")
    except KeyError as e:
        print(f"Error: Missing key in parameters: {e}")
        exit(-1)    

    # Get model
    model = fetch_model(
        model_file=args.weights, 
        params=params,
        n_joints=n_joints,
        window_size=WINDOW_SIZE
    )
    
    scores = test_and_get_scores(
        model=model,
        dataset=dataset,
        params=params,
        window_step=window_step,
        data_subset_type=data_subset_type,
        hole_frames=HOLE_FRAMES
        # moves_names=moves_names
    )
    print("Testing completed.")

    print(f"Test Results for model {args.weights} (window step: {window_step}):")
    for metric_name, metric_value in scores.items():
        print(f"  |- {metric_name}: {metric_value:.4f}")