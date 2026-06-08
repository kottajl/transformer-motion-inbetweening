from tqdm import tqdm
from torch.utils.data import DataLoader

from model.loss.FKVelocityBoundaryLoss import FKVelocityBoundaryLoss, root_pos_velocity_boundary_loss
from model.loss.SmoothnessLoss import SmoothnessLoss
from model.model import MotionTransformer
from model.loss.FKLoss import FKLoss
from dataset import BvhDataset
from interpolation import interpolate_rotations, interpolate_positions

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import datetime
import time

from utils import load_params_from_json    


def train(params: dict, full_log: bool = False, data_subset_type: str = 'all', **subset_kwargs):
    CONFIG_NAME = params.get("config_name", "new_config")

    CONTEXT_FRAMES = params["context_frames"]
    TARGET_FRAMES = params["target_frames"]
    BATCH_SIZE = params["batch_size"]
    N_EPOCHS = params["n_epochs"]
    PATIENCE = params["patience"]
    VAL_RATIO = params["val_ratio"]
    WINDOW_STEP = params["window_step"]
    OPTIMIZER_LR = params["optimizer_lr"]
    JOINT_EMBEDDING_SIZE = params["joint_embedding_size"]
    ROOT_EMBEDDING_SIZE = params["root_embedding_size"]
    NUM_ENCODER_LAYERS = params["num_encoder_layers"]
    NUM_DECODER_LAYERS = params["num_decoder_layers"]
    NUM_HEADS = params["num_heads"]
    DROPOUT = params["dropout"]

    # Get hole frames (check if fixed or variable)
    HOLE_FRAMES = params["hole_frames"]
    if isinstance(HOLE_FRAMES, int):
        min_hole_frames = HOLE_FRAMES
        max_hole_frames = HOLE_FRAMES
        print(f"Using fixed hole size of {max_hole_frames} frames.")
    elif isinstance(HOLE_FRAMES, list) and len(HOLE_FRAMES) == 2:
        min_hole_frames = HOLE_FRAMES[0]
        max_hole_frames = HOLE_FRAMES[1]
        print(f"Using variable hole size with values between {min_hole_frames} and {max_hole_frames} frames.")
    else:
        raise ValueError("Invalid 'hole_frames' parameter. Must be either an integer or a list of two integers [min, max].")
    WINDOW_SIZE = CONTEXT_FRAMES + max_hole_frames + TARGET_FRAMES
    MAX_LEN = max(64, WINDOW_SIZE)
    print(f"Maximum sequence length: {MAX_LEN}")

    VELOCITY_INCLUDED = params.get("velocity_included", False)

    PE_TYPE = params.get("pe_type", "sinusoidal")
    INTERPOLATE_BEFORE_PREDICTION = params.get("interpolate_before_prediction", False)

    LOSS_WEIGHTS = params["loss_weights"]
    # assert abs(sum(LOSS_WEIGHTS.values()) - 1.0) < 1e-6, f"LOSS_WEIGHTS values must sum to 1.0: got {sum(LOSS_WEIGHTS.values())}"
    assert "rot_6d" in LOSS_WEIGHTS and "pos" in LOSS_WEIGHTS, "LOSS_WEIGHTS must contain 'rot_6d' and 'pos' keys."

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    with open(f'generated_models/{CONFIG_NAME}.log', 'w') as file:
        file.write('')
    with open('train_log.txt', 'a') as file:
        file.write(f'Starting training with config: {CONFIG_NAME} at {datetime.datetime.now()}\n')

    dataset = BvhDataset(
        "datasets/lafan1/processed/",
        window=WINDOW_SIZE,
        step=WINDOW_STEP,
        # device=DEVICE,
        device='cpu',   # keep data on CPU, move to GPU in training loop
        interpolate_missing=INTERPOLATE_BEFORE_PREDICTION,
        subset_type=data_subset_type,
        **subset_kwargs
    )

    fk_loss_fn = FKLoss(
        dataset.parents, 
        weighted=params.get("fk_weighted", False)
    ).to(DEVICE)
    fk_vel_bnd_loss_fn = FKVelocityBoundaryLoss(dataset.parents).to(DEVICE)
    smoothness_loss_fn = SmoothnessLoss(dataset.parents).to(DEVICE)
    offsets_tensor = torch.tensor(dataset.offsets, device=DEVICE)

    val_dataset_size = int(len(dataset) * VAL_RATIO)
    train_dataset_size = len(dataset) - val_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_dataset_size, val_dataset_size]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        # num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        # num_workers=2,
        pin_memory=True
    )
    
    model = MotionTransformer(
        num_joints=len(dataset.parents),
        joint_embedding_size=JOINT_EMBEDDING_SIZE,
        root_embedding_size=ROOT_EMBEDDING_SIZE,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        velocity_included=VELOCITY_INCLUDED,
        pe_type=PE_TYPE,
        max_len=MAX_LEN
    ).to(DEVICE)
    # if torch.cuda.is_available():
        # model = torch.compile(model, backend="cudagraphs")
        # model = torch.compile(model)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.Adam(model.parameters(), lr=OPTIMIZER_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # List of context + target frames indices   -- REMOVED FOR VARIABLE HOLE SIZE, COMPUTED IN TRAINING LOOP INSTEAD
    # fixed_points = list(range(0, CONTEXT_FRAMES))
    # fixed_points.extend(list(range(WINDOW_SIZE - TARGET_FRAMES, WINDOW_SIZE)))

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0

        EPOCH_HOLE_SIZE_MULTIPLIER = 2
        START_HOLE_SIZE = 2
        epoch_max_hole_frames = min(
            min_hole_frames + START_HOLE_SIZE + EPOCH_HOLE_SIZE_MULTIPLIER * epoch,
            max_hole_frames
        )   # If hole size is fixed, this will still work ok.

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{N_EPOCHS} - training", 
            leave=False
        )
        for batch in loop:
            # Get random hole size for this batch
            batch_hole_frames = torch.randint(min_hole_frames, epoch_max_hole_frames + 1, (1,)).item()
            BATCH_WINDOW_SIZE = CONTEXT_FRAMES + batch_hole_frames + TARGET_FRAMES

            rot = batch["rotations"][:, :BATCH_WINDOW_SIZE, :, :].to(DEVICE)
            pos = batch["positions"][:, :BATCH_WINDOW_SIZE, :].to(DEVICE)
            B, T, J, D = rot.shape      # batches, frames(time), joints, data_dimension
            assert T == BATCH_WINDOW_SIZE, f"Expected window size {BATCH_WINDOW_SIZE}, but got {T}"

            # Get fixed points
            fixed_points = list(range(0, CONTEXT_FRAMES))
            fixed_points.extend(list(range(BATCH_WINDOW_SIZE - TARGET_FRAMES, BATCH_WINDOW_SIZE)))

            # Center root position to the first frame of the window
            root_offset = pos[:, 0:1, :].clone()   # (B, T, 3)
            pos -= root_offset

            # Make a hole
            src_rot = rot.clone()
            src_pos = pos.clone()
            hole_start, hole_end = CONTEXT_FRAMES, T - TARGET_FRAMES
            assert hole_end > hole_start, f"Hole length is 0 or less [{hole_start}, {hole_end})"
            src_rot[:, hole_start:hole_end, :, :] = 0.0
            src_pos[:, hole_start:hole_end, :] = 0.0

            if INTERPOLATE_BEFORE_PREDICTION:
                src_rot_q = batch["rotations_quat"][:, :BATCH_WINDOW_SIZE, :, :].to(DEVICE)
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

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_rot, pred_pos = model(
                    src_rot, src_pos,
                    # src_rot.clone(), src_pos.clone(),
                    fixed_points=fixed_points
                )

            # Compute loss - only on predicted frames in hole
            mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
            mask[fixed_points] = False
            
            mask_rot = mask.view(1, T, 1, 1).expand(B, T, J, D)
            loss_rot = F.l1_loss(pred_rot[mask_rot], rot[mask_rot])

            mask_pos = mask.view(1, T, 1).expand(B, T, 3)
            loss_pos = F.l1_loss(pred_pos[mask_pos], pos[mask_pos])

            fk_loss = fk_loss_fn(
                rot[:, hole_start:hole_end, :, :], pos[:, hole_start:hole_end, :],
                pred_rot[:, hole_start:hole_end, :, :], pred_pos[:, hole_start:hole_end, :],
                offsets=offsets_tensor
            )

            smoothness_loss = smoothness_loss_fn(
                pred_rot[:, hole_start-1:hole_end+1, :, :], pred_pos[:, hole_start-1:hole_end+1, :], 
                offsets=offsets_tensor
            )

            # pos_vel_bnd_loss = root_pos_velocity_boundary_loss(
            #     pos, pred_pos,
            #     hole_start=hole_start,
            #     hole_end=hole_end
            # )

            # fk_vel_bnd_loss = fk_vel_bnd_loss_fn(
            #     rot, pos,
            #     pred_rot, pred_pos,
            #     offsets=offsets_tensor,
            #     hole_start=hole_start,
            #     hole_end=hole_end
            # )

            loss = (
                LOSS_WEIGHTS["rot_6d"] * loss_rot + 
                LOSS_WEIGHTS["pos"] * loss_pos + 
                LOSS_WEIGHTS["fk"] * fk_loss +
                LOSS_WEIGHTS.get("smoothness", 0.0) * smoothness_loss

                # LOSS_WEIGHTS["pos_vel_bnd"] * pos_vel_bnd_loss +
                # LOSS_WEIGHTS["fk_vel_bnd"] * fk_vel_bnd_loss
            )
            train_loss_coponents = {
                "loss_rot": loss_rot.item(),
                "loss_pos": loss_pos.item(),
                "fk_loss": fk_loss.item(),
                "smoothness_loss": smoothness_loss.item(),
                # "pos_vel_bnd_loss": pos_vel_bnd_loss.item(),
                # "fk_vel_bnd_loss": fk_vel_bnd_loss.item()
            }
            
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.2f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0

        loop = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{N_EPOCHS} - validation", 
            leave=False
        )
        with torch.no_grad():
            # Get fixed points for every batch (hole size here is always max_hole_frames)
            BATCH_WINDOW_SIZE = CONTEXT_FRAMES + max_hole_frames + TARGET_FRAMES
            fixed_points = list(range(0, CONTEXT_FRAMES))
            fixed_points.extend(list(range(BATCH_WINDOW_SIZE - TARGET_FRAMES, BATCH_WINDOW_SIZE)))

            for batch in loop:
                rot = batch["rotations"].to(DEVICE)
                pos = batch["positions"].to(DEVICE)
                B, T, J, D = rot.shape    # batches, frames(time), joints, data_dimension

                # Center positions around root joint
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

                # Compute loss - only on predicted frames in hole
                mask = torch.ones(T, dtype=torch.bool, device=DEVICE)
                mask[fixed_points] = False
                
                mask_rot = mask.view(1, T, 1, 1).expand(B, T, J, D)
                loss_rot = F.l1_loss(pred_rot[mask_rot], rot[mask_rot])

                mask_pos = mask.view(1, T, 1).expand(B, T, 3)
                loss_pos = F.l1_loss(pred_pos[mask_pos], pos[mask_pos])

                fk_loss = fk_loss_fn(
                    rot[:, hole_start:hole_end, :, :], pos[:, hole_start:hole_end, :],
                    pred_rot[:, hole_start:hole_end, :, :], pred_pos[:, hole_start:hole_end, :],
                    offsets=offsets_tensor
                )

                smoothness_loss = smoothness_loss_fn(
                    pred_rot[:, hole_start-1:hole_end+1, :, :], pred_pos[:, hole_start-1:hole_end+1, :], 
                    offsets=offsets_tensor
                )

                # pos_vel_bnd_loss = root_pos_velocity_boundary_loss(
                #     pos, pred_pos,
                #     hole_start=hole_start,
                #     hole_end=hole_end
                # )

                # fk_vel_bnd_loss = fk_vel_bnd_loss_fn(
                #     rot, pos,
                #     pred_rot, pred_pos,
                #     offsets=offsets_tensor,
                #     hole_start=hole_start,
                #     hole_end=hole_end
                # )

                loss = (
                    LOSS_WEIGHTS["rot_6d"] * loss_rot + 
                    LOSS_WEIGHTS["pos"] * loss_pos + 
                    LOSS_WEIGHTS["fk"] * fk_loss + 
                    LOSS_WEIGHTS.get("smoothness", 0.0) * smoothness_loss

                    # LOSS_WEIGHTS["pos_vel_bnd"] * pos_vel_bnd_loss +
                    # LOSS_WEIGHTS["fk_vel_bnd"] * fk_vel_bnd_loss
                )

                test_loss_coponents = {
                    "loss_rot": loss_rot.item(),
                    "loss_pos": loss_pos.item(),
                    "fk_loss": fk_loss.item(),
                    "smoothness_loss": smoothness_loss.item(), 
                    # "pos_vel_bnd_loss": pos_vel_bnd_loss.item(),
                    # "fk_vel_bnd_loss": fk_vel_bnd_loss.item()
                }

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)

        # Generate log string
        log_epoch_str = f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}\n"
        if full_log:    # add concreate loss components
            log_epoch_str += "Train Loss Components:\n"
            for key, value in train_loss_coponents.items():
                log_epoch_str += f"  |- {key}: {value:.5f}\n"
            log_epoch_str += "Val Loss Components:\n"
            for key, value in test_loss_coponents.items():
                log_epoch_str += f"  |- {key}: {value:.5f}\n"
            log_epoch_str += f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            log_epoch_str += f"Hole frames in this epoch: [{min_hole_frames}, {epoch_max_hole_frames}]\n"
            log_epoch_str += "\n"

        # Print results
        print(log_epoch_str)
        with open('train_log.txt', 'a') as file:
            file.write(log_epoch_str)
        with open(f'generated_models/{CONFIG_NAME}.log', 'a') as file:
            file.write(log_epoch_str)        
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            torch.save(model.state_dict(), f"generated_models/{CONFIG_NAME}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', type=str, required=True,
        help='Name of the JSON file with training parameters (without .json extension, must be in the config/ folder)'
    )
    parser.add_argument('--full_log', action='store_true', help='Enable full logging during training (default: False)')
    # parser.add_argument('--data_subset_type', type=str, default='all', help='Subset of data to use for training (e.g., "all", "selected-moves", etc.)')

    # v Literal['all', 'selected-subjects', 'selected-moves', 'selected-subjects-and-moves', 'selected-files'] v
    data_subset_type = 'all'
    # subjects_indices = [1, 2, 3, 4]
    # moves_names: list = ['fallAndGetUp', 'jumps1']
    
    args = parser.parse_args()
    try:
        params = load_params_from_json("configs/" + args.params + ".json")
    except FileNotFoundError:
        print(f"Error: The file '{args.params}.json' was not found in the config/ folder.")
        exit(1)
    
    if not args.full_log:
        print(f"{'\033[93m'}WARNING: Full logging is disabled! This means that detailed loss components and learning rate information will not be logged. Enable full logging with the --full_log flag for more insights during training.")     
        for _ in range(3):
            time.sleep(0.7)
            print(".", flush=True)
        time.sleep(0.5)
        for _ in range(5):
            print(".", end='', flush=True)
            time.sleep(0.2)
        print(f"{'\033[0m'}")

    train(params, full_log=args.full_log, data_subset_type=data_subset_type)