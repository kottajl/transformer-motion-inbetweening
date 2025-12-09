from tqdm import tqdm
from torch.utils.data import DataLoader

from model.model import MotionTransformer
from model.loss.FKLoss import FKLoss
from dataset import BvhDataset
from interpolation import interpolate_rotations, interpolate_positions

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import datetime

from utils import load_params_from_json    


def train(params: dict):
    CONFIG_NAME = params.get("config_name", "new_config")

    CONTEXT_FRAMES = params["context_frames"]
    HOLE_FRAMES = params["hole_frames"]
    TARGET_FRAMES = params["target_frames"]
    BATCH_SIZE = params["batch_size"]
    N_EPOCHS = params["n_epochs"]
    PATIENCE = params["patience"]
    VAL_RATIO = params["val_ratio"]
    WINDOW_SIZE = CONTEXT_FRAMES + HOLE_FRAMES + TARGET_FRAMES
    WINDOW_STEP = params["window_step"]
    OPTIMIZER_LR = params["optimizer_lr"]
    JOINT_EMBEDDING_SIZE = params["joint_embedding_size"]
    ROOT_EMBEDDING_SIZE = params["root_embedding_size"]
    NUM_ENCODER_LAYERS = params["num_encoder_layers"]
    NUM_DECODER_LAYERS = params["num_decoder_layers"]
    NUM_HEADS = params["num_heads"]
    DROPOUT = params["dropout"]
    MAX_LEN = max(64, WINDOW_SIZE)

    INTERPOLATE_BEFORE_PREDICTION = params.get("interpolate_before_prediction", False)

    LOSS_WEIGHTS = params["loss_weights"]
    assert abs(sum(LOSS_WEIGHTS.values()) - 1.0) < 1e-6, f"LOSS_WEIGHTS values must sum to 1.0: got {sum(LOSS_WEIGHTS.values())}"
    assert "rot_6d" in LOSS_WEIGHTS and "pos" in LOSS_WEIGHTS, "LOSS_WEIGHTS must contain 'rot_6d' and 'pos' keys."

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    with open(f'generated_models/{CONFIG_NAME}.log', 'w') as file:
        file.write('')
    with open('train_log.txt', 'a') as file:
        file.write(f'Starting training with config: {CONFIG_NAME} at {datetime.datetime.now()}\n')

    dataset = BvhDataset(
        "datasets/lafan1/processed/",
        window=WINDOW_SIZE,
        step=WINDOW_STEP,
        device=DEVICE,
        interpolate_missing=INTERPOLATE_BEFORE_PREDICTION
    )

    fk_loss_fn = FKLoss(dataset.parents).to(DEVICE)
    offsets_tensor = torch.tensor(dataset.offsets, device=DEVICE)

    val_dataset_size = int(len(dataset) * VAL_RATIO)
    train_dataset_size = len(dataset) - val_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_dataset_size, val_dataset_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MotionTransformer(
        num_joints=len(dataset.parents),
        joint_embedding_size=JOINT_EMBEDDING_SIZE,
        root_embedding_size=ROOT_EMBEDDING_SIZE,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=OPTIMIZER_LR)

    # List of context + target frames indices
    fixed_points = list(range(0, CONTEXT_FRAMES))
    fixed_points.extend(list(range(WINDOW_SIZE - TARGET_FRAMES, WINDOW_SIZE)))

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{N_EPOCHS} - training", 
            leave=False
        )
        for batch in loop:
            rot = batch["rotations"].to(DEVICE)
            pos = batch["positions"].to(DEVICE)
            B, T, J, D = rot.shape      # batches, frames(time), joints, data_dimension
            assert T == WINDOW_SIZE, f"Expected window size {WINDOW_SIZE}, but got {T}"

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

            optimizer.zero_grad()
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
                rot, pos,
                pred_rot, pred_pos,
                offsets=offsets_tensor
            )

            loss = (
                LOSS_WEIGHTS["rot_6d"] * loss_rot + 
                LOSS_WEIGHTS["pos"] * loss_pos + 
                LOSS_WEIGHTS["fk"] * fk_loss
            )
            
            loss.backward()
            optimizer.step()

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
            for batch in loop:
                rot = batch["rotations"].to(DEVICE)
                pos = batch["positions"].to(DEVICE)
                B, T, J, D = rot.shape    # batches, frames(time), joints, data_dimension

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
                    rot, pos,
                    pred_rot, pred_pos,
                    offsets=offsets_tensor
                )

                loss = (
                    LOSS_WEIGHTS["rot_6d"] * loss_rot + 
                    LOSS_WEIGHTS["pos"] * loss_pos + 
                    LOSS_WEIGHTS["fk"] * fk_loss
                )

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)

        # Print results
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")
        with open('train_log.txt', 'a') as file:
            file.write(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}\n')
        with open(f'generated_models/{CONFIG_NAME}.log', 'a') as file:
            file.write(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}\n')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', 
        type=str, 
        required=True,
        help='Name of the JSON file with training parameters (without .json extension, must be in the config/ folder)'
    )
    args = parser.parse_args()
    try:
        params = load_params_from_json("configs/" + args.params + ".json")
    except FileNotFoundError:
        print(f"Error: The file '{args.params}.json' was not found in the config/ folder.")
        exit(1)

    train(params)