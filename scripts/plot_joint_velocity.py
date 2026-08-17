import sys

import matplotlib.pyplot as plt
import logging

import torch

from utils.bvh_parser import load_bvh
from utils.animation import Animation
from utils.rotation_convertion import rot6d_to_mat_torch
from utils.utils import forward_kinematics, to_tensor

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAMES_A = 495
FRAMES_B = 525

# FRAMES_A = 1495
# FRAMES_B = 1525

JOINTS = [
    # "RightHand",
    "Neck"
]
CONCAT_TYPE = "max"


def get_joint_velocities(
    animation: Animation, 
    frames_start: int,
    frames_end: int,
    chosen_joints: list = None, # None = all joints
    concat_type: str = "mean"       # ["mean", "max", "sum", "min", "cat", None]
):
    joint_velocities = {}
    for joint_name in animation.names:
        if chosen_joints is None or joint_name in chosen_joints:
            joint_velocities[joint_name] = []
            joint_idx = animation.names.index(joint_name)

            rot_6d = to_tensor(animation.rotations_6d[frames_start:frames_end, :, :], device=DEVICE)    # (T, J, 6)
            rot_6d = rot_6d.unsqueeze(0)                                # Add dummy batch dimension -> (1, T, J, 6)
            rot_mat = rot6d_to_mat_torch(rot_6d)                                                    # (1, T, J, 3, 3)
            pos = to_tensor(animation.positions[frames_start:frames_end, :], device=DEVICE)             # (T, J, 3)
            pos = pos.unsqueeze(0)                                      # Add dummy batch dimension -> (1, T, J, 3)

            joint_pos = forward_kinematics(
                rot_mat,    
                pos,
                parents=animation.parents,
                offsets=torch.tensor(animation.offsets, device=DEVICE)
            )   # (1, T, J, 3)
            joint_pos = joint_pos.squeeze(0)  # Remove dummy batch dimension -> (T, J, 3)
            T, J, _ = joint_pos.shape

            for frame in range(T - 1):
                pos_a = joint_pos[frame, joint_idx, :]
                pos_b = joint_pos[frame + 1, joint_idx, :]
                velocity = (pos_b - pos_a).cpu().numpy()
                velocity = (velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2) ** 0.5  # Euclidean norm
                joint_velocities[joint_name].append(velocity)

    n_joints = len(joint_velocities)
    print(f"Number of joints: {n_joints}", file=sys.stderr)

    if concat_type is None or concat_type == "none" or concat_type == "cat":
        return joint_velocities

    cat_velocities = []
    for frame in range(frames_start, frames_end - 1):
        frame_velocities = [joint_velocities[joint_name][frame - frames_start] for joint_name in joint_velocities]

        if concat_type == "mean":
            cat_velocity = sum(frame_velocities) / len(frame_velocities)
        elif concat_type == "max":
            cat_velocity = max(frame_velocities)
        elif concat_type == "sum":
            cat_velocity = sum(frame_velocities)
        elif concat_type == "min":
            cat_velocity = min(frame_velocities)
        else:
            raise ValueError(f"Invalid concat_type: {concat_type}. Must be 'mean', 'max', 'sum', or 'min'.")

        cat_velocities.append(cat_velocity)

    return cat_velocities


def main():
    GT_BVH_PATH = "eval/dance2_subject5.bvh"
    test_bvhs = [
        ("Ostatnia wartość", "eval/dance2_bef_vel_rot_20.bvh"),
        ("Średnia brzegowa", "eval/dance2_avg_vel_rot_20.bvh"),
        ("LERP + SLERP",     "eval/dance2_int_vel_rot_20.bvh")
    ]

    velocities = {}
    for bvh_info in [("Oryginalna animacja", GT_BVH_PATH)] + test_bvhs:
        bvh_key, bvh_path = bvh_info
        anim = load_bvh(bvh_path)
        velocities[bvh_key] = get_joint_velocities(
            animation=anim,
            frames_start=FRAMES_A,
            frames_end=FRAMES_B,
            chosen_joints=JOINTS,
            concat_type=CONCAT_TYPE
        )

    # Plotting
    x = range(FRAMES_A, FRAMES_B)
    plt.figure(figsize=(8, 6))
    for idx, (bvh_key, joint_velocities) in enumerate(velocities.items()):
        if idx == 0:
            plt.plot(x[1:], joint_velocities, label=bvh_key, color='black', linewidth=2.5)
        else:
            plt.plot(x[1:], joint_velocities, label=bvh_key)
    if len(JOINTS) == 1:
        plt.title(f"Prędkość stawu '{JOINTS[0]}' (klatki {FRAMES_A}-{FRAMES_B})")
    else:
        plt.title(f"Prędkości stawów {", ".join(JOINTS)} (klatki {FRAMES_A}-{FRAMES_B})")
    plt.xlabel("Klatka")
    plt.ylabel("Prędkość (norma euklidesowa)")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f"results/figures/joint_velocity.png")
    plt.close()


if __name__ == "__main__":
    main()
