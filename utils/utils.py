import torch
import json
import time


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_params_from_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        params = json.load(f)
        
    if "description" in params:
        del params["description"]
    return params


'''
Forward Kinematics
'''

def forward_kinematics(
    rot_mats: torch.Tensor,
    root_pos: torch.Tensor,
    parents: list,
    offsets: torch.Tensor,
    return_rot_mats: bool = False
) -> torch.Tensor:
    """
    rot_mats: (B,T,J,3,3)
    root_pos: (B,T,3)
    offsets: (J,3)

    Returns:
        joint_positions: (B,T,J,3) - joint world positions
    """
    B, T, J, _, _ = rot_mats.shape
    device = rot_mats.device

    global_rot_mats = [rot_mats[:, :, 0, :, :]]
    joint_positions = [root_pos]

    for j in range(1, J):
        parent_idx = parents[j]
        parent_pos = joint_positions[parent_idx]    # (B,T,3)
        
        # Get global rotation
        parent_global_rot = global_rot_mats[parent_idx]
        local_rot = rot_mats[:, :, j, :, :]
        global_rot = torch.matmul(parent_global_rot, local_rot)  # (B,T,3,3)
        global_rot_mats.append(global_rot)
        
        # Get offset and rotate it
        offset = offsets[j].to(device)                      # (3,)
        rotated_offset = torch.matmul(parent_global_rot, offset.view(3, 1)).squeeze(-1)  # (B,T,3)
        
        # Compute world position
        world_pos = parent_pos + rotated_offset    # (B,T,3)
        joint_positions.append(world_pos)

    if return_rot_mats:
        return torch.stack(joint_positions, dim=2), torch.stack(global_rot_mats, dim=2)
    else:
        return torch.stack(joint_positions, dim=2)
#forward_kinematics


'''
Logging
'''

def show_warning(log_message: str):
    print(f"{'\033[93m'}WARNING: {log_message}")
    for _ in range(3):
        time.sleep(0.7)
        print(".", flush=True)
    time.sleep(0.5)
    for _ in range(5):
        print(".", end='', flush=True)
        time.sleep(0.2)
    print(f"{'\033[0m'}")
#show_warning