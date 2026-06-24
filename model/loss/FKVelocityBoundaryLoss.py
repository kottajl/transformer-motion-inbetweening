from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation_convertion import rot6d_to_mat_torch
from utils.utils import forward_kinematics


class FKVelocityBoundaryLoss(nn.Module):
    def __init__(self, parents):
        super(FKVelocityBoundaryLoss, self).__init__()
        self.parents = parents
    
    def forward(
        self,
        gt_rot6d: torch.Tensor,
        gt_pos: torch.Tensor,
        pred_rot6d: torch.Tensor,
        pred_pos: torch.Tensor,
        offsets: torch.Tensor,
        hole_start: int,
        hole_end: int
    ) -> torch.Tensor:
        
        gt_rot_mats = rot6d_to_mat_torch(gt_rot6d)      # (B,T,J,3,3)
        pred_rot_mats = rot6d_to_mat_torch(pred_rot6d)  # (B,T,J,3,3)

        gt_joint_pos = forward_kinematics(
            gt_rot_mats,    
            gt_pos,
            parents=self.parents,
            offsets=offsets
        )   # (B,T,J,3)
        pred_joint_pos = forward_kinematics(
            pred_rot_mats,
            pred_pos,
            parents=self.parents,
            offsets=offsets
        )   # (B,T,J,3)

        # Left boundary velocity
        gt_vel_left = gt_joint_pos[:, hole_start, :, :] - gt_joint_pos[:, hole_start - 1, :, :]         # (B,J,3)
        pred_vel_left = pred_joint_pos[:, hole_start, :, :] - pred_joint_pos[:, hole_start - 1, :, :]   # (B,J,3)

        # Right boundary velocity
        gt_vel_right = gt_joint_pos[:, hole_end, :, :] - gt_joint_pos[:, hole_end - 1, :, :]        # (B,J,3)
        pred_vel_right = pred_joint_pos[:, hole_end, :, :] - pred_joint_pos[:, hole_end - 1, :, :]  # (B,J,3)

        loss = F.l1_loss(pred_vel_left, gt_vel_left) + F.l1_loss(pred_vel_right, gt_vel_right)
        return loss


def root_pos_velocity_boundary_loss(
    gt_pos: torch.Tensor,
    pred_pos: torch.Tensor,
    hole_start: int,
    hole_end: int
) -> torch.Tensor:
    """
    gt_pos, pred_pos: (B,T,3)
    """
    # Left boundary velocity
    gt_vel_left = gt_pos[:, hole_start, :] - gt_pos[:, hole_start - 1, :]         # (B,3)
    pred_vel_left = pred_pos[:, hole_start, :] - pred_pos[:, hole_start - 1, :]   # (B,3)

    # Right boundary velocity
    gt_vel_right = gt_pos[:, hole_end, :] - gt_pos[:, hole_end - 1, :]        # (B,3)
    pred_vel_right = pred_pos[:, hole_end, :] - pred_pos[:, hole_end - 1, :]  # (B,3)

    loss = F.l1_loss(pred_vel_left, gt_vel_left) + F.l1_loss(pred_vel_right, gt_vel_right)
    return loss
#root_pos_velocity_boundary_loss