from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rot6d_to_mat_torch, forward_kinematics


class FKLoss(nn.Module):
    def __init__(self, parents):
        super(FKLoss, self).__init__()
        self.parents = parents

    # def _forward_kinematics(
    #     self,
    #     rot_mats: torch.Tensor,
    #     root_pos: torch.Tensor,
    #     offsets: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     rot_mats: (B,T,J,3,3)
    #     root_pos: (B,T,3)
    #     offsets: (J,3)

    #     Returns:
    #         joint_positions: (B,T,J,3) - joint world positions
    #     """
    #     B, T, J, _, _ = rot_mats.shape
    #     device = rot_mats.device

    #     joint_positions = torch.zeros((B, T, J, 3), device=device)
    #     joint_positions[:, :, 0, :] = root_pos

    #     for j in range(1, J):
    #         parent_idx = self.parents[j]
    #         parent_pos = joint_positions[:, :, parent_idx, :]   # (B,T,3)
    #         parent_rot = rot_mats[:, :, parent_idx, :, :]       # (B,T,3,3)
            
    #         # Local offset of this joint
    #         offset = offsets[j].to(device)                      # (3,)
    #         rotated_offset = torch.matmul(parent_rot, offset.view(3, 1)).squeeze(-1)  # (B,T,3)
    #         world_pos = parent_pos + rotated_offset    # (B,T,3)
            
    #         joint_positions[:, :, j, :] = world_pos

    #     return joint_positions
    # #_forward_kinematics
    
    def forward(
        self,
        gt_rot6d: torch.Tensor,
        gt_pos: torch.Tensor,
        pred_rot6d: torch.Tensor,
        pred_pos: torch.Tensor,
        offsets: torch.Tensor
    ):
        """
        gt_rot6d, pred_rot6d: (B,T,J,6)
        gt_pos, pred_pos: (B,T,3)
        offsets: (J,3)

        Returns:
            fk_loss: L1 scalar
        """

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

        fk_loss = F.l1_loss(pred_joint_pos, gt_joint_pos)
        return fk_loss
    #forward

#FKLoss