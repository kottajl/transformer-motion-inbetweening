import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation_convertion import rot6d_to_mat_torch
from utils.utils import forward_kinematics


class SmoothnessLoss(nn.Module):
    def __init__(self, parents):
        super(SmoothnessLoss, self).__init__()
        self.parents = parents

    def forward(
        self,
        pred_rot6d: torch.Tensor,
        pred_pos: torch.Tensor,
        offsets: torch.Tensor
    ):
        
        _, T, _ = pred_pos.shape
        # T represents the length of the hole with one frame of context on each side
        # hole_length = T - 2
        # hole_start = 1
        # hole_end = hole_start + hole_length

        pred_rot_mats = rot6d_to_mat_torch(pred_rot6d)  # (B,T,J,3,3)
        pred_joint_pos = forward_kinematics(
            pred_rot_mats,
            pred_pos,
            parents=self.parents,
            offsets=offsets
        )   # (B,T,J,3)

        vel = pred_joint_pos[:, 1:, :, :] - pred_joint_pos[:, :-1, :, :]   # (B,T-1,J,3)
        acc = vel[:, 1:, :, :] - vel[:, :-1, :, :]   # (B,T-2,J,3)

        # loss = torch.mean(torch.abs(vel))
        loss = torch.mean(acc ** 2)
        return loss
    
#SmoothnessLoss

    
