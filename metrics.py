import torch

def l2p(pred_pos, gt_pos):
    l2_dist = torch.norm(gt_pos - pred_pos, dim=-1)
    return l2_dist.mean().item()


def l2q(pred_quat, gt_quat):
    l2_dist1 = torch.norm(gt_quat - pred_quat, dim=-1)
    l2_dist2 = torch.norm(gt_quat + pred_quat, dim=-1)
    l2_dist = torch.minimum(l2_dist1, l2_dist2)
    return l2_dist.mean().item()