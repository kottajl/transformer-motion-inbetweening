import torch

def l2p(pred_pos, gt_pos):
    l2_dist = torch.norm(gt_pos - pred_pos, dim=-1)
    return l2_dist.mean().item()


def l2q(pred_quat, gt_quat):
    l2_dist1 = torch.norm(gt_quat - pred_quat, dim=-1)
    l2_dist2 = torch.norm(gt_quat + pred_quat, dim=-1)
    l2_dist = torch.minimum(l2_dist1, l2_dist2)
    return l2_dist.mean().item()


def npss(pred_seq, gt_seq):
    '''
    Normalized Power Spectrum Similarity.
    pred_seq, gt_seq: [B, T, D]
    '''
    EPS = 1e-8

    # FFT along the time dimension
    gt_fft = torch.fft.rfft(gt_seq, dim=1)
    pred_fft = torch.fft.rfft(pred_seq, dim=1)
    
    # Power spectrum
    gt_power = torch.abs(gt_fft) ** 2
    pred_power = torch.abs(pred_fft) ** 2

    # Normalize power spectrum
    gt_power_norm = gt_power / (torch.sum(gt_power, dim=1, keepdim=True) + EPS)
    pred_power_norm = pred_power / (torch.sum(pred_power, dim=1, keepdim=True) + EPS)

    emd = torch.sum(torch.abs(gt_power_norm - pred_power_norm), dim=1)
    p = torch.sum(gt_power, dim=1)

    # Compute NPSs score
    npss_score = torch.sum(p * emd, dim=1) / (torch.sum(p, dim=1) + EPS)
    return npss_score.mean().item()
#npss