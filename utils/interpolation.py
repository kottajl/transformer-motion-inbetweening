import torch

from utils.rotation_convertion import quat_to_6d_torch


EPS = 1e-8

def slerp_torch(qa: torch.Tensor, qb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.
    qa, qb: [B, 4] tensors
    t: [B, 1] tensor with interpolation factor in [0, 1]
    Returns: [B, 4] tensor
    """
    # https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3

    # Normalize quaternions
    qa_norm = qa / (torch.norm(qa, dim=-1, keepdim=True) + EPS)
    qb_norm = qb / (torch.norm(qb, dim=-1, keepdim=True) + EPS)

    # Compute the dot product between normalized quaternions
    dot = torch.sum(qa_norm * qb_norm, dim=-1, keepdim=True)

    # If the dot product is negative, negate one quaternion to take the shorter path
    qb_norm = torch.where(dot < 0.0, -qb_norm, qb_norm)
    dot = torch.sum(qa_norm * qb_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # Compute interpolation coefficients
    s0 = torch.sin((1.0 - t) * omega) / (sin_omega + EPS)
    s1 = torch.sin(t * omega) / (sin_omega + EPS)

    # Compute interpolated quaternion
    out = s0 * qa_norm + s1 * qb_norm

    # If sin(omega) is very small (quaternions are very close), use linear interpolation
    small_mask = (sin_omega.abs() < 1e-4).squeeze(-1)  # (N,)
    if small_mask.any():
        lin = (1.0 - t) * qa_norm + t * qb_norm
        mask = small_mask.unsqueeze(-1)  # (N,1)
        out = torch.where(mask, lin, out)

    # Normalize output quaternion
    out = out / (torch.norm(out, dim=-1, keepdim=True) + EPS)
    return out



def interpolate_rotations(
    rot_6d: torch.Tensor,
    rot_quat: torch.Tensor,
    hole_start: int,
    hole_end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert 0 < hole_start < hole_end < rot_6d.shape[1], "Invalid hole_start and hole_end"

    B, T, J, _ = rot_quat.shape
    device = rot_quat.device
    H = hole_end - hole_start       # H = hole length

    qa = rot_quat[:, hole_start - 1, :, :]  # [B, J, 4]
    qb = rot_quat[:, hole_end, :, :]        # [B, J, 4]

    t_values = torch.linspace(0.0, 1.0, steps=H+2, device=device)[1:-1]  # [H]

    qa = qa.unsqueeze(1).expand(B, H, J, 4).reshape(-1, 4)  # [B*H*J, 4]
    qb = qb.unsqueeze(1).expand(B, H, J, 4).reshape(-1, 4)  # [B*H*J, 4]
    t_values = t_values.view(1, H, 1).expand(B, H, J).reshape(-1, 1)  # [B*H*J, 1]

    q_interp = slerp_torch(qa, qb, t_values)  # [B*H*J, 4]
    q_interp = q_interp.reshape(B, H, J, 4)   # [B, H, J, 4]

    rot_quat_out = rot_quat.clone()
    rot_quat_out[:, hole_start:hole_end, :, :] = q_interp

    rot_6d_out = rot_6d.clone()
    rot_6d_out[:, hole_start:hole_end, :, :] = quat_to_6d_torch(q_interp)

    return rot_6d_out, rot_quat_out


def interpolate_positions(
    positions: torch.Tensor,
    hole_start: int,
    hole_end: int
) -> torch.Tensor:
    assert 0 < hole_start < hole_end < positions.shape[1], "Invalid hole_start and hole_end"

    B, T, _ = positions.shape
    device = positions.device
    H = hole_end - hole_start       # H = hole length

    pa = positions[:, hole_start - 1, :]    # [B, 3]
    pb = positions[:, hole_end, :]          # [B, 3]

    t_values = torch.linspace(0.0, 1.0, steps=H+2, device=device)[1:-1]  # [H]

    pa = pa.unsqueeze(1).expand(B, H, 3).reshape(-1, 3)         # [B*H, 3]
    pb = pb.unsqueeze(1).expand(B, H, 3).reshape(-1, 3)         # [B*H, 3]
    t_values = t_values.view(1, H).expand(B, H).reshape(-1, 1)  # [B*H, 1]

    p_interp = (1.0 - t_values) * pa + t_values * pb            # [B*H, 3]
    p_interp = p_interp.reshape(B, H, 3)                        # [B, H, 3]

    positions_out = positions.clone()
    positions_out[:, hole_start:hole_end, :] = p_interp

    return positions_out
