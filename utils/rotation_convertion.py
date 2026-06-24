from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import numpy as np
import torch


'''
Scipy Rotation utility functions (CPU)
'''

def euler_to_6d(eulers, order: str, degrees: bool = True):
    """
    eulers: [T, 3]
    order: str, e.g. 'XYZ'
    degrees: bool

    returns: [T, 6]
    """
    assert eulers.ndim > 1, "'eulers' does not have a frames dimension"
    n_frames = eulers.shape[0]      # = T

    # Get rotation matrices 3x3 for every frame
    rot_matrices = R.from_euler(
        order, 
        eulers, 
        degrees=degrees
    ).as_matrix()   # -> [T, 3, 3]

    col1 = rot_matrices[:, :, 0]  # (T,3)
    col2 = rot_matrices[:, :, 1]  # (T,3)
    sixd = np.concatenate([col1, col2], axis=1)
    return sixd


def euler_to_quat(euler_angles, order, degrees: bool = True):
    """
    euler_angles: [T, 3]
    order: str, e.g. 'XYZ'
    degrees: bool

    returns: [T, 4] (x, y, z, w)
    """
    return R.from_euler(
        order, 
        euler_angles, 
        degrees=degrees
    ).as_quat()



'''
PyTorch rotation utility functions (GPU)

Some of functions below are heavily inspired by code from PyTorch3D.
https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
'''

def quat_to_mat_torch(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) x,y,z,w
    returns: (...,3,3)
    """
    x = q[..., 0]; y = q[..., 1]; z = q[..., 2]; w = q[..., 3]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    mat = torch.stack([row0, row1, row2], dim=-2)  # (...,3,3)
    return mat


def quat_to_6d_torch(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) x,y,z,w
    returns: (...,6)
    """
    mat = quat_to_mat_torch(q)
    col1 = mat[..., :, 0]
    col2 = mat[..., :, 1]
    sixd = torch.cat([col1, col2], dim=-1)
    return sixd


def rot6d_to_mat_torch(rot_6d: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    rot_6d: (...,6)
    returns: (...,3,3)
    """
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=eps)

    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1, eps=eps)

    b3 = torch.cross(b1, b2, dim=-1)

    mats = torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)
    return mats


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    positive_mask = x > 0
    safe_x = torch.where(positive_mask, x, 1.0)
    return torch.where(positive_mask, torch.sqrt(safe_x), 0.0)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def mat_to_quat_torch(mat: torch.Tensor) -> torch.Tensor:
    """
    mat: (...,3,3)
    Returns: (...,4) x,y,z,w
    """

    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {mat.shape}.")

    batch_dim = mat.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        mat.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [torch.square(q_abs[..., 0]), m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, torch.square(q_abs[..., 1]), m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, torch.square(q_abs[..., 2]), m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, torch.square(q_abs[..., 3])], dim=-1
            ),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    out = standardize_quaternion(out)
    
    # wxyz -> xyzw
    out = torch.cat([out[..., 1:], out[..., :1]], dim=-1)
    return out
