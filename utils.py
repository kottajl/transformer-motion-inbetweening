from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import numpy as np
import torch
import json


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_params_from_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        params = json.load(f)
        
    if "description" in params:
        del params["description"]
    return params


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
