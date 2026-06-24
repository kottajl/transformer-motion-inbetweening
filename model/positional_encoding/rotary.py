import torch

"""
Based on the tutorials:
https://medium.com/@ngiengkianyew/understanding-rotary-positional-encoding-40635a4d078e
https://krasserm.github.io/2022/12/13/rotary-position-embedding/
"""


def precompute_angles(
    dim: int,
    max_seq_len: int = 256,
    big_theta: float = 10000.0
):
    # Thetas values are generated just like the original paper suggests
    theta = 1.0 / (big_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # -> [theta_0, theta_1, ..., theta_{dim//2}]

    # Positions [of frames]
    m = torch.arange(max_seq_len)
    # -> [0, 1, ..., max_seq_len]

    # Compute angles for every pair
    angles = torch.outer(m, theta)      # (max_seq_len, dim//2)

    # Compute cosinuses and sinuses (and expand the pairs)
    angles_cos = torch.cos(angles).repeat_interleave(2, dim=-1) # (max_seq_len, dim)
    angles_sin = torch.sin(angles).repeat_interleave(2, dim=-1) # (max_seq_len, dim)

    return angles_cos, angles_sin
#precompute_angles


def apply_rotary_emb(
    x: torch.Tensor,
    angles_cos: torch.Tensor,
    angles_sin: torch.Tensor
) -> torch.Tensor:
    '''
    
    
    For pair (u1, u2) we want to compute:
      new_u1 = u1 * cos(...) - u2 * sin(...)
      new_u2 = u2 * cos(...) + u1 * sin(...)
    '''

    """
    x = [...][u1, u2, u3, u4, ...]
      => x_rot = [...][-u2, u1, -u4, u3, ...]
    """
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)

    return (x * angles_cos) + (x_rot * angles_sin)
#apply_rotary_emb