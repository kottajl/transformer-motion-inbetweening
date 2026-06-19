import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    # https://medium.com/@lixue421/understanding-positional-encoding-in-transformers-2c7336728be5
    def __init__(
        self,
        dim: int,
        max_len: int = 5000
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape (max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if self.dim != x.size(-1):
            raise ValueError(f'Positional encoding dim ({self.dim}) != input last dim ({x.size(-1)})')

        pe = self.pe[:T].unsqueeze(0).to(x.dtype).to(x.device)  # (1, T, D)
        return x + pe
#SinusoidalPositionalEncoding