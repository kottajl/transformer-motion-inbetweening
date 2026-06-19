import torch
import torch.nn as nn


class RelativeAttentionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        max_dist: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist

        self.embeddings = nn.Embedding(2 * max_dist + 1, num_heads)

    def forward(self, T: int, B: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(T, device=device)
        # Compute matrix with relative positions (distances) between each pair of positions
        distances = positions.unsqueeze(1) - positions.unsqueeze(0) # (T, T)

        # Shift distances to be non-negative (for embedding lookup)
        distances = torch.clamp(distances, -self.max_dist, self.max_dist)
        indices = distances + self.max_dist         # (T, T)

        # Get weights for each head based on relative distances
        bias = self.embeddings(indices)             # (T, T, num_heads)

        # Reshape to (num_heads, T, T) for multi-head attention
        bias = bias.permute(2, 0, 1)                # (num_heads, T, T)
        bias = bias.unsqueeze(0).repeat(B, 1, 1, 1) # (B, num_heads, T, T)
        bias = bias.view(B * self.num_heads, T, T)  # (B*num_heads, T, T)

        return bias
#RelativeAttentionBias