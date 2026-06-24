from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm

import model.positional_encoding.rotary as rotary


class RotaryMultiheadAttention(nn.Module):
    r"""
    RotaryMultiheadAttention is basically the MultiheadAttention, but with RoPE rotations of Key and Querry matrices.
    Built with help of The Illustrated Transformer tutorial by Jay Alammar (https://jalammar.github.io/illustrated-transformer/).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model == self.d_head * num_heads, "d_model must be divisible by num_heads"
        self.dropout_p = dropout
        self.batch_first = batch_first

        # W^Q, W^K and W^V matrices (Querry, Key, Value)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # W^O matrix (Output)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Precompute cosinuses and sinuses
        angles_cos, angles_sin = rotary.precompute_angles(self.d_head, max_seq_len)
        self.register_buffer("all_angles_cos", angles_cos)
        self.register_buffer("all_angles_sin", angles_sin)
    #__init__

    # x = self.self_attn(
    #         x,
    #         x,
    #         x,
    #         attn_mask=attn_mask,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=False,
    #         is_causal=is_causal,
    #     )[0]

    def forward(
        self,
        # x: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,

        key_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        is_causal: bool = False
    ):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        B, T, _ = query.shape

        # Create real Querry, Key and Value vectors (-> [B, T, d_model])
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Separate the heads (new dimension, -> [B, T, num_heads, d_head])
        q = q.view(B, T, self.num_heads, self.d_head)
        k = k.view(B, T, self.num_heads, self.d_head)
        v = v.view(B, T, self.num_heads, self.d_head)

        # Apply RoPE embedding
        angles_cos = self.all_angles_cos[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_head]
        angles_sin = self.all_angles_sin[:T].unsqueeze(0).unsqueeze(2)  # -||-
        q = rotary.apply_rotary_emb(q, angles_cos=angles_cos, angles_sin=angles_sin)
        k = rotary.apply_rotary_emb(k, angles_cos=angles_cos, angles_sin=angles_sin)

        # Transpose before the attention computing (-> [B, num_heads, T, d_head])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate self-attention scores
        attn_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        # Transpose back and connect the heads (-> [B, T, d_model])
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)

        # Output projection
        out = self.out_proj(attn_output)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, None
    #forward

#RotaryMultiheadAttention

class RotaryTransformerEncoderLayer(nn.Module):
    r"""
    RotaryTransformerEncoderLayer is directly based on nn.TransofrmerEncoderLayer from PyTorch. 
    The only difference is usage of custom RotaryMultiheadAttention instead of the original MultiheadAttention.

    Original code: https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/transformer.py
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_seq_len: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(
            d_model=d_model,
            num_heads=nhead,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        # pyrefly: ignore [bad-argument-type]
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # pyrefly: ignore [bad-argument-type]
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # -- cutted fastpath section --

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
#RotaryTransofrmerEncoderLayer


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
#_get_activation_fn