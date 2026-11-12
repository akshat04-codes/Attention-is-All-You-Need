"""
Transformer Encoder
====================
Section 3.1 of "Attention Is All You Need".

The encoder is composed of a stack of N = 6 identical layers.
Each layer has two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
Each sub-layer employs a residual connection followed by Layer Normalization:
    LayerNorm(x + Sublayer(x))
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single encoder layer (Figure 1, left side of the paper).

    Sub-layers:
        1. Multi-Head Self-Attention  + Add & Norm
        2. Position-wise FFN          + Add & Norm

    Args:
        d_model  (int): Model dimension.
        n_heads  (int): Number of attention heads.
        d_ff     (int): Feed-forward hidden dimension.
        dropout  (float): Dropout rate.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Sub-layer 2: Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (batch, seq_len, d_model) — encoder input
            src_mask: (batch, 1, 1, seq_len)    — padding mask

        Returns:
            output:      (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Self-Attention + Residual + LayerNorm
        attn_output, attn_weights = self.self_attention(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class Encoder(nn.Module):
    """
    Transformer Encoder — stack of N identical EncoderLayers.

    Args:
        d_model  (int): Model dimension.
        n_heads  (int): Number of attention heads.
        d_ff     (int): Feed-forward hidden dimension.
        n_layers (int): Number of encoder layers (N=6 in the paper).
        dropout  (float): Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ):
        """
        Args:
            x:        (batch, seq_len, d_model)
            src_mask: (batch, 1, 1, seq_len)
            return_attentions: whether to return all layer attention weights

        Returns:
            (batch, seq_len, d_model) — encoder output
            or (output, attentions) if return_attentions=True
        """
        attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            if return_attentions:
                attentions.append(attn_weights)
        if return_attentions:
            return x, attentions
        return x
