"""
Transformer Decoder
====================
Section 3.1 of "Attention Is All You Need".

The decoder is also composed of a stack of N = 6 identical layers.
Each layer has three sub-layers:
    1. Masked Multi-Head Self-Attention  (causal mask prevents attending to future)
    2. Multi-Head Cross-Attention        (attends to encoder output)
    3. Position-wise Feed-Forward Network
Each sub-layer employs a residual connection followed by Layer Normalization.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single decoder layer (Figure 1, right side of the paper).

    Sub-layers:
        1. Masked Multi-Head Self-Attention + Add & Norm
        2. Multi-Head Cross-Attention       + Add & Norm
        3. Position-wise FFN                + Add & Norm

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

        # Sub-layer 1: Masked Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Sub-layer 2: Cross-Attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Sub-layer 3: Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              (batch, tgt_seq_len, d_model)
            encoder_output: (batch, src_seq_len, d_model)
            tgt_mask:       (batch, 1, tgt_seq_len, tgt_seq_len) — causal + padding
            src_mask:       (batch, 1, 1, src_seq_len) — padding mask for encoder

        Returns:
            output:           (batch, tgt_seq_len, d_model)
            self_attn_weights: (batch, n_heads, tgt_seq_len, tgt_seq_len)
            cross_attn_weights:(batch, n_heads, tgt_seq_len, src_seq_len)
        """
        # Sub-layer 1: Masked Self-Attention + Residual + LayerNorm
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x, mask=tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))

        # Sub-layer 2: Cross-Attention + Residual + LayerNorm
        # Query from decoder, Key & Value from encoder output
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Sub-layer 3: Feed-Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    Transformer Decoder — stack of N identical DecoderLayers.

    Args:
        d_model  (int): Model dimension.
        n_heads  (int): Number of attention heads.
        d_ff     (int): Feed-forward hidden dimension.
        n_layers (int): Number of decoder layers (N=6 in the paper).
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
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ):
        """
        Args:
            x:              (batch, tgt_seq_len, d_model)
            encoder_output: (batch, src_seq_len, d_model)
            tgt_mask:       (batch, 1, tgt_seq_len, tgt_seq_len)
            src_mask:       (batch, 1, 1, src_seq_len)
            return_attentions: whether to return attention weights

        Returns:
            (batch, tgt_seq_len, d_model)
            or (output, self_attentions, cross_attentions) if return_attentions=True
        """
        self_attentions = []
        cross_attentions = []
        for layer in self.layers:
            x, s_attn, c_attn = layer(x, encoder_output, tgt_mask, src_mask)
            if return_attentions:
                self_attentions.append(s_attn)
                cross_attentions.append(c_attn)
        if return_attentions:
            return x, self_attentions, cross_attentions
        return x
