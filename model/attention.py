"""
Scaled Dot-Product Attention and Multi-Head Attention
=====================================================
Core attention mechanism from Section 3.2 of "Attention Is All You Need".

Equations implemented:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention (Eq. 1 in the paper).

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        temperature (float): Scaling factor, typically sqrt(d_k).
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:  (batch, n_heads, seq_len_q, d_k)
            key:    (batch, n_heads, seq_len_k, d_k)
            value:  (batch, n_heads, seq_len_v, d_v)   [seq_len_k == seq_len_v]
            mask:   broadcastable boolean/float mask (0 = keep, -inf = mask)

        Returns:
            output:    (batch, n_heads, seq_len_q, d_v)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        # (batch, n_heads, seq_q, d_k) x (batch, n_heads, d_k, seq_k)
        # -> (batch, n_heads, seq_q, seq_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # (batch, n_heads, seq_q, seq_k) x (batch, n_heads, seq_k, d_v)
        # -> (batch, n_heads, seq_q, d_v)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism (Section 3.2.2).

    Instead of performing a single attention function with d_model-dimensional
    keys, values, and queries, we linearly project them h times with different,
    learned linear projections to d_k, d_k, and d_v dimensions respectively.

    Args:
        d_model (int): Model dimension (512 in the base model).
        n_heads (int): Number of parallel attention heads (8 in the base model).
        dropout  (float): Dropout rate applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # d_k = d_v = d_model / h = 64

        # Linear projections: W_i^Q, W_i^K, W_i^V  (packed into single matrices)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection W^O
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask:  (batch, 1, 1, seq_len_k) or (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output:     (batch, seq_len_q, d_model)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1) Linear projections: (batch, seq, d_model) -> (batch, seq, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2) Split into heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(q, k, v, mask=mask)

        # 4) Concatenate heads: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 5) Final linear projection W^O
        output = self.dropout(self.w_o(attn_output))
        return output, attn_weights
