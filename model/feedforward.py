"""
Position-wise Feed-Forward Network
====================================
Section 3.3 of "Attention Is All You Need".

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

The dimensionality of input and output is d_model = 512, and the inner-layer
has dimensionality d_ff = 2048 in the base model.
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network applied to each position independently.

    Args:
        d_model (int): Model dimension (512 in the base model).
        d_ff    (int): Inner hidden dimension (2048 in the base model).
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # FFN(x) = max(0, x W1 + b1) W2 + b2
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
