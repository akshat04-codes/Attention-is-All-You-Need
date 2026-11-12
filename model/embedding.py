"""
Embeddings and Positional Encoding
====================================
Section 3.4 and Section 3.5 of "Attention Is All You Need".

Token Embedding:
    Standard learned embedding scaled by sqrt(d_model).

Positional Encoding (sinusoidal):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learned token embedding scaled by sqrt(d_model) as described in Section 3.4.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model    (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — token indices
        Returns:
            (batch, seq_len, d_model)
        """
        # The paper multiplies embedding weights by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from Section 3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model  (int): Model dimension.
        max_len  (int): Maximum sequence length.
        dropout  (float): Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Combined token + positional embedding.

    Args:
        vocab_size (int): Vocabulary size.
        d_model    (int): Model dimension.
        max_len    (int): Maximum sequence length.
        dropout    (float): Dropout rate.
        use_positional_encoding (bool): Whether to add sinusoidal positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len)
        Returns:
            (batch, seq_len, d_model)
        """
        tok_emb = self.token_embedding(x)
        if self.use_positional_encoding:
            return self.positional_encoding(tok_emb)
        return self.dropout(tok_emb)
