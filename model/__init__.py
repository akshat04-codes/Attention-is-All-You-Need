# Transformer Model - "Attention Is All You Need" (Vaswani et al., 2017)
# Paper: https://arxiv.org/abs/1706.03762
#
# This package contains a from-scratch PyTorch implementation of the
# Transformer architecture for sequence-to-sequence tasks.

from .transformer import Transformer
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .feedforward import PositionwiseFeedForward
from .embedding import TransformerEmbedding, PositionalEncoding, TokenEmbedding

__all__ = [
    "Transformer",
    "Encoder", "EncoderLayer",
    "Decoder", "DecoderLayer",
    "MultiHeadAttention", "ScaledDotProductAttention",
    "PositionwiseFeedForward",
    "TransformerEmbedding", "PositionalEncoding", "TokenEmbedding",
]
