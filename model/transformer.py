"""
Full Transformer Model
=======================
Assembles the complete Transformer architecture from "Attention Is All You Need"
(Vaswani et al., 2017).

Architecture:
    Encoder: N × [Self-Attention → FFN]
    Decoder: N × [Masked Self-Attention → Cross-Attention → FFN]
    Final:   Linear projection → Softmax

Base model hyperparameters (Table 3 in the paper):
    d_model = 512, d_ff = 2048, h = 8, N = 6, P_drop = 0.1
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .embedding import TransformerEmbedding


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        d_model  (int): Model dimension (default: 512).
        n_heads  (int): Number of attention heads (default: 8).
        d_ff     (int): Feed-forward hidden dimension (default: 2048).
        n_layers (int): Number of encoder/decoder layers (default: 6).
        max_len  (int): Maximum sequence length (default: 5000).
        dropout  (float): Dropout rate (default: 0.1).
        pad_idx  (int): Padding token index (default: 0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # Embeddings (shared weight is optional; we keep them separate here)
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_len, dropout, use_positional_encoding=use_positional_encoding
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_len, dropout, use_positional_encoding=use_positional_encoding
        )

        # Encoder and Decoder stacks
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)

        # Final linear layer projecting to target vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters (Xavier uniform as common practice)
        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform initialization for all parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create source padding mask.

        Args:
            src: (batch, src_seq_len) — source token indices

        Returns:
            (batch, 1, 1, src_seq_len) — 1 for valid, 0 for padding
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask  # (batch, 1, 1, src_seq_len)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target mask combining padding mask and causal (look-ahead) mask.

        The causal mask ensures that position i can only attend to positions <= i,
        preventing the decoder from cheating by looking at future tokens.

        Args:
            tgt: (batch, tgt_seq_len)

        Returns:
            (batch, 1, tgt_seq_len, tgt_seq_len) — combined mask
        """
        batch_size, tgt_len = tgt.shape

        # Padding mask: (batch, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask (lower triangular): (1, 1, tgt_len, tgt_len)
        tgt_causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

        # Combine: both conditions must be true
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        return tgt_mask  # (batch, 1, tgt_len, tgt_len)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        return_attentions: bool = False,
    ):
        """
        Forward pass of the Transformer.

        Args:
            src: (batch, src_seq_len) — source token indices
            tgt: (batch, tgt_seq_len) — target token indices (teacher forcing)
            return_attentions: whether to return attention weights dictionary

        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
            or (logits, attention_dict) if return_attentions=True
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        src_emb = self.src_embedding(src)
        if return_attentions:
            encoder_output, enc_attns = self.encoder(
                src_emb, src_mask, return_attentions=True
            )
            tgt_emb = self.tgt_embedding(tgt)
            decoder_output, dec_self_attns, dec_cross_attns = self.decoder(
                tgt_emb, encoder_output, tgt_mask, src_mask, return_attentions=True
            )
            logits = self.output_projection(decoder_output)
            attention_dict = {
                "encoder_self_attentions": enc_attns,
                "decoder_self_attentions": dec_self_attns,
                "decoder_cross_attentions": dec_cross_attns,
            }
            return logits, attention_dict
        else:
            encoder_output = self.encoder(src_emb, src_mask)
            tgt_emb = self.tgt_embedding(tgt)
            decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
            logits = self.output_projection(decoder_output)
            return logits

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sequence (used during inference)."""
        src_mask = self.make_src_mask(src)
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask), src_mask

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target sequence given encoder output (used during inference)."""
        tgt_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)

    def get_attention_maps(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> dict:
        """
        Extract self-attention and cross-attention weight matrices for visualization.

        Args:
            src: (1, src_seq_len)
            tgt: (1, tgt_seq_len)

        Returns:
            dict containing:
                - 'encoder_self_attentions': list of (1, n_heads, src_len, src_len)
                - 'decoder_self_attentions': list of (1, n_heads, tgt_len, tgt_len)
                - 'decoder_cross_attentions': list of (1, n_heads, tgt_len, src_len)
        """
        self.eval()
        with torch.no_grad():
            _, attns = self.forward(src, tgt, return_attentions=True)
        return attns
