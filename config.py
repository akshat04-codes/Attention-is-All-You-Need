"""
Configuration
=============
All hyperparameters and paths for the Transformer training pipeline.
Base model (Table 3): d_model=512, d_ff=2048, h=8, N=6, dropout=0.1
Reduced model (for free Colab T4 / laptop CPU): scaled down for feasibility.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""
    d_model: int = 128          # Paper: 512 (reduced for compute budget)
    n_heads: int = 4            # Paper: 8
    d_ff: int = 512             # Paper: 2048
    n_layers: int = 3           # Paper: 6
    max_len: int = 256          # Paper: no explicit limit
    dropout: float = 0.1       # Paper: 0.1
    use_positional_encoding: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # --- Data ---
    dataset: str = "multi30k"   # Multi30k German→English (small, manageable)
    src_lang: str = "de"
    tgt_lang: str = "en"
    max_seq_len: int = 128

    # --- Optimiser (Section 5.3 of the paper) ---
    # Adam with β1=0.9, β2=0.98, ε=10^-9
    lr: float = 1e-4
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    warmup_steps: int = 4000    # Paper: 4000

    # --- Training ---
    batch_size: int = 64        # Paper: ~25000 tokens/batch
    epochs: int = 30
    label_smoothing: float = 0.1  # Paper: ε_ls = 0.1
    grad_clip: float = 1.0
    accumulation_steps: int = 1

    # --- Reproducibility ---
    seed: int = 42

    # --- Logging ---
    log_every: int = 50         # Log every N batches
    save_every: int = 5         # Save checkpoint every N epochs
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


@dataclass
class AblationConfig:
    """
    Configuration for ablation studies (Table 3 in the paper).
    Each variant changes one aspect of the base model.
    """
    name: str = "base"
    description: str = ""

    # Override any ModelConfig field
    d_model: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None
    n_layers: int | None = None
    dropout: float | None = None

    # Additional ablation controls
    use_positional_encoding: bool = True    # Ablate positional encoding
    attention_type: str = "dot_product"     # "dot_product" or "additive"


# ──────────────────────────────────────────────────────────
# Pre-defined ablation variants matching Table 3 of the paper
# (scaled down to fit our reduced model)
# ──────────────────────────────────────────────────────────
ABLATION_VARIANTS: dict[str, AblationConfig] = {
    "base": AblationConfig(
        name="base",
        description="Reduced-scale base model (d=128, h=4, N=3)",
    ),
    "fewer_heads": AblationConfig(
        name="fewer_heads",
        description="Single attention head (h=1) — Table 3 row (B)",
        n_heads=1,
    ),
    "more_heads": AblationConfig(
        name="more_heads",
        description="More attention heads (h=8) — Table 3 row (B)",
        n_heads=8,
    ),
    "smaller_d_model": AblationConfig(
        name="smaller_d_model",
        description="Smaller model dimension (d=64) — exploring capacity",
        d_model=64,
        d_ff=256,
        n_heads=4,
    ),
    "deeper_model": AblationConfig(
        name="deeper_model",
        description="Deeper model (N=6 layers) — Table 3 row (C)",
        n_layers=6,
    ),
    "no_positional_encoding": AblationConfig(
        name="no_positional_encoding",
        description="No positional encoding — Table 3 row (E)",
        use_positional_encoding=False,
    ),
    "higher_dropout": AblationConfig(
        name="higher_dropout",
        description="Higher dropout (0.3) — regularization ablation",
        dropout=0.3,
    ),
    "no_label_smoothing": AblationConfig(
        name="no_label_smoothing",
        description="No label smoothing — Table 3 comparison",
    ),
}
