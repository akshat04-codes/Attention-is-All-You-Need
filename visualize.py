"""
Visualization Utilities
========================
Generates plots for the report:
    1. Training/Validation loss curves
    2. Learning rate schedule
    3. Attention heatmaps
    4. Ablation comparison bar charts
    5. Positional encoding visualization
"""

import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for Colab/server
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions disabled.")


def plot_loss_curves(results_path: str, save_dir: str = "results/plots"):
    """Plot training and validation loss curves."""
    if not HAS_MATPLOTLIB:
        return

    with open(results_path) as f:
        results = json.load(f)

    history = results["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training & Validation Loss — {results['run_name']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, history["learning_rates"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Noam Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"{results['run_name']}_loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {save_path}")


def plot_noam_schedule(
    d_model: int = 128,
    warmup_steps: int = 4000,
    total_steps: int = 20000,
    save_dir: str = "results/plots",
):
    """Plot the Noam LR schedule showing warmup and decay phases."""
    if not HAS_MATPLOTLIB:
        return

    steps = np.arange(1, total_steps + 1)
    rates = [
        d_model ** (-0.5) * min(s ** (-0.5), s * warmup_steps ** (-1.5))
        for s in steps
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rates, "b-", linewidth=2)
    ax.axvline(x=warmup_steps, color="r", linestyle="--", alpha=0.7, label=f"warmup={warmup_steps}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Noam LR Schedule (d_model={d_model}, warmup={warmup_steps})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "noam_schedule.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Noam schedule to {save_path}")


def plot_positional_encoding(
    d_model: int = 128,
    max_len: int = 100,
    save_dir: str = "results/plots",
):
    """Visualize the sinusoidal positional encoding matrix."""
    if not HAS_MATPLOTLIB:
        return

    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Position")
    ax.set_title("Sinusoidal Positional Encoding (PE)")
    plt.colorbar(im, ax=ax)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "positional_encoding.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved positional encoding to {save_path}")


def plot_ablation_comparison(
    summary_path: str = "results/ablation_summary.json",
    save_dir: str = "results/plots",
):
    """Bar chart comparing ablation variants."""
    if not HAS_MATPLOTLIB:
        return

    with open(summary_path) as f:
        summary = json.load(f)

    names = list(summary.keys())
    val_losses = [summary[n]["best_val_loss"] for n in names]
    params = [summary[n]["n_parameters"] / 1000 for n in names]  # in thousands

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Validation loss comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars1 = ax1.bar(range(len(names)), val_losses, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Best Validation Loss")
    ax1.set_title("Ablation Study — Validation Loss Comparison")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, val_losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Parameter count comparison
    bars2 = ax2.bar(range(len(names)), params, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Parameters (thousands)")
    ax2.set_title("Ablation Study — Model Size Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.0f}K", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "ablation_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation comparison to {save_path}")


def plot_attention_heatmap(
    attention_weights,
    src_tokens: list[str],
    tgt_tokens: list[str],
    head: int = 0,
    layer: int = 0,
    save_path: str = "results/plots/attention_heatmap.png",
):
    """Plot attention weights as a heatmap for a specific head and layer."""
    if not HAS_MATPLOTLIB:
        return

    if hasattr(attention_weights, "detach"):
        attention_weights = attention_weights.detach()
    if hasattr(attention_weights, "cpu"):
        attention_weights = attention_weights.cpu().numpy()

    if attention_weights.ndim == 4:
        attn = attention_weights[0, head]
    elif attention_weights.ndim == 3:
        attn = attention_weights[head]
    else:
        attn = attention_weights

    # Slice to match tokens length if necessary
    tgt_len = min(len(tgt_tokens), attn.shape[0])
    src_len = min(len(src_tokens), attn.shape[1])
    attn = attn[:tgt_len, :src_len]
    src_tokens = src_tokens[:src_len]
    tgt_tokens = tgt_tokens[:tgt_len]

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.6), max(5, len(tgt_tokens) * 0.5)))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.set_title(f"Attention Weights — Layer {layer}, Head {head}")
    plt.colorbar(im, ax=ax)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmap to {save_path}")


if __name__ == "__main__":
    # Generate standalone plots
    plot_noam_schedule()
    plot_positional_encoding()
    print("Generated standalone visualizations.")
