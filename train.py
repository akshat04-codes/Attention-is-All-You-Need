"""
Training Script
================
Main training loop for the Transformer model on Multi30k DE→EN translation.

Implements:
    - Label smoothing (Section 5.4)
    - Noam learning rate schedule (Section 5.3)
    - Adam optimizer with β1=0.9, β2=0.98, ε=10^-9
    - Gradient clipping
    - Checkpoint saving
    - Training/validation loss logging
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from model.transformer import Transformer
from config import ModelConfig, TrainConfig, AblationConfig, ABLATION_VARIANTS
from data import load_multi30k, Vocabulary
from scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss (Section 5.4 of the paper).

    Instead of one-hot targets, distributes ε_ls probability mass uniformly
    over all non-target classes. This hurts perplexity but improves accuracy
    and BLEU score.

    Args:
        smoothing (float): Label smoothing factor ε_ls (0.1 in the paper).
        vocab_size (int): Size of the target vocabulary.
        pad_idx    (int): Padding index to ignore.
    """

    def __init__(self, smoothing: float, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size) — model predictions (log-probs)
            target: (batch * seq_len,)             — target token indices
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.pad_idx] = 0

        # Mask out padding positions
        mask = target != self.pad_idx
        smooth_target = smooth_target * mask.unsqueeze(1)

        loss = self.criterion(log_probs, smooth_target)
        return loss


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: Transformer,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = model(src, tgt_input)  # (batch, tgt_len-1, vocab_size)

        # Reshape for loss computation
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        tgt_flat = tgt_output.contiguous().view(-1)

        # Compute loss
        loss = criterion(logits_flat, tgt_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            avg = total_loss / n_batches
            lr = scheduler.get_last_lr()
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg:.4f} | LR: {lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: Transformer,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        tgt_flat = tgt_output.contiguous().view(-1)

        loss = criterion(logits_flat, tgt_flat)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    ablation: AblationConfig | None = None,
    run_name: str = "base",
) -> dict:
    """
    Full training pipeline.

    Args:
        model_config: Override model config.
        train_config: Override training config.
        ablation:     Ablation variant to apply.
        run_name:     Name for this run (used in logging/checkpoints).

    Returns:
        Dictionary of training results (losses, best metrics).
    """
    if train_config is None:
        train_config = TrainConfig()
    if model_config is None:
        model_config = ModelConfig()

    # Apply ablation overrides
    if ablation is not None:
        if ablation.d_model is not None:
            model_config.d_model = ablation.d_model
        if ablation.n_heads is not None:
            model_config.n_heads = ablation.n_heads
        if ablation.d_ff is not None:
            model_config.d_ff = ablation.d_ff
        if ablation.n_layers is not None:
            model_config.n_layers = ablation.n_layers
        if ablation.dropout is not None:
            model_config.dropout = ablation.dropout
        if hasattr(ablation, "use_positional_encoding") and ablation.use_positional_encoding is not None:
            model_config.use_positional_encoding = ablation.use_positional_encoding

    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Task: Multi30k {train_config.src_lang.upper()} -> {train_config.tgt_lang.upper()}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_multi30k(
        batch_size=train_config.batch_size,
        max_len=train_config.max_seq_len,
        src_lang=train_config.src_lang,
        tgt_lang=train_config.tgt_lang,
    )

    # Build model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        n_layers=model_config.n_layers,
        max_len=model_config.max_len,
        dropout=model_config.dropout,
        pad_idx=Vocabulary.PAD_IDX,
        use_positional_encoding=model_config.use_positional_encoding,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    print(f"Config: d_model={model_config.d_model}, n_heads={model_config.n_heads}, "
          f"d_ff={model_config.d_ff}, n_layers={model_config.n_layers}, "
          f"dropout={model_config.dropout}")

    # Loss function
    label_smoothing = train_config.label_smoothing
    if ablation and ablation.name == "no_label_smoothing":
        label_smoothing = 0.0

    criterion = LabelSmoothingLoss(
        smoothing=label_smoothing,
        vocab_size=len(tgt_vocab),
        pad_idx=Vocabulary.PAD_IDX,
    )

    # Optimizer (Section 5.3: Adam with β1=0.9, β2=0.98, ε=10^-9)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,  # LR is set by the scheduler
        betas=train_config.betas,
        eps=train_config.eps,
    )

    # Noam LR scheduler
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=model_config.d_model,
        warmup_steps=train_config.warmup_steps,
    )

    # Create checkpoint directory
    ckpt_dir = Path(train_config.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(train_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "epoch_times": [],
    }
    best_val_loss = float("inf")

    print(f"\nStarting training for {train_config.epochs} epochs...")
    total_start = time.time()

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, train_config, epoch,
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(scheduler.get_last_lr())
        history["epoch_times"].append(epoch_time)

        print(
            f"Epoch {epoch}/{train_config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr():.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_config": vars(model_config),
            }, ckpt_dir / "best_model.pt")
            print(f"  [+] Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % train_config.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_dir / f"checkpoint_epoch{epoch}.pt")

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history
    results = {
        "run_name": run_name,
        "model_config": vars(model_config),
        "train_config": {
            k: v for k, v in vars(train_config).items()
            if not callable(v)
        },
        "n_parameters": n_params,
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": total_time,
        "history": history,
    }
    if ablation:
        results["ablation"] = {
            "name": ablation.name,
            "description": ablation.description,
        }

    results_path = results_dir / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Transformer from scratch")
    parser.add_argument("--run-name", type=str, default="base", help="Name for this run")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--d-ff", type=int, default=512, help="FF hidden dim")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup", type=int, default=4000, help="Warmup steps")
    parser.add_argument("--src-lang", type=str, default="en", help="Source language (en or de)")
    parser.add_argument("--tgt-lang", type=str, default="de", help="Target language (de or en)")
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=list(ABLATION_VARIANTS.keys()),
        help="Run a predefined ablation variant"
    )
    args = parser.parse_args()

    model_config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_steps=args.warmup,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    ablation = ABLATION_VARIANTS.get(args.ablation) if args.ablation else None
    run_name = args.ablation if args.ablation else args.run_name

    train(
        model_config=model_config,
        train_config=train_config,
        ablation=ablation,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
