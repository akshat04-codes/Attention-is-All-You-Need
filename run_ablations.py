"""
Ablation Study Runner
======================
Runs multiple ablation experiments as described in Table 3 of the paper,
scaled down to fit the compute budget (free Colab T4 GPU or laptop CPU).

Ablation variants:
    (A) Base model         — d=128, h=4, N=3
    (B) Fewer heads        — h=1
    (B) More heads         — h=8
    (C) Deeper model       — N=6
    (D) Smaller d_model    — d=64
    (E) No pos. encoding   — remove sinusoidal PE
    (F) Higher dropout     — dropout=0.3
    (G) No label smoothing — ε_ls=0.0
"""

import json
import argparse
from pathlib import Path

from config import ModelConfig, TrainConfig, ABLATION_VARIANTS
from train import train


def run_ablations(
    variants: list[str] | None = None,
    epochs: int = 15,
    batch_size: int = 64,
) -> dict:
    """
    Run all specified ablation experiments.

    Args:
        variants: List of ablation variant names. If None, run all.
        epochs:   Number of epochs per variant.
        batch_size: Batch size.

    Returns:
        Combined results dictionary.
    """
    if variants is None:
        variants = list(ABLATION_VARIANTS.keys())

    all_results = {}

    for variant_name in variants:
        ablation = ABLATION_VARIANTS.get(variant_name)
        if ablation is None:
            print(f"Warning: Unknown ablation variant '{variant_name}', skipping.")
            continue

        print(f"\n{'#'*60}")
        print(f"# ABLATION: {variant_name}")
        print(f"# {ablation.description}")
        print(f"{'#'*60}")

        model_config = ModelConfig()
        train_config = TrainConfig(epochs=epochs, batch_size=batch_size)

        results = train(
            model_config=model_config,
            train_config=train_config,
            ablation=ablation,
            run_name=variant_name,
        )
        all_results[variant_name] = results

    # Save combined ablation results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    combined_path = results_dir / "ablation_summary.json"
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "description": ABLATION_VARIANTS[name].description,
            "n_parameters": res["n_parameters"],
            "best_val_loss": res["best_val_loss"],
            "total_time_s": res["total_training_time_seconds"],
        }

    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'ABLATION COMPARISON':^80}")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'Params':>10} {'Best Val Loss':>15} {'Time (s)':>10}")
    print(f"{'-'*80}")
    for name, s in summary.items():
        print(
            f"{name:<25} {s['n_parameters']:>10,} "
            f"{s['best_val_loss']:>15.4f} {s['total_time_s']:>10.1f}"
        )
    print(f"{'='*80}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Specific variants to run (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per variant")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_ablations(
        variants=args.variants,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
