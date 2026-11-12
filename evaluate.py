"""
Evaluation — BLEU Score and Translation Quality
=================================================
Implements:
    - Greedy decoding (for fast evaluation)
    - BLEU score computation (tokenized BLEU-4)
    - Sample translations for qualitative analysis

BLEU score is the standard metric reported in the paper (Table 2).
"""

import os
import json
import math
import argparse
from collections import Counter
from pathlib import Path

import torch

from model.transformer import Transformer
from config import ModelConfig
from data import load_multi30k, Vocabulary


# ---------------------------------------------------------------------------
# BLEU Score (from scratch — no nltk dependency needed)
# ---------------------------------------------------------------------------


def compute_ngrams(tokens: list[str], n: int) -> Counter:
    """Compute n-gram counts for a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> dict:
    """
    Compute corpus-level BLEU score (BLEU-1 through BLEU-4).

    Uses the standard brevity penalty and clipped n-gram precision.

    Args:
        references:  List of reference token lists.
        hypotheses:  List of hypothesis token lists.
        max_n:       Maximum n-gram order (4 for BLEU-4).

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    """
    precisions = []
    bp_r = 0  # total reference length
    bp_c = 0  # total hypothesis length

    for n in range(1, max_n + 1):
        clipped_count = 0
        total_count = 0

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = compute_ngrams(ref, n)
            hyp_ngrams = compute_ngrams(hyp, n)

            for ngram, count in hyp_ngrams.items():
                clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += max(len(hyp) - n + 1, 0)

        precision = clipped_count / max(total_count, 1)
        precisions.append(precision)

    # Brevity penalty
    for ref, hyp in zip(references, hypotheses):
        bp_r += len(ref)
        bp_c += len(hyp)

    if bp_c == 0:
        bp = 0
    elif bp_c >= bp_r:
        bp = 1.0
    else:
        bp = math.exp(1 - bp_r / bp_c)

    # Compute BLEU scores
    results = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            log_avg = sum(math.log(p) for p in precisions[:n]) / n
            bleu = bp * math.exp(log_avg)
        else:
            bleu = 0.0
        results[f"BLEU-{n}"] = round(bleu * 100, 2)

    return results


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy decoding: at each step, pick the token with highest probability.

    Args:
        model:   Trained Transformer model.
        src:     (1, src_len) — source sequence.
        max_len: Maximum decoding length.
        sos_idx: Start-of-sequence token index.
        eos_idx: End-of-sequence token index.
        device:  Torch device.

    Returns:
        (1, decoded_len) — decoded token indices.
    """
    model.eval()
    src = src.to(device)

    # Encode source
    encoder_output, src_mask = model.encode(src)

    # Start with <sos>
    tgt = torch.tensor([[sos_idx]], device=device)

    for _ in range(max_len):
        logits = model.decode(tgt, encoder_output, src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return tgt


from visualize import plot_attention_heatmap


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_model(
    checkpoint_path: str,
    run_name: str = "base",
    num_samples: int = 5,
    src_lang: str = "en",
    tgt_lang: str = "de",
) -> dict:
    """
    Full evaluation: compute BLEU on test set and show sample translations.

    Args:
        checkpoint_path: Path to model checkpoint.
        run_name:        Name for results file.
        num_samples:     Number of sample translations to display.
        src_lang:        Source language.
        tgt_lang:        Target language.

    Returns:
        Evaluation results dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader, src_vocab, tgt_vocab = load_multi30k(
        batch_size=1, src_lang=src_lang, tgt_lang=tgt_lang
    )

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("model_config", {})

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 512),
        n_layers=model_cfg.get("n_layers", 3),
        max_len=model_cfg.get("max_len", 256),
        dropout=model_cfg.get("dropout", 0.1),
        use_positional_encoding=model_cfg.get("use_positional_encoding", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Decode all test examples
    references = []
    hypotheses = []
    sample_translations = []

    for idx, (src, tgt) in enumerate(test_loader):
        src = src.to(device)

        # Greedy decode
        decoded = greedy_decode(
            model, src, max_len=128,
            sos_idx=Vocabulary.SOS_IDX,
            eos_idx=Vocabulary.EOS_IDX,
            device=device,
        )

        # Convert to tokens
        ref_tokens = tgt_vocab.decode(tgt[0].tolist())
        hyp_tokens = tgt_vocab.decode(decoded[0].tolist())
        src_tokens = src_vocab.decode(src[0].tolist())

        references.append(ref_tokens)
        hypotheses.append(hyp_tokens)

        # Collect samples and generate attention maps
        if idx < num_samples:
            sample_translations.append({
                "source": " ".join(src_tokens),
                "reference": " ".join(ref_tokens),
                "hypothesis": " ".join(hyp_tokens),
            })

            try:
                if decoded.size(1) > 1:
                    attns = model.get_attention_maps(src, decoded)
                    cross_attns = attns["decoder_cross_attentions"][-1]
                    plot_attention_heatmap(
                        cross_attns,
                        src_tokens=src_tokens,
                        tgt_tokens=hyp_tokens,
                        head=0,
                        layer=len(attns["decoder_cross_attentions"]) - 1,
                        save_path=f"results/plots/{run_name}_attn_sample{idx+1}.png",
                    )
            except Exception as e:
                print(f"Warning: Could not save attention map for sample {idx+1}: {e}")

    # Compute BLEU
    bleu_scores = compute_bleu(references, hypotheses)

    print(f"\n{'='*60}")
    print(f"Evaluation Results -- {run_name}")
    print(f"{'='*60}")
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.2f}")

    print(f"\nSample Translations:")
    print(f"{'-'*60}")
    for i, sample in enumerate(sample_translations):
        print(f"  [{i+1}] SRC: {sample['source']}")
        print(f"       REF: {sample['reference']}")
        print(f"       HYP: {sample['hypothesis']}")
        print()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    eval_results = {
        "run_name": run_name,
        "bleu_scores": bleu_scores,
        "sample_translations": sample_translations,
        "test_set_size": len(references),
    }

    eval_path = results_dir / f"{run_name}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Transformer")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (e.g., checkpoints/base/best_model.pt)"
    )
    parser.add_argument("--run-name", type=str, default="base", help="Run name")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to show")
    parser.add_argument("--src-lang", type=str, default="en", help="Source language (en or de)")
    parser.add_argument("--tgt-lang", type=str, default="de", help="Target language (de or en)")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
        num_samples=args.num_samples,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )


if __name__ == "__main__":
    main()
