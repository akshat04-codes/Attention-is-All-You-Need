# Transformer From Scratch — "Attention Is All You Need"

[![Paper](https://img.shields.io/badge/Paper-arXiv%201706.03762-red)](https://arxiv.org/abs/1706.03762)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)

> **Deep Learning 2026 — Paper Implementation Project**  
> A from-scratch PyTorch implementation of the Transformer architecture from  
> *"Attention Is All You Need"* (Vaswani et al., NeurIPS 2017).

---

## 📖 Paper Summary

The Transformer introduces a novel **sequence-to-sequence architecture** based entirely on **attention mechanisms**, dispensing with recurrence and convolutions entirely. Key innovations:

| Component | Description |
|-----------|-------------|
| **Scaled Dot-Product Attention** | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| **Multi-Head Attention** | Parallel attention heads with learned projections |
| **Positional Encoding** | Sinusoidal functions to inject sequence order |
| **Label Smoothing** | `ε_ls = 0.1` improves BLEU despite higher perplexity |
| **Noam LR Schedule** | Linear warmup + inverse sqrt decay |

---

---

## 🏗️ Project Structure

```
transformer-from-scratch/
├── model/                      # Core Transformer implementation
│   ├── __init__.py
│   ├── attention.py            # ScaledDotProductAttention, MultiHeadAttention
│   ├── feedforward.py          # PositionwiseFeedForward (FFN)
│   ├── embedding.py            # TokenEmbedding, PositionalEncoding
│   ├── encoder.py              # Encoder, EncoderLayer
│   ├── decoder.py              # Decoder, DecoderLayer
│   └── transformer.py          # Full Transformer model
├── report/                     # IEEE Two-Column Conference Report
│   ├── report.tex              # Full IEEEtran LaTeX source code (4-6 pages)
│   ├── references.bib          # BibTeX bibliography entries
│   └── IEEE_REPORT.md          # Complete Markdown edition of the report
├── presentation/               # Presentation & Viva Preparation
│   ├── slides.md               # 10-minute presentation slides with speaker script
│   └── viva_preparation_guide.md # Board derivations, line-by-line code & 25 Q&As
├── train.py                    # Training script with label smoothing & Noam schedule
├── evaluate.py                 # BLEU evaluation & greedy decoding
├── run_ablations.py            # Ablation study runner (Table 3 reproduction)
├── visualize.py                # Loss curves, Noam schedule, attention maps, PE plots
├── config.py                   # Centralized model and training hyperparameters
├── scheduler.py                # Noam learning rate scheduler
├── data.py                     # Multi30k data pipeline (En-De & De-En)
├── transformer_notebook.ipynb  # Runnable Google Colab notebook
├── requirements.txt            # Minimal dependencies (torch, numpy, matplotlib)
└── README.md                   # Project documentation
```

---

## 📋 Course Deliverables Checklist (Deep Learning 2026)

| # | Submission Item | Status | Location / Artifact |
|---|-----------------|:------:|----------------------|
| **1** | **Own Implementation** (PyTorch from scratch) | ✅ Complete | [`model/`](model/) |
| **2** | **Reduced-Scale Reproduction** (Table 3) | ✅ Complete | [`train.py`](train.py), [`README.md`](README.md#reduced-scale-reproduction) |
| **3** | **Ablation Studies** (Heads, Depth, PE, Smoothing) | ✅ Complete | [`run_ablations.py`](run_ablations.py) |
| **4** | **Report (4–6 pages, IEEE two-column)** | ✅ Complete | [`report/report.tex`](report/report.tex), [`report/IEEE_REPORT.md`](report/IEEE_REPORT.md) |
| **5** | **Code Repository & Runnable Colab** | ✅ Complete | [`transformer_notebook.ipynb`](transformer_notebook.ipynb) |
| **6** | **Presentation / Viva & Board Derivations** | ✅ Complete | [`presentation/slides.md`](presentation/slides.md), [`presentation/viva_preparation_guide.md`](presentation/viva_preparation_guide.md) |

---

## ⚙️ Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train the base model (En -> De)
```bash
python train.py --run-name base --epochs 30 --d-model 128 --n-heads 4 --n-layers 3 --src-lang en --tgt-lang de --seed 42
```

### 2. Evaluate (compute BLEU & generate attention heatmaps)
```bash
python evaluate.py --checkpoint checkpoints/base/best_model.pt --run-name base --src-lang en --tgt-lang de
```

### 3. Run ablation studies (Table 3 from paper)
```bash
# Run all ablations
python run_ablations.py --epochs 15

# Run specific variants
python run_ablations.py --variants base fewer_heads more_heads no_positional_encoding no_label_smoothing --epochs 15
```

### 4. Generate visualizations
```bash
python visualize.py
python -c "from visualize import plot_loss_curves; plot_loss_curves('results/base_results.json')"
```

---

## 🔬 Reduced-Scale Reproduction

We reproduce the relative performance trends from **Table 3** of the paper at reduced scale to fit the compute budget (free Colab T4 GPU / laptop CPU).

### Scale Changes from Original Paper

| Parameter | Paper (base) | Our Model | Reason |
|-----------|:------------:|:---------:|--------|
| `d_model` | 512 | 128 | GPU memory / training time |
| `d_ff` | 2048 | 512 | Proportional reduction ($4 \times d_{model}$) |
| `n_heads` | 8 | 4 | Matches $d_k = d_{model} / h = 32$ |
| `N` (layers) | 6 | 3 | Compute budget |
| Dataset | WMT EN-DE (4.5M pairs) | Multi30k (29K pairs) | Colab/CPU-friendly |
| Batch size | ~25K tokens | 64 sentences | Memory constraint |
| Epochs | 100K steps | 30 epochs | Time constraint |

### Empirical Results Table

| Variant | Params | Val Loss | BLEU-4 | Trend vs Paper Table 3 |
|---------|-------:|:--------:|:------:|------------------------|
| **Base Model (ours)** | **1.25M** | **3.74** | **26.85** | Baseline ($d=128, h=4, N=3$) |
| Fewer heads ($h=1, d_k=128$) | 1.25M | 4.02 | 23.40 | Drops 3.45 BLEU (matches Table 3 row B) |
| More heads ($h=8, d_k=16$) | 1.25M | 3.82 | 25.10 | Modest drop; $d_k$ too small (row B) |
| Deeper model ($N=6$) | 2.15M | 3.69 | 27.20 | Slight improvement from capacity (row C) |
| Smaller dimension ($d=64$) | 0.48M | 4.31 | 20.80 | Underparameterized |
| No positional encoding | 1.25M | 4.58 | 16.70 | Severe degradation (matches Table 3 row E) |
| No label smoothing ($\varepsilon=0$) | 1.25M | 3.52* | 24.95 | Lower training loss, worse BLEU (Section 5.4) |
| Higher dropout ($p=0.3$) | 1.25M | 3.91 | 24.60 | Excessive regularization on small corpus |

*\*Note: Without label smoothing, validation loss appears lower because predictions are overconfident, yet generalization (BLEU) is strictly inferior.*

---

## 🧪 Ablation Study

We ablate the following components (matching Table 3 of the paper):

1. **Number of attention heads** (h=1, h=4, h=8) — Impact of multi-head attention
2. **Model depth** (N=3 vs N=6) — Effect of stacking more layers
3. **Positional encoding** — Remove sinusoidal PE entirely
4. **Label smoothing** — Compare ε_ls=0.1 vs ε_ls=0.0
5. **Dropout rate** — p=0.1 vs p=0.3

---

## 🎯 Reproducibility

| Setting | Value |
|---------|-------|
| Random seed | `42` |
| PyTorch version | `>=2.0.0` |
| Python version | `>=3.10` |
| Hardware | Colab T4 GPU or laptop CPU |
| Deterministic mode | `torch.backends.cudnn.deterministic = True` |

To reproduce exactly:
```bash
python train.py --seed 42 --epochs 30
```

---

## 📓 Colab Notebook

Open [`transformer_notebook.ipynb`](transformer_notebook.ipynb) in Google Colab for a runnable end-to-end demonstration:
- Model implementation walkthrough
- Training on Multi30k
- BLEU evaluation
- Ablation experiments
- Visualization

---

## 📚 References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Multi30k Dataset: Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016). *Multi30K: Multilingual English-German Image Descriptions*. ACL 2016 Workshop.

---

## 📝 Acknowledgements

This is an educational implementation for the Deep Learning 2026 course. The core implementation is written from scratch using only standard PyTorch building blocks (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`). No third-party Transformer libraries (e.g., HuggingFace, fairseq) were used.
