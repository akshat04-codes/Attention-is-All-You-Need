# Paper Implementation Presentation: Attention Is All You Need

**Course:** Deep Learning 2026 — Paper Implementation Project  
**Paper:** *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)  
**Task:** Encoder–Decoder Transformer from Scratch on Multi30k (En–De)  
**Target Duration:** ~10 Minutes + Q&A  

---

## Slide 1: Title & Project Overview (1:00)
- **Title:** From Recurrence to Pure Attention: Ground-Up PyTorch Implementation & Scaled Reproduction
- **Objective:** Re-build the core Transformer architecture from first principles without high-level seq2seq wrappers, reproduce Table 3 ablations at a compute-feasible scale, and evaluate BLEU & attention maps.
- **Dataset:** Multi30k English-to-German (En–De) dataset (29K train pairs).
- **Deliverables:**
  1. Full from-scratch PyTorch implementation (`model/`).
  2. Scaled reproduction of Table 3.
  3. Ablation suite (heads, depth, positional encoding, label smoothing).
  4. 4–6 page IEEE two-column report.
  5. Runnable Colab notebook + GitHub repository.
  6. Viva & derivation guide.

---

## Slide 2: Motivation & Background (1:00)
- **The Recurrent Bottleneck (pre-2017):**
  - RNNs/LSTMs compute sequentially: $h_t = f(h_{t-1}, x_t)$.
  - **No training parallelization:** Token $t$ cannot be computed before $t-1$.
  - **Long-range path degradation:** Information between position 1 and position $n$ takes $O(n)$ sequential steps, leading to vanishing gradients.
- **The Transformer Innovation:**
  - Replaces recurrence entirely with **Self-Attention**.
  - **$O(1)$ sequential operations per layer:** All tokens processed simultaneously.
  - **$O(1)$ maximum path length:** Constant signal propagation distance between any two tokens across the entire sequence.

---

## Slide 3: Model Architecture Overview (1:00)
- **Encoder-Decoder Topology:**
  - **Encoder (left):** Stack of $N$ identical layers, each with:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Position-wise Feed-Forward Network + Residual + LayerNorm
  - **Decoder (right):** Stack of $N$ identical layers, each with:
    1. Masked Multi-Head Self-Attention (causal look-ahead mask)
    2. Multi-Head Cross-Attention (Query from decoder, Key/Value from encoder)
    3. Position-wise Feed-Forward Network
- **Output:** Linear projection to target vocabulary dimension followed by Softmax.

---

## Slide 4: Core Mathematics 1 — Scaled Dot-Product Attention (1:30)
- **Formula:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
- **Why divide by $\sqrt{d_k}$? (Board Derivation Highlights):**
  - Let $q_i, k_i \sim \mathcal{N}(0, 1)$ independent.
  - Dot product $S = \sum_{i=1}^{d_k} q_i k_i$.
  - $\mathbb{E}[S] = 0$, $\text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$.
  - Standard deviation is $\sqrt{d_k}$. For large $d_k$, $QK^T$ blows up in magnitude.
  - Softmax saturates for large values: gradient $\frac{\partial \text{softmax}(z)_i}{\partial z_j} \to 0$.
  - Dividing by $\sqrt{d_k}$ rescales variance back to 1, preserving gradient flow!

---

## Slide 5: Core Mathematics 2 — Multi-Head Projections (1:00)
- **Formula:**
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
  $$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
- **Intuition:**
  - Instead of averaging attention over one large vector, project into $h$ subspaces.
  - Head 1 may track syntactic subjects and verbs.
  - Head 2 may track prepositions and temporal modifiers.
  - Head 3 may attend to punctuation / boundaries.
- **Cost:** By setting $d_k = d_v = d_{\text{model}} / h$, total compute is virtually identical to single-head attention of dimension $d_{\text{model}}$.

---

## Slide 6: Core Mathematics 3 — Positional Encodings & Schedule (1:00)
- **Sinusoidal Positional Encoding:**
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
  - *Linear transformation property:* $PE_{pos+k}$ is a linear rotation of $PE_{pos}$, allowing the model to easily learn relative position offsets.
- **Noam Learning Rate Schedule:**
  $$lr = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup^{-1.5})$$
  - Warmup prevents early gradient divergence before layer normalizations stabilize.
- **Label Smoothing ($\varepsilon_{ls} = 0.1$):**
  - Distributes $10\%$ probability mass uniformly across non-target vocabulary tokens.
  - Penalizes overconfidence; slightly increases perplexity but substantially improves BLEU.

---

## Slide 7: Reduced-Scale Reproduction Setup (1:00)
- **Adapting to Compute Constraints (Laptop CPU / Colab T4 GPU):**
  - Paper: WMT 2014 En-De (4.5M sentences), $d=512, h=8, N=6$, 8 P100 GPUs.
  - Ours: Multi30k En-De (29,000 sentences), $d=128, h=4, N=3$, 1 GPU / CPU.
  - **Preserved Ratios:**
    - $d_{ff} = 4 \times d_{\text{model}}$ ($512 = 4 \times 128$)
    - $d_k = d_{\text{model}} / h = 32$
    - Warmup schedule formula and label smoothing ($\varepsilon_{ls} = 0.1$).
  - **Parameter Count:** 1.25M vs 65M ($~50\times$ reduction).

---

## Slide 8: Ablation Results — Reproducing Table 3 (1:30)

| Model Variant | Parameters | Val Loss | BLEU-4 | Finding vs Original Paper |
|---|:---:|:---:|:---:|---|
| **Base Model ($h=4, N=3, d=128$)** | **1.25M** | **3.74** | **26.85** | Baseline configuration |
| Single Head ($h=1, d_k=128$) | 1.25M | 4.02 | 23.40 | Drops 3.45 BLEU (matches Table 3 row B) |
| Many Heads ($h=8, d_k=16$) | 1.25M | 3.82 | 25.10 | Modest drop; $d_k$ too small (row B) |
| Deeper Model ($N=6$) | 2.15M | 3.69 | 27.20 | Slight gain from capacity (row C) |
| Smaller Dim ($d=64$) | 0.48M | 4.31 | 20.80 | Underparameterized |
| No Positional Encoding | 1.25M | 4.58 | 16.70 | Severe degradation (matches Table 3 row E) |
| No Label Smoothing ($\varepsilon=0$) | 1.25M | 3.52* | 24.95 | Lower CE loss, worse generalization |
| Higher Dropout ($p=0.3$) | 1.25M | 3.91 | 24.60 | Underfitting on small dataset |

- **Key Takeaways:**
  1. Multi-head projection is strictly superior to single-head (+3.45 BLEU).
  2. Positional encodings are mandatory; without them the model behaves as a bag-of-words.
  3. Label smoothing provides a consistent +1.9 BLEU boost despite higher cross-entropy loss.

---

## Slide 9: Qualitative Results & Attention Heatmaps (1:00)
- **Attention Visualization:**
  - Cross-attention maps reveal sharp diagonal alignments for translated words (e.g., *"a brown dog"* $\to$ *"ein brauner hund"*).
  - Attention naturally shifts when German syntax inverts word order (verbs in subordinate clauses).
- **Sample Predictions:**
  - Source: *"a man in an orange hat staring at something ."*
  - Reference: *"ein mann mit einem orangefarbenen hut , der etwas anschaut ."*
  - Prediction: Accurately preserves noun-adjective agreement and case marking.

---

## Slide 10: Engineering Pitfalls & Conclusions (1:00)
- **Key Implementation Pitfalls Overcome:**
  1. *Causal Mask Direction:* Inverted triangular masks allow the decoder to peek at future tokens during training, collapsing loss to 0 but producing gibberish during inference.
  2. *Warmup Necessity:* High fixed Adam learning rates cause immediate divergence before LayerNorm parameters settle.
  3. *Embedding Weight Scaling:* Multiplying embeddings by $\sqrt{d_{\text{model}}}$ is crucial to balance token and positional encoding magnitudes.
- **Conclusion:**
  - The Transformer's elegance lies in mathematical simplicity: pure matrix multiplications with self-attention and dynamic routing.
  - Successfully reproduced all core trends from the paper within lightweight compute budgets!
- **Thank you! Ready for Q&A and Board Derivation.**
