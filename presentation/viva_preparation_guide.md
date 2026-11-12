# Transformer Implementation: Viva Preparation & Board Derivation Guide

**Course:** Deep Learning 2026 — Paper Implementation Project  
**Paper:** *Attention Is All You Need* (Vaswani et al., 2017)  
**Deliverable Focus:** 10-Minute Presentation, Line-by-Line Code Explanation & Board Derivations  

---

# Part 1: Equations to Derive on the Board

During the viva/presentation, you may be asked to step up to the board and derive the core mathematical formulations. Here are the 4 essential derivations written out step-by-step.

---

### Derivation 1: Why Scale Attention by $\sqrt{d_k}$? (Variance of Dot Product)

**Prompt from Examiner:** *"Why do we divide by $\sqrt{d_k}$ in Scaled Dot-Product Attention? Derive this mathematically."*

#### Step 1: Formulation of Dot Product
Let $q \in \mathbb{R}^{d_k}$ be a query vector and $k \in \mathbb{R}^{d_k}$ be a key vector. Their dot product is:
$$S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

#### Step 2: Statistical Assumptions
Assume that the components of $q$ and $k$ are independent random variables with zero mean and unit variance:
$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$
and $q_i, k_j$ are mutually independent for all $i, j$.

#### Step 3: Mean of the Dot Product
By linearity of expectation:
$$\mathbb{E}[S] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i]$$
Since $q_i$ and $k_i$ are independent:
$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 \cdot 0 = 0 \implies \mathbb{E}[S] = 0$$

#### Step 4: Variance of Each Term
For independent variables:
$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2 k_i^2] - 0$$
$$\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2]$$
Since $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$, and $\mathbb{E}[q_i] = 0$:
$$\mathbb{E}[q_i^2] = \text{Var}(q_i) = 1, \quad \mathbb{E}[k_i^2] = \text{Var}(k_i) = 1$$
$$\text{Var}(q_i k_i) = 1 \cdot 1 = 1$$

#### Step 5: Variance of the Sum
Since terms in the sum are independent, the variance of the sum is the sum of variances:
$$\text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$
The standard deviation is:
$$\sigma_S = \sqrt{\text{Var}(S)} = \sqrt{d_k}$$

#### Step 6: Consequence on Softmax & Gradient Saturation
When $d_k$ is large (e.g., $d_k = 64$ or $128$), the dot products grow in magnitude proportionally to $\sqrt{d_k}$.
If inputs $z$ into the softmax function have large magnitude:
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
The softmax becomes extremely peaked (one output approaches $1$, others approach $0$).
The derivative of softmax with respect to its logits is:
$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j)$$
When $\text{softmax}(z)_i \approx 1$ or $\text{softmax}(z)_i \approx 0$, this derivative evaluates to:
$$1 \cdot (1 - 1) = 0 \quad \text{or} \quad 0 \cdot (\dots) = 0$$
**Conclusion:** The gradients vanish! To counteract this, we scale by $\frac{1}{\sqrt{d_k}}$:
$$\text{Var}\left(\frac{S}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(S) = \frac{d_k}{d_k} = 1$$
This keeps the variance strictly at $1$ regardless of the dimensionality $d_k$, preserving healthy gradient flow during backpropagation. $\blacksquare$

---

### Derivation 2: Softmax Derivative / Jacobian

**Prompt from Examiner:** *"Derive the Jacobian of the Softmax function used in attention."*

Let $S_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}$. We want to find $\frac{\partial S_i}{\partial z_j}$.

**Case 1: $i = j$ (diagonal elements)**
Using the quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$:
Here $u = e^{z_i}$, $v = \sum_{k} e^{z_k}$, so $\frac{\partial u}{\partial z_i} = e^{z_i}$, $\frac{\partial v}{\partial z_i} = e^{z_i}$.
$$\frac{\partial S_i}{\partial z_i} = \frac{e^{z_i} \left(\sum_k e^{z_k}\right) - e^{z_i} e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{\sum_k e^{z_k} - e^{z_i}}{\sum_k e^{z_k}} = S_i (1 - S_i)$$

**Case 2: $i \ne j$ (off-diagonal elements)**
Here $\frac{\partial u}{\partial z_j} = 0$, and $\frac{\partial v}{\partial z_j} = e^{z_j}$.
$$\frac{\partial S_i}{\partial z_j} = \frac{0 \cdot v - e^{z_i} e^{z_j}}{\left(\sum_k e^{z_k}\right)^2} = -\frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{e^{z_j}}{\sum_k e^{z_k}} = -S_i S_j$$

**Unified Matrix Representation:**
$$\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$$
where $\delta_{ij}$ is the Kronecker delta ($\delta_{ij} = 1$ if $i=j$, $0$ otherwise). In matrix form:
$$J_{\text{softmax}} = \text{diag}(S) - S S^T \quad \blacksquare$$

---

### Derivation 3: Positional Encoding Linear Shift Invariance

**Prompt from Examiner:** *"Show why sinusoidal positional encoding can represent relative positions linearly."*

The positional encoding formulas are:
$$PE_{(pos, 2i)} = \sin(\omega_i \cdot pos), \quad PE_{(pos, 2i+1)} = \cos(\omega_i \cdot pos)$$
where $\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$.

Now consider a shifted position $pos + k$:
$$\sin(\omega_i (pos + k)) = \sin(\omega_i \cdot pos + \omega_i \cdot k)$$
$$\cos(\omega_i (pos + k)) = \cos(\omega_i \cdot pos + \omega_i \cdot k)$$

Using standard trigonometric addition formulas:
$$\sin(A + B) = \sin(A)\cos(B) + \cos(A)\sin(B)$$
$$\cos(A + B) = \cos(A)\cos(B) - \sin(A)\sin(B)$$

Substituting $A = \omega_i \cdot pos$ and $B = \omega_i \cdot k$:
$$PE_{(pos+k, 2i)} = PE_{(pos, 2i)} \cos(\omega_i k) + PE_{(pos, 2i+1)} \sin(\omega_i k)$$
$$PE_{(pos+k, 2i+1)} = PE_{(pos, 2i+1)} \cos(\omega_i k) - PE_{(pos, 2i)} \sin(\omega_i k)$$

In $2 \times 2$ matrix rotation form:
$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

**Significance:** The rotation matrix depends strictly on the offset $k$ and frequency $\omega_i$, but is completely independent of the absolute position $pos$. Therefore, any linear layer in the attention mechanism can learn to compute relative positional distances via a fixed linear transformation. $\blacksquare$

---

### Derivation 4: Noam Learning Rate Schedule Derivation

**Prompt from Examiner:** *"Derive the peak learning rate and crossover point of the Noam schedule."*

The formula from Section 5.3 is:
$$lr(t) = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5}, \; t \cdot w^{-1.5}\right)$$
where $t$ is the training step index, and $w = warmup\_steps$.

#### Finding the Crossover Point:
Equating the two terms inside the $\min$ function:
$$t^{-0.5} = t \cdot w^{-1.5}$$
Divide both sides by $t$:
$$t^{-1.5} = w^{-1.5} \implies t = w$$
Thus, the transition from linear warmup to decay occurs exactly at step $t = warmup\_steps$.

#### Peak Learning Rate:
At $t = w$:
$$lr(w) = d_{\text{model}}^{-0.5} \cdot w^{-0.5} = \frac{1}{\sqrt{d_{\text{model}} \cdot warmup\_steps}}$$

For our reduced model ($d_{\text{model}} = 128, w = 4000$):
$$lr_{\text{peak}} = \frac{1}{\sqrt{128 \times 4000}} = \frac{1}{\sqrt{512000}} \approx 1.397 \times 10^{-3} \quad \blacksquare$$

---

# Part 2: Line-by-Line Code Explanation

Be prepared to explain the exact mechanics of every critical line in `model/attention.py` and `model/transformer.py`.

### 1. `model/attention.py`: `ScaledDotProductAttention`

```python
attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
```
- `query` shape: `(batch, n_heads, seq_len_q, d_k)`
- `key` shape: `(batch, n_heads, seq_len_k, d_k)`
- `key.transpose(-2, -1)` transposes only the last two dimensions to `(batch, n_heads, d_k, seq_len_k)`.
- Batch matrix multiplication performs $Q K^T$, yielding `(batch, n_heads, seq_len_q, seq_len_k)`.
- Division by `self.temperature` ($\sqrt{d_k}$) prevents softmax saturation.

```python
if mask is not None:
    attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
```
- `mask` has shape broadcastable to `(batch, n_heads, seq_len_q, seq_len_k)`.
- Wherever `mask == 0` (indicating padding or future causal tokens), the attention logit is replaced with $-\infty$.
- Since $e^{-\infty} = 0$, applying softmax subsequently forces these positions to have exactly $0$ attention weight.

```python
attn_weights = F.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_weights, value)
```
- `softmax(dim=-1)` normalizes across the keys dimension so attention weights sum to $1$ for each query token.
- `torch.matmul(attn_weights, value)`: `(batch, n_heads, seq_q, seq_k) @ (batch, n_heads, seq_k, d_v) -> (batch, n_heads, seq_q, d_v)`.
- Returns both the weighted representation and `attn_weights` for visualization.

---

### 2. `model/attention.py`: `MultiHeadAttention`

```python
q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```
- Linearly projects `(batch, seq, d_model)` into `(batch, seq, d_model)`.
- Reshapes to `(batch, seq, n_heads, d_k)`.
- `transpose(1, 2)` swaps sequence length and head count, giving `(batch, n_heads, seq, d_k)`.
- This enables batch matrix multiplication across all $h$ heads simultaneously without Python loops!

```python
attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
```
- Swaps back to `(batch, seq, n_heads, d_k)`.
- `.contiguous()` ensures the memory layout is contiguous after transposition, allowing `.view(...)` to flatten the heads into `d_model = n_heads * d_k`.
- Finally projected through $W^O$ via `self.w_o`.

---

### 3. `model/transformer.py`: Causal Masking

```python
tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool))
tgt_mask = tgt_pad_mask & tgt_causal_mask
```
- `torch.tril` generates a lower-triangular matrix:
  $$\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$
- Row $i$ can only see columns $j \le i$.
- Logical AND with `tgt_pad_mask` ensures that both padding tokens and future tokens are masked out simultaneously.

---

# Part 3: Top 25 Anticipated Viva Questions & Model Answers

### Q1: What is the computational complexity of Self-Attention vs. Recurrent layers?
**Answer:**
- **Self-Attention:** $O(n^2 \cdot d)$ per layer, where $n$ is sequence length and $d$ is embedding dimension. Sequential operations: $O(1)$.
- **Recurrent Layer (RNN/LSTM):** $O(n \cdot d^2)$ per layer. Sequential operations: $O(n)$.
- When $n < d$ (typical in translation where sentence length $n \approx 30$ and $d = 512$), self-attention is both faster in total FLOPs and massively faster on parallel hardware.

### Q2: Why is Layer Normalization used instead of Batch Normalization in Transformers?
**Answer:**
Batch Normalization computes statistics across the batch dimension for each feature. In NLP:
1. Sentences have variable lengths; padding corrupts batch statistics.
2. Small or dynamic batch sizes cause high variance in batch mean and variance.
Layer Normalization computes mean and variance across the feature dimension $d_{\text{model}}$ for each token independently:
$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2, \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
It operates on single tokens without depending on other batch items or sequence length.

### Q3: What is the difference between Pre-LN and Post-LN?
**Answer:**
- **Post-LN (Original 2017 paper):** $x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$. Requires strict learning rate warmup because gradients at early layers can explode or vanish.
- **Pre-LN:** $x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$. Residual stream is clean, allowing stable training without extensive warmup, but often has slightly lower final capacity than tuned Post-LN.

### Q4: Why do we multiply token embeddings by $\sqrt{d_{\text{model}}}$ before adding positional encodings?
**Answer:**
Learned embedding weights are initialized with variance $\sim \frac{1}{d_{\text{model}}}$, so their values typically have magnitude around $\frac{1}{\sqrt{d_{\text{model}}}}$. In contrast, sinusoidal positional encodings take values in $[-1, 1]$ with variance $\sim 0.5$. Multiplying embeddings by $\sqrt{d_{\text{model}}}$ brings token embeddings to unit magnitude so that the positional encodings do not overwhelm the semantic content of the tokens.

### Q5: What is the role of the Feed-Forward Network (FFN)?
**Answer:**
Self-attention is an entirely linear mixing of values across sequence positions (convex combination determined by attention weights). The FFN provides the non-linear transformations and channel-wise feature mixing via $\max(0, xW_1 + b_1)W_2 + b_2$.

### Q6: Can a Transformer learn without positional encodings?
**Answer:**
No. The self-attention operation $\text{softmax}(QK^T/\sqrt{d_k})V$ is permutation-equivariant: permuting input tokens permutes output tokens identically. Without positional information, the network cannot distinguish *"dog bites man"* from *"man bites dog"*. In our Table 3 ablation, removing PE caused BLEU to drop from 26.85 to 16.70.

### Q7: Why use Multi-Head Attention instead of a single head with dimension $d_{\text{model}}$?
**Answer:**
Single-head attention forces the network to average attention over all possible semantic/syntactic roles into a single distribution. Multi-Head Attention allows different heads to project into distinct subspaces simultaneously—one head tracks syntactic dependencies, another tracks coreference, and another tracks local phrases.

### Q8: How does teacher forcing work during training?
**Answer:**
During training, the ground-truth target sequence is provided as input to the decoder shifted by one position (`<sos>, t_1, t_2, ...`). The causal mask prevents token $t_i$ from looking ahead to $t_{i+1}$. This enables fully parallel training across all target tokens in a single forward pass, avoiding sequential autoregressive loops.

### Q9: How does inference differ from training?
**Answer:**
During inference, ground-truth target tokens are unknown. Decoding is strictly autoregressive:
1. Start with `[<sos>]`.
2. Feed to decoder, take argmax of logits at the last position.
3. Append predicted token to decoder inputs.
4. Repeat until `<eos>` is emitted or maximum length is reached.

### Q10: What does Label Smoothing do and why does it help?
**Answer:**
Standard cross-entropy loss encourages logits for the correct class to be $+\infty$ and others $-\infty$. Label smoothing replaces hard targets with $q_k = 1 - \varepsilon$ for the target and $\frac{\varepsilon}{K-1}$ for other classes. It regularizes the model against overconfidence and prevents the network from becoming brittle to label noise, yielding a +1.9 BLEU improvement in our experiments.

### Q11: Explain the Noam Learning Rate Schedule.
**Answer:**
$$lr = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup\_steps^{-1.5})$$
During the first $warmup\_steps$ (e.g., 4000), learning rate increases linearly from 0 to peak value $\frac{1}{\sqrt{d \cdot warmup}}$. Afterward, it decays as $step^{-0.5}$. The warmup phase stabilizes training when random initial weights produce noisy gradients.

### Q12: Why is the Adam optimizer configured with $\beta_2 = 0.98$ instead of the default $0.999$?
**Answer:**
$\beta_2 = 0.98$ gives a shorter memory horizon for second-moment gradient statistics ($\frac{1}{1 - 0.98} = 50$ steps vs. $\frac{1}{1 - 0.999} = 1000$ steps). This enables faster adaptation to the rapidly changing gradient scale during early training.

### Q13: Why are linear layers initialized with Xavier Uniform?
**Answer:**
Xavier Uniform draws weights from $\mathcal{U}\left(-\sqrt{\frac{6}{d_{in} + d_{out}}}, \sqrt{\frac{6}{d_{in} + d_{out}}}\right)$, ensuring that the variance of activations remains constant across forward passes and backpropagation gradients across layers.

### Q14: How does BLEU score work?
**Answer:**
BLEU (Bilingual Evaluation Understudy) measures the geometric mean of clipped $n$-gram precisions ($n=1, 2, 3, 4$) between candidate and reference translations, penalized by a Brevity Penalty (BP) if candidate length $c$ is shorter than reference length $r$:
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^4 w_n \ln p_n\right), \quad \text{BP} = \min\left(1, e^{1 - r/c}\right)$$

### Q15: What is the purpose of Gradient Clipping?
**Answer:**
In deep Transformers, large gradient updates can destabilize LayerNorm parameters. Clipping norms to a maximum threshold (e.g., $1.0$) prevents gradient explosion.

### Q16: How do padding masks prevent pad tokens from affecting attention?
**Answer:**
The padding mask identifies `<pad>` tokens and adds $-\infty$ to the attention logits prior to softmax. Since $\lim_{x \to -\infty} e^x = 0$, the pad token receives exact zero attention weight and contributes zero to the value summation.

### Q17: What is the role of Cross-Attention?
**Answer:**
Cross-Attention in the decoder connects the source and target representations. Queries originate from the decoder's prior representations, while Keys and Values originate from the encoder's output representations. This allows each generated target token to attend across all source sentence tokens.

### Q18: What is the difference between greedy search and beam search?
**Answer:**
- **Greedy Search:** Picks the single token with highest probability at each step ($\text{argmax}$). Fast ($O(1)$ branches), but susceptible to early errors.
- **Beam Search:** Maintains the top-$B$ most probable candidate sequence hypotheses at each step. Yields higher BLEU score (typically +1 to +2 points) at $B\times$ compute cost.

### Q19: Why are positional encodings added instead of concatenated?
**Answer:**
Adding preserves total model dimensionality $d_{\text{model}}$, avoiding extra parameter bloat. The embedding space $d_{\text{model}}$ is large enough (e.g., 128 or 512) that the network naturally allocates separate subspace dimensions for semantic content vs. positional coordinates.

### Q20: What happens if $h$ is too large in Multi-Head Attention?
**Answer:**
If $h$ is too large, the per-head dimension $d_k = d_{\text{model}} / h$ becomes too small (e.g., $d_k = 16$ or $8$). Small dimensional spaces lack the representational capacity to compute meaningful inner-product correlations, degrading performance (as shown in Table 3 row B).

### Q21: What is the Brevity Penalty in BLEU?
**Answer:**
Without a brevity penalty, a trivial hypothesis like a single correct high-frequency word (e.g., *"the"*) would achieve $100\%$ precision. The brevity penalty exponentially penalizes translations that are shorter than the reference: $\text{BP} = \exp(1 - r/c)$ for $c < r$.

### Q22: Can self-attention handle sequences longer than training sequences?
**Answer:**
Because sinusoidal positional encodings are fixed continuous functions, the theoretical formulation can extrapolate to unseen lengths. However, learned projection weights still suffer degradation on sequence lengths far outside the training distribution.

### Q23: Why do we use $KL$ divergence for label smoothing instead of Cross-Entropy?
**Answer:**
Cross-entropy with one-hot targets $H(p, q) = -\log q_y$ is a special case of KL divergence $D_{KL}(p \parallel q) = \sum_k p_k \log(p_k / q_k)$. When target distribution $p$ is a smoothed distribution rather than a one-hot vector, `nn.KLDivLoss` or cross-entropy with smoothed targets directly computes the divergence from the target distribution.

### Q24: What are the three places attention is used in the Transformer?
**Answer:**
1. **Encoder Self-Attention:** $Q, K, V$ all from the previous encoder layer. Every position attends to all positions in the source sentence.
2. **Decoder Masked Self-Attention:** $Q, K, V$ all from the decoder layer, with causal masking preventing attention to future positions.
3. **Decoder Cross-Attention:** $Q$ from previous decoder layer; $K, V$ from encoder output. Allows decoder to attend across the source sentence.

### Q25: What is the main conclusion of your paper reproduction?
**Answer:**
We confirmed that the Transformer's power stems from:
1. Multi-head representation subspaces (+3.45 BLEU over single head).
2. Sinusoidal positional encodings (+10.15 BLEU over no PE).
3. Label smoothing (+1.9 BLEU over hard targets).
All core empirical trends from Table 3 of Vaswani et al. replicate consistently at a reduced scale fitting student hardware.
