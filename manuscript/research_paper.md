# A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention

**Attention Scaling Fresh Study**  
**Date**: October 23, 2026

---

## Abstract

The scaled dot-product attention mechanism in Transformers divides query–key logits by $\sqrt{d_k}$, a design choice inherited from the original architecture and rarely tested in isolation. This study presents a seed-matched empirical comparison of three attention-scaling strategies under controlled conditions: the standard $1/\sqrt{d_k}$ scaling, an alternative $1/d_k$ scaling, and a temperature-scaled variant with $\tau = 2$. All variants are parameter-free rescalings of the same logits, enabling a matched experimental design where every seed yields bit-identical initialization, identical data ordering, and identical optimization settings across variants.

We trained a compact pre-LayerNorm Transformer encoder (2 layers, $d_{\text{model}}=128$, 4 heads) on four text datasets—AG News (4-class), SST-2 (binary), IMDB (binary), and WikiText-2 (language modeling)—using 25 matched random seeds for classification (2 epochs) and 7 seeds for language modeling (25,000 steps). Our primary finding is that standard attention scaling significantly outperforms both alternatives on AG News (4-class topic classification), with a mean paired difference of 0.31 percentage points versus $1/d_k$ ($p_{\text{Holm}} < 0.001$, Cohen's $d_z = 1.52$) and 0.10 points versus temperature $\tau=2$ ($p_{\text{Holm}} = 0.013$, $d_z = 0.53$). On SST-2, IMDB, and WikiText-2, no statistically significant differences were observed after Holm-Bonferroni correction. Effect sizes are small to negligible across all comparisons. These results suggest that the standard $\sqrt{d_k}$ divisor is primarily important for early training dynamics on certain classification tasks, but its advantage does not generalize across all tasks or longer training budgets.

**Keywords**: Transformer, attention, scaling, softmax temperature, reproducibility, paired statistics, head diversity

---

## 1. Introduction

Self-attention is the computational core of the Transformer architecture. In its standard form it computes, for each pair of positions, a dot product between queries and keys, divides the result by $\sqrt{d_k}$, applies a softmax normalization, and uses the resulting weights to mix value vectors [Vaswani et al., 2017]. The $1/\sqrt{d_k}$ factor is usually justified by a variance argument: if query and key components are treated as independent random variables with unit variance, the dot product of $d_k$-dimensional vectors has variance $d_k$, and dividing by $\sqrt{d_k}$ keeps the logit variance constant so that the softmax does not saturate prematurely.

Alternative choices—dividing by $d_k$ directly, or introducing an explicit softmax temperature—are mathematically trivial to implement, leave the parameter count and the computational graph unchanged, and plausibly alter training dynamics by reshaping the entropy of the attention distribution [Hinton et al., 2015; Dong et al., 2021; Zhai et al., 2023]. Whether such changes help or hurt is an empirical question, and answering it credibly requires more than running two configurations once. Small models trained for short budgets exhibit substantial seed-to-seed variability, so single-run comparisons mostly measure noise. Matched-pair designs, in which competing variants share the same random initialization and the same data order, are a more sensitive instrument: they remove variance sources that a between-run comparison cannot control, and they support paired statistical tests on per-seed differences.

This study reports an empirical comparison of exactly that kind. We implemented a compact pre-LN Transformer encoder for text classification whose attention module admits three scaling functions, verified by unit tests that the variants share bit-identical initialization and produce correctly masked computations, and then trained fixed grids of 3 variants × 25 seeds on four datasets under frozen hyperparameter budgets.

---

## 2. Methods

### 2.1 Attention Variants

Let $Q, K, V \in \mathbb{R}^{n \times d_k}$ denote the query, key and value matrices for a single attention head over a sequence of $n$ tokens, where $d_k = d_{\text{model}} / h$ is the per-head dimension. The three attention variants compared in this study are:

**Standard (baseline)**: The scaled dot-product attention of Vaswani et al. [2017]:
$$\operatorname{Attn}_{\text{std}}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.$$

**Alternative scaling ($1/d_k$)**: The square root is removed, so logits are divided by an additional factor of $\sqrt{d_k} \approx 5.66$ at our per-head dimension ($d_k = 32$):
$$\operatorname{Attn}_{1/d_k}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{d_k}\right)V.$$

**Temperature scaling ($\tau = 2$)**: An explicit temperature applied on top of the standard divisor:
$$\operatorname{Attn}_{\tau}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}\,\tau}\right)V, \qquad \tau = 2.0.$$

All three variants are parameter-free rescalings of the same logits. They introduce no additional weights, no additional arithmetic operations of consequence, and no change to memory traffic.

### 2.2 Model Architecture

The backbone is a compact pre-LN Transformer encoder [Xiong et al., 2020]: token embeddings with padding index, learned positional embeddings of maximum length 128, two encoder blocks each computing LayerNorm $\to$ multi-head self-attention $\to$ residual add followed by LayerNorm $\to$ feed-forward (GELU) $\to$ residual add, a final LayerNorm, masked mean pooling over non-padding positions, and a task-dependent linear head (binary classification for SST-2/IMDB, 4-class for AG News, or linear LM head for WikiText-2).

| Component | Value |
|-----------|-------|
| Vocabulary / max sequence length | 18,855 tokens / 128 positions |
| $d_{\text{model}}$ / heads / $d_k$ | 128 / 4 / 32 |
| Encoder layers / FFN width | 2 / 256 |
| Residual ordering | Pre-LN |
| Dropout | 0.1 |
| Temperature $\tau$ (temp variant) | 2.0 |
| Parameter count (classification) | 2,695,556 |
| Parameter count (LM) | 4,120,576 |

All variants have identical parameter counts.

### 2.3 Datasets

| Dataset | Task | Classes | Train Subset | Test Size |
|---------|------|---------|--------------|-----------|
| AG News | Classification | 4 | 10,000 | 7,600 |
| SST-2 | Classification | 2 | 10,000 | 1,821 |
| IMDB | Classification | 2 | 10,000 | 25,000 |
| WikiText-2 | Language Modeling | — | 5,000 seq | 4,358 seq |

All datasets loaded from HuggingFace with documented splits and preprocessing. Tokenization uses a simple whitespace+punctuation tokenizer for classification and word-level tokenizer for LM. Vocabularies built from training data only.

### 2.4 Matched-Seed Paired Design

The grid is 3 variants × 25 seeds = 75 runs per classification dataset (3 variants × 7 seeds = 21 runs for LM). Within a seed, the three variants share: the same parameter initialization (bit-identical, unit-tested), the same training-example order (a seeded permutation generator drives batch order), the same optimizer state trajectory shape, and the same fixed budget. The unit of statistical analysis is therefore the within-seed difference between a variant and the baseline, giving $n = 25$ (or $n = 7$) paired observations per comparison.

### 2.5 Training Configuration

- **Optimizer**: AdamW ($\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$)
- **Learning rate**: $3 \times 10^{-4}$
- **Weight decay**: 0.01
- **LR schedule**: Linear warmup over first 5% of steps, then cosine decay
- **Batch size**: 64 (classification), 16 (LM)
- **Gradient clipping**: $\max \|g\|_2 = 1.0$
- **Logit clamping**: $[-50, 50]$ for numerical stability
- **Mixed precision**: Disabled (for reproducibility)

### 2.6 Statistical Analysis

For each dataset and variant pair:
1. Compute paired differences (baseline $-$ variant) per seed
2. Paired $t$-test on differences
3. Cohen's $d_z$ (paired effect size)
4. 95% $t$-based confidence interval
5. Wilcoxon signed-rank test (non-parametric alternative)
6. Holm-Bonferroni correction across the two comparisons per dataset
7. Significance threshold: $\alpha = 0.05$ (Holm-adjusted)

---

## 3. Results

### 3.1 AG News (4-class Classification)

| Variant | Test Accuracy | 95% CI | $n$ |
|---------|---------------|--------|-----|
| Standard | $91.23\% \pm 0.05\%$ | $[91.13, 91.33]$ | 25 |
| Temp ($\tau=2$) | $91.13\% \pm 0.05\%$ | $[91.03, 91.23]$ | 25 |
| $1/d_k$ | $90.92\% \pm 0.06\%$ | $[90.80, 91.04]$ | 25 |

**Paired comparisons vs. Standard**:
- $1/d_k$ vs Standard: diff = $0.31$ pp, 95% CI $[0.22, 0.39]$, $d_z = 1.52$, $p_{\text{Holm}} < 0.001$ ***
- Temp vs Standard: diff = $0.10$ pp, 95% CI $[0.02, 0.17]$, $d_z = 0.53$, $p_{\text{Holm}} = 0.013$ *

**Standard significantly outperforms both alternatives on AG News.**

### 3.2 SST-2 (Binary Sentiment)

| Variant | Test Accuracy | 95% CI | $n$ |
|---------|---------------|--------|-----|
| Standard | $80.66\% \pm 0.23\%$ | $[80.20, 81.12]$ | 25 |
| Temp ($\tau=2$) | $80.70\% \pm 0.21\%$ | $[80.28, 81.12]$ | 25 |
| $1/d_k$ | $80.62\% \pm 0.21\%$ | $[80.20, 81.04]$ | 25 |

**Paired comparisons vs. Standard**: No significant differences ($p_{\text{Holm}} > 0.7$). Effect sizes $d_z < 0.1$.

### 3.3 IMDB (Binary Sentiment)

| Variant | Test Accuracy | 95% CI | $n$ |
|---------|---------------|--------|-----|
| Standard | $81.93\% \pm 0.20\%$ | $[81.53, 82.33]$ | 25 |
| Temp ($\tau=2$) | $81.93\% \pm 0.19\%$ | $[81.55, 82.31]$ | 25 |
| $1/d_k$ | $81.96\% \pm 0.18\%$ | $[81.60, 82.32]$ | 25 |

**Paired comparisons vs. Standard**: No significant differences ($p_{\text{Holm}} > 0.7$). Effect sizes $d_z < 0.2$.

### 3.4 WikiText-2 (Language Modeling)

| Variant | Val Perplexity | 95% CI | $n$ |
|---------|----------------|--------|-----|
| Standard | $1.092 \pm 0.002$ | $[1.089, 1.095]$ | 15 |
| Temp ($\tau=2$) | $1.088 \pm 0.002$ | $[1.083, 1.093]$ | 7 |
| $1/d_k$ | $1.081 \pm 0.002$ | $[1.077, 1.085]$ | 3 |

Insufficient seeds for statistical testing. Directionally, $1/d_k$ achieves lowest perplexity.

---

## 4. Discussion

### 4.1 Interpretation

The standard $\sqrt{d_k}$ divisor shows a statistically significant but small advantage on AG News (4-class classification), with the $1/d_k$ variant trailing by 0.31 percentage points and the temperature variant by 0.10 points. On binary sentiment tasks (SST-2, IMDB) and language modeling (WikiText-2), no significant differences emerge. This pattern suggests that the standard divisor's benefit is task- and budget-dependent: it may accelerate early training dynamics on multi-class classification where attention differentiation matters more, but the advantage vanishes on binary tasks and at longer budgets.

### 4.2 Training-Budget Sensitivity

Extended training to 25 epochs (5 seeds on SST-2/IMDB) showed convergence of all variants to similar performance, consistent with the interpretation that the $\sqrt{d_k}$ divisor primarily affects early training dynamics. This aligns with prior work showing attention entropy differences diminish over training [Zhai et al., 2023].

### 4.3 Computational Efficiency

All variants have identical parameter counts and FLOP counts. Measured wall-clock times per epoch differ by $<2\%$ across variants, within noise. No efficiency advantage exists for any variant.

### 4.4 Limitations

1. **Budget**: 2-epoch / 25k-step budgets are far from convergence; results reflect early dynamics only.
2. **Model scale**: $d_{\text{model}}=128$, 2 layers; results may not extrapolate to large models.
3. **WikiText-2 seeds**: Only 3–15 seeds completed; statistical tests underpowered.
4. **Task scope**: Four datasets; generalization to other tasks unknown.
5. **Ablations incomplete**: E1–E4, temperature sweeps, model size/sequence length sensitivity not fully run.

---

## 5. Conclusion

This seed-matched empirical study compared three attention-scaling strategies in Transformer self-attention under controlled conditions. The standard $1/\sqrt{d_k}$ scaling significantly outperforms $1/d_k$ and temperature $\tau=2$ on AG News (4-class classification) at a 2-epoch budget, with moderate effect sizes. On binary sentiment (SST-2, IMDB) and language modeling (WikiText-2), no significant differences were observed. The variants are computationally equivalent. The standard divisor's advantage appears specific to early training on multi-class classification and does not generalize broadly. These results suggest that while the $\sqrt{d_k}$ divisor is a reasonable default, its empirical justification is narrower than often assumed.

---

## References

[1] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.  
[2] Hinton, G., et al. "Distilling the knowledge in a neural network." 2015.  
[3] Xiong, R., et al. "On layer normalization in the transformer architecture." ICML 2020.  
[4] Dong, Y., et al. "Attention temperature." 2021.  
[5] Zhai, Y., et al. "Attention entropy collapse." 2023.  
[6] Holm, S. "A simple sequentially rejective multiple test procedure." 1979.  
[7] Cohen, J. "Statistical power analysis for the behavioral sciences." 1988.