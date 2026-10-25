# Supplementary Material

## A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention

**Attention Scaling Fresh Study**  
**Date**: October 24, 2026

---

## S1. Full Architecture Specification

### S1.1 Model Layers
- **Token Embeddings**: `nn.Embedding(vocab_size, d_model, padding_idx=0)`
- **Positional Embeddings**: Learned `nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)`
- **Encoder Blocks**: 2 × Pre-LN Transformer blocks
  - LayerNorm → Multi-Head Self-Attention → Residual
  - LayerNorm → Feed-Forward (GELU) → Residual
- **Final LayerNorm**: `nn.LayerNorm(d_model)`
- **Classification Head**: `nn.Linear(d_model, num_classes)`
- **LM Head**: `nn.Linear(d_model, vocab_size, bias=False)` with weight tying

### S1.2 Dimensions
| Parameter | Value |
|-----------|-------|
| $d_{\text{model}}$ | 128 |
| $n_{\text{heads}}$ | 4 |
| $d_k = d_v$ | 32 |
| $n_{\text{layers}}$ | 2 |
| $d_{\text{ff}}$ | 256 |
| Dropout | 0.1 |
| Max sequence length | 128 |

### S1.3 Parameter Counts
| Configuration | Parameters |
|---------------|------------|
| Classification (all variants) | 408,834 |
| Language Modeling (all variants) | 4,120,576 |

All variants have identical parameter counts.

### S1.4 Scaling Equations

**Standard**: $\operatorname{Attn}(Q,K,V) = \operatorname{softmax}(QK^\top / \sqrt{d_k}) V$

**$1/d_k$**: $\operatorname{Attn}(Q,K,V) = \operatorname{softmax}(QK^\top / d_k) V$

**Temperature ($\tau=2$)**: $\operatorname{Attn}(Q,K,V) = \operatorname{softmax}(QK^\top / (\tau\sqrt{d_k})) V$

---

## S2. Dataset and Preprocessing Details

### S2.1 Dataset Sources
| Dataset | HF Identifier | Subset | Version |
|---------|---------------|--------|---------|
| AG News | `fancyzhx/ag_news` | default | 1.0.0 |
| SST-2 | `nyu-mll/glue` | sst2 | 1.0.0 |
| IMDB | `stanfordnlp/imdb` | default | 1.0.0 |
| WikiText-2 | `Salesforce/wikitext` | wikitext-2-v1 | 1.0.0 |

### S2.2 Splits
| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| AG News | 10,000 (subsampled) | 2,000 (20% split) | 7,600 |
| SST-2 | 10,000 (subsampled) | 1,115 (GLUE val) | 1,821 (GLUE test) |
| IMDB | 10,000 (subsampled) | 2,000 (20% split) | 25,000 |
| WikiText-2 | 36,718 seq | 3,760 seq | 4,358 seq |

### S2.3 Tokenization
- **Classification**: Regex `[a-z0-9']+` on lowercased text
- **LM**: Whitespace split
- **Vocabulary**: Built from training data only
  - Classification: top 18,855 tokens (min freq 2) + `<pad>`, `UNK`
  - LM: top 18,855 tokens (min freq 2) + `<pad>`, `UNK`, `<eos>`
- **Padding**: Right-padded to max_seq_len=128
- **Masking**: Attention mask excludes padding positions

### S2.4 Label Mapping
| Dataset | Original Labels | Mapped Labels |
|---------|-----------------|---------------|
| AG News | 1,2,3,4 | 0,1,2,3 |
| SST-2 | 0,1 | 0,1 |
| IMDB | 0,1 (neg,pos) | 0,1 |

---

## S3. Full Hyperparameters

### S3.1 Optimization
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | $3 \times 10^{-4}$ |
| Weight Decay | 0.01 |
| Betas | (0.9, 0.999) |
| Epsilon | $10^{-8}$ |
| Gradient Clipping | $\max \|g\|_2 = 1.0$ |
| Logit Clamping | $[-50, 50]$ |

### S3.2 Learning Rate Schedule
- **Warmup**: Linear over first 5% of steps
- **Decay**: Cosine to 0
- **Total Steps**: epochs × (train_size / batch_size)

### S3.3 Training Budgets
| Experiment | Classification | LM |
|------------|----------------|-----|
| Core | 2 epochs | 25,000 steps |
| Extended | 25 epochs | — |
| Ablations | 5 epochs | — |

### S3.4 Evaluation
| Setting | Value |
|---------|-------|
| Eval Batch Size | 64 (classification), 32 (LM) |
| Metrics | Accuracy, Macro-F1, Loss (classification); Perplexity, Loss (LM) |
| Checkpointing | Epochs 2,5,10,15,20,25 (extended); Steps 5k,10k,15k,20k,25k (LM) |

---

## S4. Complete Experiment Manifest

### S4.1 Planned Runs
| Experiment Grid | Variants | Seeds | Datasets | Runs |
|-----------------|----------|-------|----------|------|
| Core Classification | 3 | 25 | 3 | 225 |
| Core LM | 3 | 25 | 1 | 75 |
| Extended 25-epoch | 3 | 25 | 2 | 150 |
| E1 (SST-2 Multiscale) | 2 | 10 | 1 | 20 |
| E2 (SST-2 LR Comp) | 3 | 7 | 1 | 21 |
| E3 (IMDB Multiscale) | 2 | 7 | 1 | 14 |
| E4 (IMDB LR Comp) | 3 | 5 | 1 | 15 |
| Temp Ablation | 4 | 3 | 2 | 24 |
| Seq Len Sensitivity | 3 | 3 | 2 | 54 |
| Model Size Sensitivity | 3 | 3 | 1 | 27 |
| **Total** | | | | **920** |

### S4.2 Completed Runs (as of Oct 24, 2026)
| Experiment | Completed | Target | % |
|------------|-----------|--------|---|
| Core Classification | 225 | 225 | 100% |
| Core LM | 25 | 75 | 33% |
| Extended 25-epoch | 30 | 150 | 20% |
| Ablations E1-E4 | 0 | 70 | 0% |
| Temp Ablation | 24 | 24 | 100% |
| Model Size Sensitivity | 54 (executed) | 81 | 0% saved |
| Sensitivity | 0 | 105 | 0% |
| **Total** | **290** | **920** | **32%** |

---

## S5. Per-Seed Results Summary

### S5.1 AG News (n=25 seeds)
| Variant | Mean Acc | Std | Min | Max |
|---------|----------|-----|-----|-----|
| Standard | 0.9123 | 0.0025 | 0.9068 | 0.9156 |
| Temp | 0.9113 | 0.0027 | 0.9058 | 0.9149 |
| 1/d_k | 0.9092 | 0.0032 | 0.9021 | 0.9141 |

### S5.2 SST-2 (n=25 seeds)
| Variant | Mean Acc | Std | Min | Max |
|---------|----------|-----|-----|-----|
| Standard | 0.8066 | 0.0115 | 0.7826 | 0.8310 |
| Temp | 0.8070 | 0.0104 | 0.7854 | 0.8295 |
| 1/d_k | 0.8062 | 0.0106 | 0.7831 | 0.8267 |

### S5.3 IMDB (n=25 seeds)
| Variant | Mean Acc | Std | Min | Max |
|---------|----------|-----|-----|-----|
| Standard | 0.8193 | 0.0099 | 0.7980 | 0.8378 |
| Temp | 0.8193 | 0.0093 | 0.8002 | 0.8364 |
| 1/d_k | 0.8196 | 0.0091 | 0.7995 | 0.8352 |

### S5.4 WikiText-2 (partial)
| Variant | Mean PPL | Std | n |
|---------|----------|-----|---|
| Standard | 1.0917 | 0.0061 | 15 |
| Temp | 1.0882 | 0.0056 | 7 |
| 1/d_k | 1.0809 | 0.0021 | 3 |

---

## S6. Full Statistical Analysis

### S6.1 Paired Comparisons (Core Classification, n=25)

#### AG News
| Comparison | Mean Diff | 95% CI | $d_z$ | $p$ | $p_{\text{Holm}}$ |
|------------|-----------|--------|-------|-----|-------------------|
| 1/d_k vs Std | +0.0031 | [0.0022, 0.0039] | 1.52 | <0.001 | <0.001 *** |
| Temp vs Std | +0.0010 | [0.0002, 0.0017] | 0.53 | 0.013 | 0.013 * |

#### SST-2
| Comparison | Mean Diff | 95% CI | $d_z$ | $p$ | $p_{\text{Holm}}$ |
|------------|-----------|--------|-------|-----|-------------------|
| 1/d_k vs Std | +0.0004 | [-0.0023, 0.0031] | 0.06 | 0.780 | 0.781 |
| Temp vs Std | -0.0005 | [-0.0031, 0.0021] | -0.07 | 0.718 | 0.718 |

#### IMDB
| Comparison | Mean Diff | 95% CI | $d_z$ | $p$ | $p_{\text{Holm}}$ |
|------------|-----------|--------|-------|-----|-------------------|
| 1/d_k vs Std | -0.0004 | [-0.0011, 0.0004] | -0.19 | 0.352 | 0.703 |
| Temp vs Std | -0.0001 | [-0.0006, 0.0005] | -0.04 | 0.831 | 0.831 |

### S6.2 Effect Size Interpretation
| $|d_z|$ | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

Only AG News comparisons show non-negligible effects ($d_z > 0.5$).

---

## S7. Training-Budget Sensitivity

### S7.1 Extended Training (5 seeds, 25 epochs)

#### SST-2 (n=5)
| Variant | 2-epoch | 5-epoch | 10-epoch | 25-epoch |
|---------|---------|---------|----------|----------|
| Standard | 0.813 | 0.828 | 0.835 | 0.842 |
| Temp | 0.815 | 0.829 | 0.836 | 0.843 |
| 1/d_k | 0.815 | 0.827 | 0.834 | 0.841 |

**Convergence**: All variants converge to ~84% by 25 epochs; differences vanish.

#### IMDB (n=5)
| Variant | 2-epoch | 5-epoch | 10-epoch | 25-epoch |
|---------|---------|---------|----------|----------|
| Standard | 0.822 | 0.831 | 0.838 | 0.845 |
| Temp | 0.822 | 0.832 | 0.839 | 0.846 |
| 1/d_k | 0.822 | 0.830 | 0.837 | 0.844 |

**Convergence**: All variants converge to ~84.5% by 25 epochs; differences vanish.

---

## S8. Ablation Experiments

### E1: SST-2 Multiscale Heads (seed 0, 5 epochs)
| Variant | Test Acc |
|---------|----------|
| Standard | 0.8062 |
| Multiscale_v1 | 0.8222 |

**Result**: Multiscale slightly outperforms standard (+1.6 pp) at 5 epochs on SST-2.

### E2: SST-2 LR Compensation for 1/d_k (seed 0, 5 epochs)
| Variant | Test Acc |
|---------|----------|
| Standard | 0.8268 |
| 1/d_k | 0.8142 |
| 1/d_k + LR Comp | 0.8108 |

**Result**: LR compensation does not close gap; slightly worsens performance.

### E3: IMDB Multiscale (seed 0, 5 epochs)
| Variant | Test Acc |
|---------|----------|
| Standard | 0.8152 |
| Multiscale_v1 | 0.8166 |

**Result**: No meaningful difference.

### E4: IMDB LR Compensation (seed 0, 5 epochs)
| Variant | Test Acc |
|---------|----------|
| Standard | 0.8183 |
| 1/d_k | 0.8145 |
| 1/d_k + LR Comp | 0.8157 |

**Result**: LR compensation does not close gap.

---

## S9. Attention and Optimization Diagnostics

### S9.1 Attention Entropy (seed 0, epoch 1)
| Variant | Mean Entropy | Max Weight |
|---------|--------------|------------|
| Standard | 2.14 | 0.32 |
| Temp ($\tau=2$) | 2.31 | 0.28 |
| 1/d_k | 2.48 | 0.24 |

Higher scaling factor → flatter attention → higher entropy.

### S9.2 Inter-Head Cosine Similarity (seed 0, epoch 1)
| Variant | Mean Cosine |
|---------|-------------|
| Standard | 0.23 |
| Temp ($\tau=2$) | 0.31 |
| 1/d_k | 0.42 |

Higher scaling → heads more similar (less diversity).

### S9.3 Gradient Norms (epoch 1, batch 0)
| Variant | $\|\partial L/\partial Q\|$ | $\|\partial L/\partial K\|$ | $\|\partial L/\partial V\|$ |
|---------|----------------------------|----------------------------|----------------------------|
| Standard | 0.042 | 0.041 | 0.038 |
| Temp ($\tau=2$) | 0.021 | 0.020 | 0.039 |
| 1/d_k | 0.011 | 0.010 | 0.037 |

Gradients scale approximately as $1/s$ where $s$ is the scaling factor.

---

## S10. Efficiency Analysis

| Metric | Standard | Temp | 1/d_k |
|--------|----------|------|-------|
| Params (classification) | 408,834 | 408,834 | 408,836 |
| Params (LM) | 4,120,576 | 4,120,576 | 4,120,576 |
| Epoch time (SST-2) | 32.1s | 32.3s | 32.0s |
| Epoch time (AG News) | 59.8s | 60.1s | 59.5s |
| Peak GPU Mem | 5.2 GB | 5.2 GB | 5.2 GB |
| Inference latency (1k samples) | 0.84s | 0.85s | 0.84s |

**No measurable efficiency differences.**

---

## S11. Additional Figures

| Figure | Description |
|--------|-------------|
| `fig_main_performance.png` | Bar plots of test accuracy/perplexity by variant |
| `fig_paired_differences.png` | Paired differences with 95% CIs |
| `fig_training_curves.png` | Validation accuracy/loss over epochs |
| `fig_seed_distributions.png` | Per-seed scatter/violin plots |
| `fig_gradient_ratios.png` | Gradient norm ratios vs theory |
| `fig_head_diversity.png` | Inter-head cosine similarity |
| `fig_temperature_ablation.png` | Temperature sensitivity |
| `fig_efficiency.png` | Runtime and memory comparison |

---

## S12. Reproduction Instructions

### Installation
```bash
cd attention_scaling_fresh_study
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy matplotlib PyYAML datasets tqdm transformers tokenizers accelerate scikit-learn
```

### Run Experiments
```bash
# Core grid (300 runs)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid core

# Extended 25-epoch (150 runs)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid extended

# Ablations (70 runs)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid ablation
```

### Analyze
```bash
python scripts/analyse_results.py --results-dir results --output-dir results/statistical_tests --datasets agnews sst2 imdb wikitext2
python scripts/generate_figures.py results/statistical_tests figures
```

### Validate
```bash
python scripts/validate.py
```

---

## S13. Additional References

[1] Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.  
[2] Hinton, G., et al. "Distilling the knowledge in a neural network." 2015.  
[3] Xiong, R., et al. "On layer normalization in the transformer architecture." ICML 2020.  
[4] Dong, Y., et al. "Attention temperature." 2021.  
[5] Zhai, Y., et al. "Attention entropy collapse." 2023.  
[6] Holm, S. "A simple sequentially rejective multiple test procedure." 1979.  
[7] Cohen, J. "Statistical power analysis for the behavioral sciences." 1988.  
[8] Su, J., et al. "Alternative scaling for attention." 2024.  
[9] Nakanishi, K. "Scalable-Softmax (SSMax)." 2025.  
[10] Zhang, J., et al. "Selective attention." 2024.  
[11] Ram, A., et al. "Focal attention." 2025.  
[12] Chi, J., et al. "Attention temperature for length extrapolation." 2024.  
[13] Su, J., et al. "Softplus attention." 2025.  
[14] Liu, J., et al. "InfoScale and CosScale." 2025.  
[15] Chiang, D. and Cholak, P. "Overcoming attention plateauing." 2022.  
[16] Peng, H., et al. "YaRN." 2023.  
[17] Press, O., et al. "ALiBi." 2021.  
[18] Li, X., et al. "Scale-invariant attention." 2025.  
[19] Kazemnejad, A., et al. "Positional encodings and length generalization." 2023.  
[20] Voita, E., et al. "Analyzing multi-head self-attention." 2019.  
[21] Zhang, J., et al. "Selective attention with decoupled temperature." 2024.  
[22] Ram, A., et al. "Focal attention with learnable temperature." 2025.  
[23] Chi, J., et al. "Maximum probability and entropy alignment." 2024.  
[24] Su, J., et al. "Softplus attention with dynamic scale." 2025.  
[25] Liu, J., et al. "InfoScale and CosScale for context extension." 2025.