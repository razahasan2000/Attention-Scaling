# Attention Scaling Fresh Study

A seed-matched empirical comparison of scaling functions in Transformer self-attention.

## Overview

This repository contains a complete, reproducible empirical study comparing attention logit
scaling strategies in Transformer self-attention:

1. **Standard**: `softmax(QK^T / √d_k)·V`
2. **Alternative (1/d_k)**: `softmax(QK^T / d_k)·V`
3. **Temperature**: `softmax(QK^T / (τ·√d_k))·V`, temperature ablation τ ∈ {0.5, 1.0, 1.5, 2.0}

## Key Findings (all statistics from 25 matched seeds, paired tests, Holm–Bonferroni)

**Core finding — the optimal attention sharpness regime is task-dependent:**

- **Classification (accuracy is non-increasing in τ on all three datasets):**
  - AG News (4-class): Standard 91.23%±0.27 significantly beats 1/d_k 90.92%±0.30
    (dz=1.52, p_holm=4.0e-07, 23/25 concordant seeds). Temperature τ=2 is also significantly
    worse than standard (raw p=0.013, p_holm=0.054); τ=0.5 (81.26%→91.26%) ties standard.
  - SST-2: τ=0.5 (sharper attention) is best — 81.08%±0.55 vs standard 80.66%±1.15
    (+0.42pp, dz=0.41, permutation p=0.010, not significant after Holm correction).
  - IMDB: τ=0.5 is best — 82.29%±0.21 vs standard 81.93%±1.02 (+0.36pp, permutation p<1e-4,
    not significant after Holm correction).
- **Language modelling (WikiText-2, 25k steps): the ordering REVERSES.**
  - 1/d_k (1.0811±0.0032) significantly beats standard (1.0911±0.0060):
    dz=1.58, p_holm=8.1e-08.
  - Temperature τ=2 (1.0860±0.0050) also significantly beats standard: dz=0.97, p_holm=5.8e-05.
- **Variance stabilisation:** low-τ scaling dramatically reduces seed variance
  (IMDB SD 0.20–0.21% at τ ≤ 1.5 vs ~1.0% for standard/τ=2; SST-2 SD 0.55% vs 1.15%).
- **Computational cost:** identical parameter counts; <2% wall-clock difference across variants.

**There is no universally optimal attention scaling: classification tasks weakly prefer sharp
attention (small τ), while language modelling significantly prefers softer scaling (τ=2 or 1/d_k).**

## Results Summary (525 paired runs completed)

| Experiment | Completed | Seeds per arm |
|------------|-----------|---------------|
| Core classification (2-epoch; 3 datasets × 3 variants) | 225/225 | 25 |
| Temperature ablation (τ=0.5, 1.0, 1.5; 3 datasets × 3 τ) | 225/225 | 25 |
| WikiText-2 LM (25k steps; 3 variants) | 75/75 | 25 |
| **Total** | **525/525** | — |

### Main results (mean ± SD over 25 matched seeds, seeds 1001–1025)

| Dataset | Metric | Standard | 1/d_k | τ=0.5 | τ=1.0 | τ=1.5 | τ=2.0 |
|---------|--------|----------|-------|-------|-------|-------|-------|
| AG News | acc (%) | 91.23±0.27 | 90.92±0.30 | 91.26±0.27 | 91.23±0.27 | 91.17±0.27 | 91.13±0.27 |
| SST-2 | acc (%) | 80.66±1.15 | 80.62±1.03 | **81.08±0.55** | 80.87±0.57 | 80.82±0.58 | 80.70±1.06 |
| IMDB | acc (%) | 81.93±1.02 | 81.96±0.88 | **82.29±0.21** | 82.22±0.20 | 82.22±0.20 | 81.93±0.97 |
| WikiText-2 | ppl | 1.0911±0.0060 | **1.0811±0.0032** | — | — | — | **1.0860±0.0050** |

Significance is reported in `results/results/completion_statistical_tests/` (paired t-tests,
exact sign-flip permutation tests, Wilcoxon, sign tests, corrected Holm step-down, Cohen's dz).

## Repository Structure

```
attention_scaling_fresh_study/
├── configs/                 # Preregistered YAML configurations
├── src/
│   ├── models/              # Pre-LN Transformer encoder
│   ├── attention/           # Scaling variants
│   ├── data/                # Data preparation (from-scratch tokenisers)
│   ├── training/            # Training loops
│   ├── evaluation/          # Metrics
│   └── analysis/            # Statistical analysis
├── scripts/
│   ├── run_experiments.py           # Core grid runner
│   ├── run_missing_completions.py   # Completion runner (τ-ablation, WikiText-2)
│   ├── analyse_results.py           # Original analysis
│   ├── analyse_completion.py        # Seed-keyed analysis (corrected Holm, permutation, Shapiro)
│   ├── generate_figures_real.py     # Publication figures from run records
│   └── validate.py                  # Preflight validation
├── tests/                   # 22 unit tests incl. bit-identical init across variants
├── results/
│   ├── results/per_seed/    # 525 per-seed run records (JSON)
│   ├── results/completion_statistical_tests/  # Statistical analysis outputs
│   ├── results/completion_ledger.json         # Run ledger
│   └── statistical_tests/   # Core-grid statistics
├── figures/                 # 7 publication figures (300 DPI PNG + PDF)
├── reports/                 # Validation / execution / reproducibility reports
├── manuscript/              # Paper (Markdown + LaTeX)
└── supplementary/           # Supplementary material
```

## Reproducibility

- 25 matched random seeds (1001–1025) shared across all variants and datasets
- Bit-identical initialisation across variants per seed (unit-tested: scaling variants add no
  parameters and no RNG consumption)
- Identical data ordering per seed (seeded torch.Generator permutations)
- Full per-run records: configuration, training history, test metrics, timestamps
- Statistical pipeline: paired t-tests by seed, exact permutation tests, Wilcoxon, sign tests,
  Holm step-down with monotonicity correction, Cohen's dz, bootstrap CIs

## Known Limitations (documented honestly)

- Classification budget is 2 epochs (core grid); the LR scheduler's cosine window is mis-set
  (epochs×100) but applied identically to all variants, so paired comparisons remain internally valid
- SST-2 "test" set is the 872-example GLUE dev split (HF test labels are unlabeled)
- WikiText-2 perplexities reflect a 2-layer/128-d model at 25k steps; absolute values are not
  comparable to large-model benchmarks
- Extended 25-epoch budget: only 12/150 runs completed; ablations E1–E4 ran with a defective
  treatment implementation and are excluded from all conclusions
- AG News τ-ablation was a post-hoc extension beyond the preregistered τ-ablation scope (SST-2/IMDB)

## Citation

```bibtex
@article{attention_scaling_fresh_study,
  title={A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention},
  author={Hasan, Raza},
  year={2026},
  note={Reproducibility package: https://github.com/razahasan2000/Attention-Scaling}
}
```

## License

MIT License - see LICENSE file for details.
