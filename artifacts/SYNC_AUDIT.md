# SYNC AUDIT: Manuscript Numbers vs Frozen Results

**Study**: A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention  
**Audit Date**: October 24, 2026  
**Frozen Results Source**: `attention_scaling_fresh_study/results/statistical_tests/summary_table.json` (250 completed runs, 25 seeds per classification dataset)

---

## Phase 1: Number Mapping Audit

### AG News (4-class, n=25 seeds, 2 epochs)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Abstract | "74.36% (±0.70)" | 91.23% ± 0.05% (n=25) | **REPLACE** with 91.23% ± 0.05% |
| Abstract | "72.73% (±1.00)" for 1/d_k | 90.92% ± 0.06% | **REPLACE** |
| Abstract | "73.42% (±0.95)" for temp τ=2 | 91.13% ± 0.05% | **REPLACE** |
| Abstract | "−1.62 pp (CI [-2.09,-1.15], dz=-4.31, p=0.0013)" | +0.31 pp (CI [0.22,0.39], dz=1.52, p<0.001) | **REPLACE** (sign flipped: old had variant worse, new has standard better) |
| Abstract | "−0.94 pp (CI [-1.27,-0.60], dz=-3.46, p=0.0015)" | +0.10 pp (CI [0.02,0.17], dz=0.53, p=0.013) | **REPLACE** (sign flipped) |
| Sec 3.1 / Tab 3.1 | 74.36% ± 0.70 | 91.23% ± 0.05% | REPLACE |
| Sec 3.1 / Tab 3.1 | 72.73% ± 1.00 | 90.92% ± 0.06% | REPLACE |
| Sec 3.1 / Tab 3.1 | 73.42% ± 0.95 | 91.13% ± 0.05% | REPLACE |
| Sec 3.1 | "-1.62 pp, CI [-2.09,-1.15], dz=-4.31" | +0.31 pp, CI [0.22,0.39], dz=1.52 | REPLACE (sign/interpretation) |
| Sec 3.1 | "-0.94 pp, CI [-1.27,-0.60], dz=-3.46" | +0.10 pp, CI [0.02,0.17], dz=0.53 | REPLACE |
| Table 1 (arch) | "Parameter count: 2,695,556" | 408,834 (per variant) | REPLACE |

### SST-2 (n=25 seeds, 2 epochs)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Abstract | Not explicitly stated | 80.66% ± 0.23% (std) | ADD to abstract |
| Sec 3.2 / Tab 3.2 | 75.23% ± 1.75% | 80.66% ± 0.23% | REPLACE |
| Sec 3.2 / Tab 3.2 | 74.75% ± 2.99% | 80.70% ± 0.21% | REPLACE |
| Sec 3.2 / Tab 3.2 | 75.05% ± 2.45% | 80.62% ± 0.21% | REPLACE |
| Sec 3.2 | "-0.18 pp, CI [-1.18,+0.82], dz=-0.23" | +0.04 pp, CI [-0.23,+0.31], dz=0.06 | REPLACE |
| Sec 3.2 | "-0.48 pp, CI [-2.20,+1.24], dz=-0.35" | -0.05 pp, CI [-0.31,+0.21], dz=-0.07 | REPLACE |
| Table paired | -0.0018, CI [-0.0118,+0.0082], dz=-0.23, p=0.6372 | +0.0004, CI [-0.0023,+0.0031], dz=0.06, p=0.78 | REPLACE |
| Sec 3.2 | "-0.0048, CI [-0.0220,+0.0124], dz=-0.35" | -0.0005, CI [-0.0031,+0.0021], dz=-0.07 | REPLACE |

### IMDB (n=25)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Sec 3.3 / Tab 3.3 | 79.27% ± 0.69% | 81.93% ± 0.20% | REPLACE |
| Sec 3.3 / Tab 3.3 | 79.20% ± 0.75% | 81.93% ± 0.19% | REPLACE |
| Sec 3.3 / Tab 3.3 | 79.25% ± 0.81% | 81.96% ± 0.18% | REPLACE |
| Sec 3.3 | "-0.02 pp, CI [-0.24,+0.20], dz=-0.11" | -0.04 pp, CI [-0.11,+0.04], dz=-0.19 | REPLACE |
| Sec 3.3 | "-0.07 pp, CI [-0.16,+0.02], dz=-0.99" | -0.01 pp, CI [-0.06,+0.05], dz=-0.04 | REPLACE |

### WikiText-2 (LM)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Abstract | "91.69 ± 3.35 vs 133.26 ± 6.28, 136.45 ± 5.93" | n=3-15 (insufficient) | **REMOVE formal stats** / add caveat |
| Sec 3.4 / Tab 3.4 | 91.69 ± 3.35 | 1.092 ± 0.002 (n=15) | **REPLACE** (units: perplexity) |
| Sec 3.4 / Tab 3.4 | 133.25 ± 6.28 | 1.088 ± 0.002 (n=7) | REPLACE |
| Sec 3.4 / Tab 3.4 | 136.45 ± 5.93 | 1.081 ± 0.002 (n=3) | REPLACE |
| Sec 3.4 | "paired p < 0.0001" | No paired stats possible (n=3-15) | REMOVE |
| Table paired | +44.76, CI [+41.28,+48.24], dz=15.96 | n=3, no valid paired test | REMOVE |

### Temperature Ablation (τ ∈ {0.5, 1.0, 1.5, 2.0})

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Abstract | Not mentioned | τ=0.5 best on SST-2/IMDB | ADD to abstract |
| Temp ablation section | Not in PAPER_FINAL.md | Add new section | ADD new subsection |

### E1-E4 Ablations

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Sec 4.3 (E1-E4) | Full results reported | **NOT completed in 320-run grid** | MOVE to Limitations / Future Work |
| E1 multiscale | "mean_diff = -0.0050, CI [-0.0104,+0.0004]" | Not in 320-run grid | REMOVE / move to Future Work |
| E2 dk_lrcomp | "gap_closed ≈ -38.0" | Not in grid | REMOVE |
| E3 IMDB multiscale | "mean_diff = -0.0032" | Not in grid | REMOVE |
| E4 IMDB LR comp | "gap_closed = -6.82" | Not in grid | REMOVE |

### Extended Budget (25-epoch)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Abstract | Not mentioned | "gaps become negligible (<0.5%)" | QUALIFY as preliminary (5 seeds only) |
| Sec 4.2 | "gaps become negligible (<0.5%)" | Preliminary, 5 seeds only | QUALIFY as preliminary |

### Extended Budget 10-epoch

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Supp | "gaps become negligible (<0.5%)" | Partial (5 seeds) | QUALIFY as preliminary |

### Extended Budget 5-epoch (E1-E4)

| Location | Old Text/Number | Frozen Value | New Action |
|----------|-----------------|--------------|------------|
| Sec 4.3 | Full results reported | **NOT completed** | MOVE to Future Work |

---

## Key Discrepancy Summary

| Category | Old Claim | Frozen Reality | Verdict |
|----------|-----------|----------------|---------|
| AG News std accuracy | 74.36% | 91.23% | **MAJOR REPLACE** |
| AG News 1/d_k accuracy | 72.73% | 90.92% | MAJOR REPLACE |
| AG News temp accuracy | 73.42% | 91.13% | MAJOR REPLACE |
| AG News std vs 1/d_k | -1.62 pp (std worse) | +0.31 pp (std better) | **SIGN FLIP** |
| AG News std vs temp | -0.94 pp (std worse) | +0.10 pp (std better) | **SIGN FLIP** |
| SST-2 std accuracy | 75.23% | 80.66% | MAJOR REPLACE |
| SST-2 1/d_k | 72.73% | 80.62% | MAJOR REPLACE |
| IMDB std | 79.27% | 81.93% | MAJOR REPLACE |
| WikiText-2 ppl | 91.69 / 133 / 136 | 1.09 / 1.09 / 1.08 (n<15) | REMOVE FORMAL STATS |
| WikiText-2 paired | p < 0.0001 | n=3-15 insufficient | REMOVE |
| E1-E4 completed | Full results | NOT DONE | MOVE TO LIMITATIONS |
| Extended 25-epoch | 25 seeds | 5 seeds only | QUALIFY |

---

## Required Changes Checklist

- [ ] Abstract: Replace all accuracy numbers with frozen values (91.23%, 80.66%, 81.93%)
- [ ] Abstract: Fix sign on AG News differences (standard BETTER, not worse)
- [ ] Abstract: Add τ=0.5 best on SST-2/IMDB
- [ ] Abstract: Add WikiText-2 caveat (insufficient seeds)
- [ ] Abstract: Remove E1-E4 completion claim
- [ ] Sec 3.1 AG News: Replace all numbers with frozen (91.23%, 91.13%, 90.92%)
- [ ] Sec 3.1 AG News: Fix paired comparison signs and p-values
- [ ] Sec 3.2 SST-2: Replace all numbers (80.66%, 80.70%, 80.62%)
- [ ] Sec 3.2 SST-2: Update paired comparisons
- [ ] Sec 3.3 IMDB: Replace all numbers (81.93%, 81.93%, 81.96%)
- [ ] Sec 3.3 IMDB: Update paired comparisons
- [ ] Sec 3.4 WikiText-2: Remove formal paired stats, add caveat
- [ ] Sec 4.2: Qualify extended budget as preliminary (5 seeds)
- [ ] Sec 4.3/4.4: Move E1-E4 to Limitations/Future Work
- [ ] Limitations: Add incomplete WikiText-2, E1-E4, model size, seq len
- [ ] Tables: Regenerate with frozen numbers
- [ ] Figures: Regenerate or mark as pilot-only
- [ ] Supplementary: Only include completed arms

---

## Verification Sources

- Frozen numbers from: `attention_scaling_fresh_study/results/statistical_tests/summary_table.json`
- Per-variant stats: `*_per_variant_stats.json` in each dataset folder
- Paired comparisons: `*_paired_comparisons.json` in each dataset folder
- Temperature ablation results: `results/results/per_seed/` files with `temp_tau0.5`, `temp_tau1.0`, etc.
- Temperature ablation seeds: 3 seeds (1001-1003) per temp per dataset