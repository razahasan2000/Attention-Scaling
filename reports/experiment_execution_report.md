# Experiment Execution Report

## Attention Scaling Fresh Study

**Date**: October 23, 2026  
**Execution Period**: October 22-23, 2026

---

## 1. Execution Summary

### Hardware
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8.6 GB VRAM)
- **CPU**: Multi-core (Python multiprocessing not used)
- **RAM**: System memory (exact amount not logged)

### Software Environment
- **Python**: 3.14.5
- **PyTorch**: 2.14.1+cu126
- **CUDA**: 12.6
- **Key Packages**: 
  - torch 2.14.1+cu126
  - numpy 2.4.6
  - scipy 1.17.1
  - datasets 5.1.0
  - transformers 5.12.1
  - matplotlib 3.11.0
  - scipy 1.17.1 (for stats)

### Execution Timeline
| Date | Phase | Duration |
|------|-------|----------|
| Oct 22 | Environment setup, config creation | ~2 hours |
| Oct 22 | Unit tests, validation | ~30 min |
| Oct 22-23 | Core grid execution (classification) | ~8 hours |
| Oct 22-23 | WikiText-2 LM training | ~6 hours |
| Oct 23 | Analysis, figures, validation | ~2 hours |
| **Total** | | **~16.5 hours** |

---

## 2. Runs Executed

| Experiment | Seeds | Variants | Datasets | Runs | Status |
|------------|-------|----------|----------|------|--------|
| Core Classification (2-epoch) | 25 | 3 | 3 | 225 | ✅ Complete |
| Core LM (25k steps) | 7 | 3 | 1 | 25 | ⚠️ Partial |
| Extended (25-epoch) | 5 | 3 | 2 | 30 | ⚠️ Partial |
| Ablations (E1-E4) | 1 | 2-3 | 2 | 10 | ⚠️ Partial |
| **Total** | | | | **250** | **44% of planned** |

### Detailed Run Breakdown

| Dataset | Variant | Seeds Completed | Target | Completion |
|---------|---------|-----------------|--------|------------|
| AG News | standard | 25 | 25 | 100% |
| AG News | temp | 25 | 25 | 100% |
| AG News | dk | 25 | 25 | 100% |
| SST-2 | standard | 25 | 25 | 100% |
| SST-2 | temp | 25 | 25 | 100% |
| SST-2 | dk | 25 | 25 | 100% |
| IMDB | standard | 25 | 25 | 100% |
| IMDB | temp | 25 | 25 | 100% |
| IMDB | dk | 25 | 25 | 100% |
| WikiText-2 | standard | 15 | 25 | 60% |
| WikiText-2 | temp | 7 | 25 | 28% |
| WikiText-2 | dk | 3 | 25 | 12% |

---

## 3. Resource Usage

### GPU Memory
- **Peak**: ~6.2 GB / 8.6 GB (72%)
- **Typical**: ~4.5 GB during classification training
- **LM Training**: ~5.5 GB

### Training Time (per run)
| Dataset | Epochs/Steps | Time per Run |
|---------|--------------|--------------|
| AG News | 2 epochs | ~60 sec |
| SST-2 | 2 epochs | ~32 sec |
| IMDB | 2 epochs | ~12 sec |
| WikiText-2 | 25k steps | ~180 sec |

### Total Compute
- **Classification runs**: 225 × ~35 sec avg = ~2.2 hours
- **LM runs**: 25 × ~180 sec = ~1.25 hours
- **Extended runs**: 30 × ~5 min = ~2.5 hours
- **Ablations**: 10 × ~5 min = ~0.8 hours
- **Overhead (data prep, analysis, etc.)**: ~2 hours
- **Total GPU time**: ~8.5 hours

---

## 4. Data Integrity

### Run Records
- **Format**: JSON (one file per run)
- **Location**: `results/per_seed/`
- **Fields**: run_id, dataset, variant, seed, config, training_history, test_metrics, checkpoint_history, status, timestamp

### Checkpoints
- **Format**: PyTorch .pt files
- **Location**: `checkpoints/`
- **Frequency**: Per config (epoch 2 for core, epochs 2,5,10,15,20,25 for extended)

### Aggregated Results
- `results/aggregated/all_results.json`: All run records combined
- `results/statistical_tests/*/`: Per-dataset analysis
- `results/statistical_tests/summary_table.json`: Cross-dataset summary

---

## 5. Issues Encountered

| Issue | Impact | Resolution |
|-------|--------|------------|
| WikiText-2 HF namespace | Initial load failed | Changed to `Salesforce/wikitext` |
| AG News HF namespace | Initial load failed | Changed to `fancyzhx/ag_news` |
| IMDB HF namespace | Initial load failed | Changed to `stanfordnlp/imdb` |
| NaN losses in LM training | Training instability | Added gradient clipping (max_norm=1.0) and logit clamping (-50,50) |
| Multiscale variant not recognized | Config error | Updated ScaledDotProductAttention to accept multiscale_v1/v2 |
| LR compensation CUDA assert | Index out of bounds | Not fully resolved; used standard model with split_qkv=False |
| WikiText-2 training slow | 25k steps takes ~3 min | Reduced eval frequency for testing |

---

## 6. Recommendations for Completion

1. **Priority 1**: Complete WikiText-2 seeds (need 18 standard, 18 temp, 22 dk)
2. **Priority 2**: Extended 25-epoch for all 25 seeds on SST-2/IMDB
3. **Priority 3**: Ablation experiments E1-E4 (full seed ranges)
3. **Priority 4**: Sensitivity analyses (temperature, model size, seq length)
4. **Priority 5**: Manuscript PDF compilation and supplementary document

Estimated remaining compute: ~200 GPU hours