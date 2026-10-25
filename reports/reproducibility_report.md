# Reproducibility Report

## Attention Scaling Fresh Study

**Study**: A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention  
**Date**: October 23, 2026

---

## 1. Environment Specification

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥6 GB VRAM (tested on RTX 4070 8.6 GB)
- **CPU**: x86_64, multi-core recommended
- **RAM**: ≥16 GB system RAM
- **Storage**: ≥10 GB free space

### Software Dependencies
```
torch>=2.14.0+cu126
numpy>=2.4.0
scipy>=1.17.0
matplotlib>=3.11.0
PyYAML>=6.0.3
datasets>=5.1.0
tqdm>=4.68.3
transformers>=5.12.0
tokenizers>=0.22.0
accelerate>=1.15.0
scikit-learn>=1.9.0
```

### Installation
```bash
cd attention_scaling_fresh_study
pip install -r requirements.txt
# Or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy matplotlib PyYAML datasets tqdm transformers tokenizers accelerate scikit-learn
```

---

## 2. Configuration

All experimental parameters are defined in `configs/`:

| File | Purpose |
|------|---------|
| `model_config.yaml` | Model architecture (d_model=128, n_heads=4, n_layers=2, etc.) |
| `experiment_config.yaml` | Datasets, optimizer, training budgets, hardware settings |
| `seed_config.yaml` | Seed schedules (25 primary seeds 1001-1025, ablation seeds) |
| `training_budget_config.yaml` | Epoch/step budgets, checkpoint schedules |
| `statistical_config.yaml` | Statistical analysis parameters |
| `dataset_configs.yaml` | Dataset-specific HF configs and preprocessing |

### Key Experimental Parameters

| Parameter | Value |
|-----------|-------|
| Model | 2-layer pre-LN Transformer encoder |
| d_model | 128 |
| n_heads | 4 (d_k = 32) |
| n_layers | 2 |
| d_ff | 256 |
| dropout | 0.1 |
| max_seq_len | 128 |
| vocab_size | ~18,855 (dataset-dependent) |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| LR Schedule | Cosine with 5% warmup |
| Batch Size | 64 (classification), 16 (LM) |
| Classification Epochs | 2 (core), 25 (extended) |
| LM Steps | 25,000 (eval every 5,000) |
| Primary Seeds | 1001-1025 (25 seeds) |

---

## 3. Reproduction Instructions

### 1. Prepare Environment
```bash
cd attention_scaling_fresh_study
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python scripts/validate.py
# Should output: ALL PREFLIGHT CHECKS PASSED
```

### 3. Run Core Experiments
```bash
# Full core grid (25 seeds × 3 variants × 4 datasets = 300 runs)
python scripts/run_experiments.py \
    --config-dir configs \
    --output-dir results \
    --grid core

# Or run specific datasets:
python scripts/run_experiments.py --config-dir configs --output-dir results --grid core --datasets agnews sst2 imdb --seeds 1001 1002 1003
```

### 4. Run Extended Experiments
```bash
# 25-epoch classification (25 seeds × 3 variants × 2 datasets = 450 runs)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid extended

# Ablations (E1-E4)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid ablation
```

### 5. Analyze Results
```bash
python scripts/analyse_results.py \
    --results-dir results \
    --output-dir results/statistical_tests \
    --datasets agnews sst2 imdb wikitext2
```

### 6. Generate Figures
```bash
python scripts/generate_figures.py \
    results/statistical_tests \
    figures
```

### 7. Validate Results
```bash
python scripts/validate.py
```

---

## 4. Expected Outputs

### Per-Seed Records (`results/per_seed/`)
- `agnews_standard_seed_1001_25epoch.json` (75 files)
- `sst2_temp_seed_1001_25epoch.json` (75 files)
- `imdb_dk_seed_1001_25epoch.json` (75 files)
- `wikitext2_standard_seed_1001_25000steps.json` (75 files)

### Statistical Analysis (`results/statistical_tests/`)
- `agnews_statistical_analysis.json`
- `sst2_statistical_analysis.json`
- `imdb_statistical_analysis.json`
- `wikitext2_statistical_analysis.json`
- `summary_table.json`

### Figures (`figures/`)
- `fig_main_performance.png`
- `fig_paired_differences.png`
- `fig_training_curves.png`
- `fig_seed_distributions.png`
- `fig_gradient_ratios.png`
- `fig_head_diversity.png`
- `fig_temperature_ablation.png`
- `fig_efficiency.png`

### Checkpoints (`checkpoints/`)
- `agnews_standard_seed_1001_25epoch_epoch_2.pt`
- `wikitext2_standard_seed_1001_25000steps_step_25000.pt`

---

## 5. Key Reproducibility Guarantees

### Seed-Matched Design
- All variants share identical initialization per seed (verified by `tests/test_seed_match.py`)
- Identical data ordering per seed (seeded permutation generator)
- Identical optimizer state trajectory initialization

### Deterministic Operations
```python
# In scripts/run_experiments.py and training/trainer.py
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.deterministic = True
torch.cuda.benchmark = False
gen = torch.Generator(device="cpu").manual_seed(seed)  # for batch permutation
```

### Verified Properties (Unit Tests)
- ✅ Identical parameter counts across variants
- ✅ Bit-identical initialization across variants per seed
- ✅ Different seeds produce different initializations
- ✅ Gradient flow and shapes correct
- ✅ Forward pass produces finite outputs
- ✅ LM head weight tying works correctly

---

## 6. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| HF dataset download fails | Check internet; verify HF namespace in `dataset_configs.yaml` |
| CUDA OOM | Reduce batch size in `dataset_configs.yaml` |
| NaN losses | Gradient clipping already enabled; check logit clamping |
| Slow LM training | Reduce `max_steps` or `eval_every` for testing |
| HF rate limits | Set `HF_TOKEN` environment variable |

### Debug Mode
```bash
# Run with single seed for quick debug
python scripts/run_experiments.py --config-dir configs --output-dir results --grid core --seeds 1001 --variants standard --datasets agnews
```

---

## 7. Artifact Checklist for Release

- [x] Source code (`src/`)
- [x] Configurations (`configs/`)
- [x] Scripts (`scripts/`)
- [x] Tests (`tests/`)
- [x] Per-seed run records (`results/per_seed/`)
- [x] Statistical analysis (`results/statistical_tests/`)
- [x] Figures (`figures/`)
- [x] Validation reports (`reports/`)
- [ ] Manuscript PDF (`manuscript/`)
- [ ] Supplementary PDF (`supplementary/`)
- [ ] Requirements file (`requirements.txt`)
- [ ] README with instructions

---

## 8. Contact and Citation

**Repository**: `attention_scaling_fresh_study/`  
**Main Script**: `scripts/run_experiments.py`  
**Validation**: `scripts/validate.py`

To cite this work:
```
@article{attention_scaling_fresh_study_2026,
  title={A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention},
  author={Attention Scaling Verification Study},
  year={2026},
  note={Reproducibility package at attention_scaling_fresh_study/}
}
```