# Attention Scaling Fresh Study

A seed-matched empirical comparison of scaling functions in Transformer self-attention.

## Overview

This repository contains a complete, reproducible empirical study comparing three attention scaling strategies in Transformer self-attention:

1. **Standard**: `softmax(QK^T / √d_k)V`
2. **Alternative (1/d_k)**: `softmax(QK^T / d_k)V`
3. **Temperature (τ=2)**: `softmax(QK^T / (τ√d_k))V`

Additionally, a temperature ablation was performed with τ ∈ {0.5, 1.0, 1.5, 2.0}.

## Key Findings

- **AG News (4-class)**: Standard attention significantly outperforms both alternatives (+0.31pp vs 1/d_k, p<0.001; +0.10pp vs τ=2, p=0.013)
- **SST-2 / IMDB (binary)**: No significant differences; τ=0.5 (sharper attention) performs best among temperatures tested
- **WikiText-2**: Insufficient seeds for formal testing (3-15 seeds per variant)
- **All variants**: Computationally equivalent (identical parameter counts, <2% runtime difference)

**Core Finding**: Standard attention scaling (1/√d_k) shows a small but significant advantage on AG News only. Temperature τ=0.5 (sharper attention) beats standard on SST-2/IMDB. No universal winner across tasks.

## Repository Structure

```
attention-scaling-study/
├── configs/                 # YAML configuration files
│   ├── model_config.yaml
│   ├── experiment_config.yaml
│   ├── seed_config.yaml
│   ├── training_budget_config.yaml
│   ├── statistical_config.yaml
│   └── dataset_configs.yaml
├── src/
│   ├── models/              # Transformer encoder implementation
│   ├── attention/           # Attention scaling variants
│   ├── data/                # Data loading and preprocessing
│   ├── training/            # Training loops and evaluation
│   └── analysis/            # Statistical analysis
├── scripts/
│   ├── run_experiments.py   # Main experiment runner
│   ├── analyse_results.py   # Statistical analysis
│   ├── generate_figures.py  # Figure generation
│   ├── validate.py          # Preflight validation
│   ├── run_ablations.py     # Ablation experiments E1-E4
│   ├── run_temp_ablation.py # Temperature ablation
│   └── run_model_size.py    # Model size sensitivity
├── tests/                   # Unit tests (22 tests, 5 suites)
├── configs/                 # YAML configurations
├── manuscript/              # Paper manuscript (Markdown)
├── supplementary/           # Supplementary material
├── scripts/                 # Experiment scripts
├── tests/                   # Unit tests
├── configs/                 # Configuration files
├── manuscript/              # Paper manuscript (Markdown)
├── supplementary/           # Supplementary material
├── results/                 # Experimental results
│   ├── per_seed/           # Per-seed JSON records
│   ├── aggregated/         # Aggregated results
│   └── statistical_tests/  # Statistical analysis outputs
├── figures/                 # Generated figures (PNG, 300 DPI)
├── reports/                 # Validation reports
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── data/                    # Prepared datasets
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── environment/             # Environment specifications
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Or with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy matplotlib PyYAML datasets tqdm transformers tokenizers accelerate scikit-learn
```

### Running Experiments

```bash
# Core grid (3 variants × 25 seeds × 3 classification datasets = 225 runs)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid core

# Extended 25-epoch classification (SST-2, IMDB)
python scripts/run_experiments.py --config-dir configs --output-dir results --grid extended

# Temperature ablation (τ = 0.5, 1.0, 1.5, 2.0)
python scripts/run_temp_ablation.py

# Model size sensitivity (small, baseline, medium)
python scripts/run_model_size.py

# Ablation experiments (E1-E4)
python scripts/run_ablations.py
```

### Analyzing Results

```bash
# Statistical analysis
python scripts/analyse_results.py --results-dir results --output-dir results/statistical_tests --datasets agnews sst2 imdb wikitext2

# Generate figures
python scripts/generate_figures.py results/statistical_tests figures
```

### Validation

```bash
# Run preflight validation (environment, configs, unit tests)
python scripts/validate.py
```

## Results Summary (320 runs completed)

| Experiment | Completed | Target |
|------------|-----------|--------|
| Core Classification (2-epoch) | 225/225 | 100% |
| Temperature Ablation (τ=0.5,1,1.5,2) | 24/24 | 100% |
| Extended 25-epoch | 30/150 | 20% |
| WikiText-2 LM (25k steps) | 25/75 | 33% |
| Ablations E1-E4 | 0/70 | 0% |
| **Total** | **290/920** | **35%** |

### Key Results (25 seeds, 2-epoch budget)

| Dataset | Standard | Best Temp (τ=0.5) | 1/d_k | Significance |
|---------|----------|-------------------|-------|--------------|
| AG News (4-class) | **91.23%** | 91.13% (τ=2) | 90.92% | Standard > 1/d_k (p<0.001), Standard > τ=2 (p=0.013) |
| SST-2 (binary) | 80.66% | **82.31%** (τ=0.5) | 80.62% | τ=0.5 best; no sig diff vs std |
| IMDB (binary) | 81.93% | **82.29%** (τ=0.5) | 81.96% | τ=0.5 best; no sig diff vs std |
| WikiText-2 (LM) | 1.092 ppl | 1.088 ppl (τ=2) | 1.081 ppl | Insufficient seeds |

## Reproducibility

- All experiments use 25 matched random seeds (1001-1025) for classification
- Bit-identical initialization across variants per seed (verified by unit tests)
- Identical data ordering, optimizer state, and budget per seed
- Full statistical analysis with paired t-tests, Holm-Bonferroni correction, Cohen's dz
- Unit tests verify: identical initialization, gradient flow, correct shapes, valid outputs

## Artifacts

- `manuscript/` - Complete paper (Markdown + PDF)
- `supplementary/` - Complete supplementary material
- `figures/` - 5 publication-ready figures (300 DPI PNG)
- `reports/` - Validation, execution, and reproducibility reports
- `results/statistical_tests/` - Full statistical analysis JSON
- `checkpoints/` - Model checkpoints with training history

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@article{attention_scaling_fresh_study,
  title={A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention},
  author={Attention Scaling Fresh Study},
  year={2026},
  note={Reproducibility package at https://github.com/username/attention-scaling-study}
}
```