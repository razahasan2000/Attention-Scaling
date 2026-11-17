# A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention

**Authors:** Raza Hasan, Vishal Dattana, and Salman Mahmood

This repository contains the code, experiment configurations, tests, result records, analysis scripts, and figures associated with the study. The manuscript and supplementary material are intended for separate journal publication and are **not stored in this repository**.

## 1. Research overview

The study investigates how the scaling divisor in Transformer self-attention affects attention distributions and model behaviour. The central question is when changing the divisor becomes consequential, given the scale of the raw attention logits.

For query matrix $Q$, key matrix $K$, value matrix $V$, and per-head key dimension $d_k$, define the raw attention-logit matrix as

$$
Z = QK^\top.
$$

The attention output for a divisor $s$ is

$$
\operatorname{Attention}(Q,K,V;s)
=
\operatorname{softmax}\left(\frac{Z}{s}\right)V.
$$

Equivalently, define the inverse-divisor coefficient $b=1/s$. Then

$$
\operatorname{Attention}(Q,K,V;b)
=
\operatorname{softmax}(bZ)V.
$$

### 1.1 Scaling variants

| Variant | Divisor $s$ | Inverse-divisor coefficient $b=1/s$ |
|---|---:|---:|
| Standard | $\sqrt{d_k}$ | $1/\sqrt{d_k}$ |
| Reciprocal-dimension | $d_k$ | $1/d_k$ |
| Temperature, $\tau=2$ | $\tau\sqrt{d_k}=2\sqrt{d_k}$ | $1/(2\sqrt{d_k})$ |

For a single attention row with raw logits $z=(z_1,\ldots,z_n)$, the corresponding probability distribution is

$$
P_b(i)
=
\frac{\exp(bz_i)}
{\displaystyle\sum_{j=1}^{n}\exp(bz_j)},
\qquad i=1,\ldots,n.
$$

Here, $n$ is the number of valid positions attended to in that row. Masked positions are excluded.

## 2. Operating-point relationship

The manuscript defines the operating point using the standard deviation $\sigma$ of the **raw, pre-division attention logits**. It is measured before applying any of the three scaling divisors.

For two inverse-divisor coefficients $b_1$ and $b_2$, the second-order softmax approximation reported in the manuscript is

$$
D_{\mathrm{KL}}\!\left(P_{b_1}\,\middle\|\,P_{b_2}\right)
\approx
\frac{1}{2}(b_1-b_2)^2\sigma^2
\left(1-\frac{1}{n}\right).
\tag{1}
$$

In this expression:

- $D_{\mathrm{KL}}(P_{b_1}\|P_{b_2})$ is the Kullback-Leibler divergence between the two attention distributions.
- $b_1$ and $b_2$ are the inverse-divisor coefficients, not the divisors themselves.
- $\sigma$ is the standard deviation of the raw attention logits before scaling.
- $n$ is the number of valid positions in the attention row.
- The factor $(1-1/n)$ is the finite-population correction in the approximation.

Equation (1) describes a second-order approximation, not an exact identity for all logit scales. It captures the approximately quadratic increase in distributional divergence with $\sigma$ while the approximation is valid. At larger scales, softmax saturation causes the measured divergence to depart from the quadratic prediction. The manuscript and supplementary derivation discuss the assumptions and validation range.

The practical implication is that different divisors can produce very similar attention distributions when the raw-logit operating point is low, while their effects can become pronounced at higher operating points. The relevant $\sigma$ should therefore be measured and reported rather than replaced by a universal fixed threshold.

## 3. Experimental programme

The manuscript reports **787 completed runs** across three stages:

| Stage | Completed runs | Focus |
|---|---:|---|
| Stage 1 | 314 | Primary seed-matched comparisons |
| Stage 2 | 339 | Reduced-scale replication, diagnostics, convergence and masking investigations |
| Stage 3 | 134 | Operating-point analysis, logit-gain and stabilisation experiments, and pretrained-model evaluations |
| **Total** | **787** | |

Stage 2 includes 27 mask-artifact probes. Stage 3 includes 128 matched-seed training runs and six pretrained-model evaluations.

The study uses AG News, SST-2, IMDB, and WikiText-2. Its analyses include paired-seed comparisons, attention diagnostics, head-dimension and logit-gain investigations, QK normalisation, soft-capping, causal-mask verification, and inference-time divisor swaps in pretrained GPT-2 models.

The results are operating-point dependent. This README should not be interpreted as claiming that one divisor is universally optimal across tasks, model scales, or training regimes. The Stage 2 reduced-scale AG News replication did not reproduce the Stage 1 separation, and the corrected language-modelling mask is essential when interpreting the WikiText-2 experiments.

## 4. Repository structure

The tree below reflects the files and directories currently present on the repository's main branch. It is a repository map, not a list of manuscript or supplementary files.

\`\`\`text
Attention-Scaling/
├── LICENSE
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset_configs.yaml
│   ├── experiment_config.yaml
│   ├── model_config.yaml
│   ├── seed_config.yaml
│   ├── statistical_config.yaml
│   └── training_budget_config.yaml
├── figures/
│   ├── fig_architecture.png
│   ├── fig_convergence.png
│   ├── fig_design.png
│   ├── fig_dk_grid.png
│   ├── fig_efficiency.png
│   ├── fig_frontier.png
│   ├── fig_gain_sweep.png
│   ├── fig_gradient_ratios.png
│   ├── fig_head_diversity.png
│   ├── fig_lm_corrected.png
│   ├── fig_lm_perplexity.png
│   ├── fig_main_performance.png
│   ├── fig_paired_differences.png
│   ├── fig_regime_law.png
│   ├── fig_stage2_core.png
│   ├── fig_temperature_ablation.png
│   └── fig_training_curves.png
├── results/
│   ├── aggregate.json
│   ├── diagnostics.json
│   ├── frontier_gpt2.json
│   ├── mask_verification.json
│   ├── results.jsonl
│   ├── stage3_operatingpoint.json
│   ├── stage3_theory.json
│   ├── verification_checks.json
│   └── verification_checks_dev.json
├── scripts/
│   ├── analyse_results.py
│   ├── experiments/
│   │   ├── agg_stage3.py
│   │   ├── aggregate.py
│   │   ├── clean_results.py
│   │   ├── common.py
│   │   ├── diagnostics.py
│   │   ├── frontier_gpt2.py
│   │   ├── prepare_data.py
│   │   ├── prepare_regime.py
│   │   ├── regime_law.py
│   │   ├── runner.py
│   │   ├── smoke_stage3.py
│   │   └── verify_mask.py
│   ├── frontier/
│   │   └── frontier_gpu_1b7b.py
│   ├── generate_figures.py
│   ├── run_ablations.py
│   ├── run_ablations_complete.py
│   ├── run_all_v2.py
│   ├── run_experiments.py
│   ├── run_model_size.py
│   ├── run_temp_ablation.py
│   ├── smoke_v2.py
│   ├── validate.py
│   └── verify_numbers.py
├── src/
│   ├── analysis/
│   │   └── analyze_results.py
│   ├── attention/
│   │   └── scaling.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── models/
│   │   └── transformer_encoder.py
│   └── training/
│       └── trainer.py
└── tests/
    ├── test_attention_scale.py
    ├── test_causal_mask.py
    ├── test_config.py
    ├── test_data_prepare.py
    ├── test_dk_grid.py
    ├── test_early_stopping.py
    ├── test_gradient_paths.py
    └── test_seed_match.py
\`\`\`

### 4.1 Key implementation and result files

| Path | Purpose |
|---|---|
| [src/attention/scaling.py](src/attention/scaling.py) | Attention scaling implementation |
| [src/models/transformer_encoder.py](src/models/transformer_encoder.py) | Transformer encoder |
| [src/training/trainer.py](src/training/trainer.py) | Training routines |
| [src/data/prepare.py](src/data/prepare.py) | Data preparation implementation |
| [scripts/experiments/common.py](scripts/experiments/common.py) | Shared experiment and model utilities |
| [scripts/experiments/runner.py](scripts/experiments/runner.py) | Staged experiment runner |
| [scripts/experiments/prepare_data.py](scripts/experiments/prepare_data.py) | Replica dataset preparation |
| [scripts/experiments/verify_mask.py](scripts/experiments/verify_mask.py) | Causal and padding mask checks |
| [scripts/experiments/aggregate.py](scripts/experiments/aggregate.py) | Aggregation of experiment results |
| [scripts/experiments/agg_stage3.py](scripts/experiments/agg_stage3.py) | Stage 3 aggregation |
| [scripts/experiments/regime_law.py](scripts/experiments/regime_law.py) | Operating-point analysis |
| [scripts/experiments/frontier_gpt2.py](scripts/experiments/frontier_gpt2.py) | GPT-2 and GPT-2-medium inference evaluation |
| [scripts/frontier/frontier_gpu_1b7b.py](scripts/frontier/frontier_gpu_1b7b.py) | Optional larger-model inference evaluation script |
| [results/aggregate.json](results/aggregate.json) | Aggregated statistics |
| [results/results.jsonl](results/results.jsonl) | Experiment-level result records |
| [results/stage3_operatingpoint.json](results/stage3_operatingpoint.json) | Stage 3 operating-point records |
| [results/stage3_theory.json](results/stage3_theory.json) | Stage 3 theoretical analysis records |
| [results/mask_verification.json](results/mask_verification.json) | Mask verification output |

## 5. Model and datasets

The primary encoder described in the manuscript uses a pre-layer-normalised Transformer with two blocks, model width 128, four attention heads, head dimension 32, feed-forward dimension 256, maximum sequence length 128, GELU activation, and dropout 0.1. The reported optimisation setup includes AdamW, learning rate $3\times10^{-4}$, weight decay 0.01, gradient clipping at 1.0, and FP32 computation.

The repository also contains reduced-scale replica experiments. Their settings differ from the primary encoder configuration. Use the stage-specific configuration files and experiment records when reproducing a particular condition.

| Dataset | Task |
|---|---|
| AG News | Four-class news classification |
| SST-2 | Binary sentiment classification |
| IMDB | Binary movie-review sentiment classification |
| WikiText-2 | Language modelling |

The datasets are not redistributed in this repository. Obtain them from their original providers and follow the corresponding terms of use. The preparation scripts expect local dataset files in the paths specified in the code and configuration.

## 6. Installation

Clone the repository and install its Python dependencies:

\`\`\`bash
git clone https://github.com/razahasan2000/Attention-Scaling.git
cd Attention-Scaling

python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

For a CPU-only PyTorch installation, use the appropriate command from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 7. Reproduction workflow

Run commands from the repository root. Prepare the required local dataset files before running the preparation script.

### 7.1 Prepare data and verify the language-modelling mask

\`\`\`bash
python scripts/experiments/prepare_data.py
python scripts/experiments/verify_mask.py
\`\`\`

### 7.2 Run experiment phases

The experiment runner accepts a phase name. The following are examples of supported phases:

\`\`\`bash
python scripts/experiments/runner.py --phase core --procs 2
python scripts/experiments/runner.py --phase conv --procs 2
python scripts/experiments/runner.py --phase grid --procs 2
python scripts/experiments/runner.py --phase lm --procs 2
python scripts/experiments/runner.py --phase gain --procs 2
\`\`\`

The runner records completed jobs and skips matching completed entries when resumed. Consult the runner for the complete phase definitions, seed ranges, and job settings. These example commands do not, by themselves, reproduce all 787 reported runs.

### 7.3 Aggregate results and generate figures

\`\`\`bash
python scripts/experiments/aggregate.py
python scripts/experiments/agg_stage3.py
python scripts/generate_figures.py
\`\`\`

### 7.4 Run tests

\`\`\`bash
pytest tests/
\`\`\`

## 8. Pretrained-model inference evaluation

The GPT-2 evaluation swaps the attention divisor at inference time while retaining the pretrained weights:

\`\`\`bash
python scripts/experiments/frontier_gpt2.py \
  --models gpt2,gpt2-medium \
  --nseq 160 \
  --seqlen 128
\`\`\`

The repository also includes [scripts/frontier/frontier_gpu_1b7b.py](scripts/frontier/frontier_gpu_1b7b.py), which provides an inference-time evaluation pathway for supported larger Hugging Face decoder models. The presence of this script does not mean that 1B–7B model experiments were completed as part of the reported 787 runs. The reported pretrained-model evaluations are GPT-2 (124M) and GPT-2-medium (355M).

## 9. Publication and supplementary material

The manuscript and supplementary material will be published separately through the journal. Neither document is included in this repository. The final article DOI and the publisher-hosted supplementary-material URL can be added here once the journal publication details are confirmed.

## 10. Data and code availability

The source code, configurations, tests, result records, analysis scripts, and figures are available in this repository. Dataset files are obtained from their original providers and are not redistributed here. The manuscript and supplementary material are separate publication resources.

## 11. Citation

Please cite the associated article using its final published bibliographic details once available. The final DOI will be added here after publication.

## 12. License

This project is released under the [MIT License](LICENSE).
