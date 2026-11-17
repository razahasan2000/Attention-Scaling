# A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention

**Authors:** Raza Hasan, Vishal Dattana, and Salman Mahmood

This repository contains the implementation, configurations, experiment runners, tests, result records, analysis utilities, and figures for the study. The manuscript and supplementary material are intended for separate journal publication and are **not stored in this repository**.

## 1. Research overview

The study examines how the scaling divisor in Transformer self-attention affects attention distributions and downstream model behaviour. The key question is when changing the divisor becomes consequential, given the scale of the raw attention logits.

For query matrix $Q$, key matrix $K$, value matrix $V$, and per-head key dimension $d_k$, define the raw attention-logit matrix:

$$
Z = QK^\top.
$$

For a scaling divisor $s$, the attention output is:

$$
\operatorname{Attention}(Q,K,V;s)
=
\operatorname{softmax}\!\left(\frac{Z}{s}\right)V.
$$

It is also useful to define the inverse-divisor coefficient $\beta=1/s$. The same expression can then be written as:

$$
\operatorname{Attention}(Q,K,V;\beta)
=
\operatorname{softmax}(\beta Z)V.
$$

### 1.1 The three scaling variants

| Variant | Divisor $s$ | Inverse-divisor coefficient $\beta=1/s$ |
|---|---:|---:|
| Standard | $\sqrt{d_k}$ | $1/\sqrt{d_k}$ |
| Reciprocal-dimension | $d_k$ | $1/d_k$ |
| Temperature, $\tau=2$ | $2\sqrt{d_k}$ | $1/(2\sqrt{d_k})$ |

For one attention row with raw logits $z=(z_1,\ldots,z_n)$, the probability assigned to valid position $i$ is:

$$
P_\beta(i)
=
\frac{\exp(\beta z_i)}
{\displaystyle\sum_{j=1}^{n}\exp(\beta z_j)},
\qquad i=1,\ldots,n.
$$

Here, $n$ is the number of positions attended to by that query after applying the attention mask.

## 2. The operating-point relationship

The manuscript defines the operating point using $\sigma$, the standard deviation of the raw attention logits **before scaling or division**. The logits used for this measurement must be restricted to valid, unmasked query-key positions.

For two inverse-divisor coefficients $\beta_1$ and $\beta_2$, the second-order approximation derived in the manuscript and supplementary Section S17 is:

$$
D_{\mathrm{KL}}\!\left(P_{\beta_1}\,\middle\|\,P_{\beta_2}\right)
\approx
\frac{1}{2}(\beta_1-\beta_2)^2\sigma^2
\left(1-\frac{1}{n}\right).
$$

The approximation assumes an attention row whose pre-softmax logits are approximately independent and identically distributed, with raw-logit variance $\sigma^2$. The factor $(1-1/n)$ is the finite-population correction. This is an approximation, not an exact identity: at larger logit scales, softmax saturation causes measured divergence to grow more slowly than the quadratic prediction.

### 2.1 Interpreting the thresholds

The supplementary material quantifies the operating point at which a divisor pair reaches a selected KL-divergence level. For the specific reference setting $d_k=32$ and $n=64$, the reported approximate thresholds are:

| Divisor comparison | $\sigma$ for KL = 0.01 nats | $\sigma$ for KL = 0.1 nats |
|---|---:|---:|
| Standard vs. $1/d_k$ | 0.98 | 3.10 |
| Standard vs. $\tau=2$ | 1.61 | 5.10 |

These are pair-specific thresholds for the stated $d_k$, $n$, and target divergence. They are not universal cut-offs for every model, head dimension, sequence length, or task. The practical recommendation is to measure and report the raw pre-division logit standard deviation, together with the attention-mask and sequence-length conditions.

## 3. Experimental programme

The manuscript reports **787 completed runs** across three stages:

| Stage | Runs | Scope |
|---|---:|---|
| Stage 1 | 314 | GPU-scale experiments |
| Stage 2 | 339 | Replica-scale completion: 312 planned runs and 27 mask-artifact probes |
| Stage 3 | 134 | Operating-point programme: 128 matched-seed training runs and six pretrained-model evaluations |
| **Total** | **787** | |

The Stage 1 manifest covers the full-scale core classification comparison, temperature sensitivity, extended-budget experiments, an initial language-modelling protocol, and focused ablations. The Stage 2 and Stage 3 run groups are described below so that their records can be identified in the repository.

### 3.1 Stage 2: replica-scale experiments

The Stage 2 runner implements the following groups:

| Runner phase | Runs | Experiment |
|---|---:|---|
| **core** | 75 | AG News core comparison, 25 seeds and three variants |
| **lm** | 75 | WikiText-2 with corrected causal masking, 25 seeds and three variants |
| **grid** | 108 | AG News head-dimension grid, $d_k\in\{32,64,128\}$, 12 seeds and three variants |
| **conv** | 30 | SST-2 and IMDB train-to-convergence experiments |
| **abl** | 24 | Replica E1-E4 ablation arms |
| **lmleak** and **lmc** | 27 | Mask-artifact and causal-mask comparison probes |
| **Total** | **339** | |

The replica protocol deliberately reduces the data scale and sequence length relative to Stage 1. Classification uses a 2,000-example training subset, 500 validation examples, 64-token sequences, and batch size 32. The test subsets are 4,000 AG News examples, the official 872-example SST-2 development set, and 2,500 IMDB examples. WikiText-2 uses 96,000 training tokens and 40,000 validation tokens in the corrected causal language-modelling protocol.

The original near-one WikiText-2 perplexity results came from a causal-mask leak and must not be interpreted as valid autoregressive language-modelling performance. The corrected causal-mask implementation and its verification probes are part of the repository.

### 3.2 Stage 3: operating-point experiments

| Runner phase or resource | Runs | Experiment |
|---|---:|---|
| **gain** | 72 | AG News logit-gain sweep: six gain values, three variants, four seeds |
| **regime** and **regime128** | 18 | Instrumented AG News training at two head dimensions |
| **bridge** | 20 | QK-Norm and soft-cap stabiliser arms |
| **lmgain** | 18 | WikiText-2 logit-gain experiments at sequence length 128 |
| **frontier_gpt2.py** | 6 | Inference-time divisor swaps in GPT-2 and GPT-2-medium |
| **Total** | **134** | |

The pretrained-model evaluations swap the divisor at inference time without retraining the model. The separate larger-model GPU script provides an optional extension pathway; it is not evidence that 1B-7B training or fine-tuning experiments were completed in this study.

### 3.3 Results interpretation

The results are conditional on the operating point and experimental regime. The manuscript reports a Stage 1 separation on AG News, while the binary sentiment comparisons are not statistically distinguishable in the primary 2-epoch comparisons. The Stage 2 reduced-scale replica does not reproduce the Stage 1 AG News separation. Stage 3 investigates this difference by measuring and manipulating the raw-logit operating point.

## 4. Repository structure

The following tree reflects the files and directories present on the repository's main branch. It lists repository resources only; manuscript and supplementary files are intentionally excluded.

```text
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
```

### 4.1 Key implementation and result files

| Path | Purpose |
|---|---|
| [src/attention/scaling.py](src/attention/scaling.py) | Attention scaling implementations |
| [src/models/transformer_encoder.py](src/models/transformer_encoder.py) | Transformer encoder |
| [src/training/trainer.py](src/training/trainer.py) | Training routines |
| [src/data/prepare.py](src/data/prepare.py) | Dataset preparation utilities |
| [scripts/experiments/common.py](scripts/experiments/common.py) | Shared replica model, masking, seed and probe utilities |
| [scripts/experiments/runner.py](scripts/experiments/runner.py) | Stage 2 and Stage 3 phase runner |
| [scripts/experiments/prepare_data.py](scripts/experiments/prepare_data.py) | Replica dataset preparation |
| [scripts/experiments/verify_mask.py](scripts/experiments/verify_mask.py) | Causal and padding mask verification |
| [scripts/experiments/aggregate.py](scripts/experiments/aggregate.py) | Stage 2 result aggregation |
| [scripts/experiments/agg_stage3.py](scripts/experiments/agg_stage3.py) | Stage 3 result aggregation |
| [scripts/experiments/regime_law.py](scripts/experiments/regime_law.py) | Operating-point and divisor-law analysis |
| [scripts/experiments/frontier_gpt2.py](scripts/experiments/frontier_gpt2.py) | GPT-2 and GPT-2-medium inference evaluation |
| [scripts/frontier/frontier_gpu_1b7b.py](scripts/frontier/frontier_gpu_1b7b.py) | Optional larger-model inference evaluation |
| [results/results.jsonl](results/results.jsonl) | Stage 2 and Stage 3 training-run records |
| [results/aggregate.json](results/aggregate.json) | Stage 2 aggregate statistics |
| [results/stage3_operatingpoint.json](results/stage3_operatingpoint.json) | Stage 3 operating-point results |
| [results/stage3_theory.json](results/stage3_theory.json) | Stage 3 theoretical analysis and thresholds |
| [results/frontier_gpt2.json](results/frontier_gpt2.json) | Pretrained GPT-2 evaluation results |
| [results/mask_verification.json](results/mask_verification.json) | Saved causal-mask verification results |

## 5. Model and datasets

### 5.1 Primary model

The primary encoder configuration uses a pre-layer-normalised Transformer with two blocks, model width 128, four attention heads, head dimension 32, feed-forward dimension 256, maximum sequence length 128, GELU activation, and dropout 0.1. The optimisation configuration specifies AdamW, learning rate $3\times10^{-4}$, weight decay 0.01, and gradient clipping at 1.0.

Stage 2 is a reduced-scale replica and does not use the same data scale and sequence length as Stage 1. For example, the replica classification runner uses 64-token sequences. Do not assume that the primary model configuration alone describes every Stage 2 or Stage 3 run; use the phase runner and configuration files for the relevant protocol.

### 5.2 Datasets

| Dataset | Task |
|---|---|
| AG News | Four-class news classification |
| SST-2 | Binary sentiment classification |
| IMDB | Binary movie-review sentiment classification |
| WikiText-2 | Language modelling |

### 5.3 Required local raw-data layout

The Stage 2 preparation script expects the following files under **data/raw/**:

```text
data/
└── raw/
    ├── agnews_train.csv
    ├── agnews_test.csv
    ├── SST-2/
    │   ├── train.tsv
    │   └── dev.tsv
    ├── imdb_train.parquet
    ├── imdb_test.parquet
    ├── wt2_train.parquet
    └── wt2_valid.parquet
```

The preparation script does not download these files automatically. Obtain each dataset from its original provider, prepare the files in the expected format, and follow the relevant dataset terms. The dataset files are not redistributed in this repository.

## 6. Installation

Clone the repository and install the dependencies:

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
python -m pip install pytest
\`\`\`

The final command installs the test runner, which is not currently listed in requirements.txt. For a CPU-only PyTorch installation, use the appropriate command from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 7. Reproduction workflow

Run the commands from the repository root. Prepare the raw dataset files described in Section 5.3 before starting Stage 2.

### 7.1 Prepare the replica datasets and verify masking

```bash
python scripts/experiments/prepare_data.py
python scripts/experiments/verify_mask.py
```

### 7.2 Run Stage 2

The runner is resume-safe. It skips completed jobs, and its default wall-time limit is 520 seconds per invocation. Re-run a phase command until it reports that there is nothing left to do.

```bash
python scripts/experiments/runner.py --phase core --procs 2
python scripts/experiments/runner.py --phase lm --procs 2
python scripts/experiments/runner.py --phase grid --procs 2
python scripts/experiments/runner.py --phase conv --procs 2
python scripts/experiments/runner.py --phase abl --procs 2
python scripts/experiments/runner.py --phase lmleak --procs 2
python scripts/experiments/runner.py --phase lmc --procs 2
```

### 7.3 Run Stage 3

```bash
python scripts/experiments/runner.py --phase gain --procs 2
python scripts/experiments/runner.py --phase bridge --procs 2
python scripts/experiments/runner.py --phase regime --procs 2
python scripts/experiments/runner.py --phase regime128 --procs 2
python scripts/experiments/runner.py --phase lmgain --procs 2
```

The GPT-2 and GPT-2-medium inference evaluations are run separately as described in Section 8.

### 7.4 Aggregate the recorded results

After the relevant phase records have been generated:

```bash
python scripts/experiments/aggregate.py
python scripts/experiments/agg_stage3.py
```

The Stage 2 aggregator writes results/aggregate.json; the Stage 3 aggregator writes results/stage3_operatingpoint.json. These scripts expect the corresponding completed records in results/results.jsonl.

### 7.5 Run the tests

```bash
python -m pytest tests/
```

The tests cover attention scaling, causal masking, configuration handling, data preparation, head-dimension grids, early stopping, gradient paths, and seed matching.

### 7.6 Existing figures and legacy figure-generation utility

The publication figures are already stored in [figures/](figures/). The current scripts/generate_figures.py utility expects legacy statistical-analysis JSON files under results/statistical_tests/; those files are not part of the current Stage 2/Stage 3 result layout. It should not be treated as the figure-generation step for the current replica pipeline.

## 8. Pretrained-model inference evaluation

### 8.1 GPT-2 and GPT-2-medium

The GPT-2 evaluation swaps the attention divisor at inference time while retaining pretrained weights. It reads the WikiText-2 validation parquet file at data/raw/wt2_valid.parquet.

```bash
python scripts/experiments/frontier_gpt2.py \
  --models gpt2,gpt2-medium \
  --nseq 160 \
  --seqlen 128
```

The results are written to results/frontier_gpt2.json.

### 8.2 Larger pretrained decoder models

The optional GPU script supports inference-time divisor evaluation for supported Hugging Face decoder architectures. For example:

```bash
python scripts/frontier/frontier_gpu_1b7b.py \
  --model gpt2-xl \
  --nseq 500 \
  --seqlen 256
```

The script can load WikiText-2 validation data through the Hugging Face datasets library, or accept a local parquet file through --data. GPU memory requirements depend on the selected model, sequence length, batch size, and precision. Some model repositories require authentication and licence acceptance.

**Scope:** this script is an optional inference-time evaluation pathway. It does not establish trained or fine-tuned 1B-7B model behaviour, and those experiments are not included in the reported 787 runs.

## 9. Results available in this repository

The current results/results.jsonl contains **467 run records**: 339 Stage 2 records and 128 Stage 3 matched-seed training records. The six pretrained GPT-2 evaluations are stored separately in results/frontier_gpt2.json.

The manuscript's Stage 1 contributes a further 314 runs to the reported total of 787. Those Stage 1 runs are not represented as individual records in the current results/results.jsonl. The Stage 2 and Stage 3 files should therefore not be described as a complete per-run archive of all 787 experiments.

The Stage 1 WikiText-2 values used the earlier v1 masking protocol and are superseded by the corrected causal-mask Stage 2 language-modelling results. They should not be interpreted as valid autoregressive language-modelling results.

## 10. Publication and supplementary material

The manuscript and supplementary material will be published separately through the journal. Neither document is included in this repository. The final article DOI and publisher-hosted supplementary-material URL can be added here once the journal publication details are confirmed.

## 11. Data and code availability

The source code, configurations, tests, Stage 2 and Stage 3 run records, aggregate outputs, diagnostics, and figures are available in this repository. Dataset files are obtained from their original providers and are not redistributed here. The manuscript and supplementary material are separate publication resources. The Stage 1 run-level records are not included in the current results/results.jsonl file.

## 12. Citation

Please cite the associated article using its final published bibliographic details once available. The final DOI will be added here after publication.

## 13. License

This project is released under the [MIT License](LICENSE).
