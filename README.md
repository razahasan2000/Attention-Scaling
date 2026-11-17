# Attention Scaling in Transformer Self-Attention

This repository contains the implementation, experiment configurations, analysis scripts, verification checks, result records, and figures for the study:

**A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention**

**Authors:** Raza Hasan, Vishal Dattana, and Salman Mahmood

The study investigates how alternative scaling divisors affect attention distributions and model behaviour under controlled, seed-matched experimental conditions. This repository supports inspection and reproduction of the computational work. The manuscript and journal supplementary material are maintained separately for publication and are not hosted here.

---

## Research overview

Scaled dot-product attention is commonly written as

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{s}\right)V,
\]

where \(s\) is the scaling divisor and \(d_k\) is the key/query head dimension.

This study compares three scaling choices:

| Variant | Scaling divisor | Attention logits |
|---|---:|---|
| Standard | \(\sqrt{d_k}\) | \(QK^\top/\sqrt{d_k}\) |
| Reciprocal-dimension | \(d_k\) | \(QK^\top/d_k\) |
| Temperature (\(\tau=2\)) | \(2\sqrt{d_k}\) | \(QK^\top/(2\sqrt{d_k})\) |

The variants are compared while controlling the experimental setup, including matched random seeds and model initialisation where specified by each experimental stage. The purpose is to isolate the effect of the attention-logit divisor rather than attribute differences to unrelated changes in training conditions.

## Experimental programme

The manuscript reports a three-stage experimental programme comprising **787 completed runs**:

| Stage | Runs | Purpose |
|---|---:|---|
| Stage 1 | 314 | Primary seed-matched comparison across the classification tasks |
| Stage 2 | 339 | Reduced-scale replication, additional diagnostics, convergence and masking investigations |
| Stage 3 | 134 | Operating-point analysis, logit-gain and stabilisation experiments, and pretrained-model evaluations |
| **Total** | **787** | |

Stage 2 includes 27 mask-artifact probes in addition to the planned experimental runs. Stage 3 includes 128 matched-seed training runs and six pretrained-model evaluations.

### Main experimental considerations

- **Matched-seed comparisons:** paired experimental conditions are evaluated using shared seeds to support controlled comparisons.
- **Multiple tasks:** the study includes AG News, SST-2, IMDB, and WikiText-2.
- **Attention operating point:** the analysis examines the standard deviation of raw attention logits before divisor scaling and relates it to changes in attention distributions.
- **Mask verification:** causal and padding masks are checked explicitly. The study identifies and investigates a masking artefact that affected earlier language-modelling results.
- **Additional diagnostics:** the later stages examine head dimension, logit gain, QK normalisation, soft-capping, and inference-time divisor changes in pretrained GPT-2 models.

The results do not support a universal claim that one divisor is always best. The observed effect depends on the task, experimental regime, and attention-logit operating point. In particular, the reduced-scale Stage 2 replication did not reproduce the Stage 1 AG News separation, while the binary sentiment comparisons were statistically indistinguishable in the reported primary comparisons. The corrected WikiText-2 masking experiments are important when interpreting the language-modelling results.

## Repository structure

The links below point to files and directories currently present in this repository.

\`\`\`text
Attention-Scaling/
├── configs/
│   ├── dataset_configs.yaml
│   ├── experiment_config.yaml
│   ├── model_config.yaml
│   ├── seed_config.yaml
│   ├── statistical_config.yaml
│   └── training_budget_config.yaml
├── figures/
│   ├── fig_architecture.png
│   ├── fig_design.png
│   ├── fig_main_performance.png
│   ├── fig_paired_differences.png
│   ├── fig_convergence.png
│   ├── fig_lm_corrected.png
│   ├── fig_regime_law.png
│   └── ...
├── results/
│   ├── aggregate.json
│   ├── diagnostics.json
│   ├── frontier_gpt2.json
│   ├── mask_verification.json
│   ├── results.jsonl
│   ├── stage3_operatingpoint.json
│   └── stage3_theory.json
├── scripts/
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── common.py
│   │   ├── prepare_data.py
│   │   ├── verify_mask.py
│   │   ├── aggregate.py
│   │   ├── agg_stage3.py
│   │   ├── regime_law.py
│   │   └── frontier_gpt2.py
│   ├── frontier/
│   │   └── frontier_gpu_1b7b.py
│   ├── analyse_results.py
│   ├── generate_figures.py
│   └── ...
├── src/
│   ├── attention/
│   │   └── scaling.py
│   ├── analysis/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── training/
├── tests/
├── requirements.txt
└── LICENSE
\`\`\`

### Useful code and result links

| Resource | Description |
|---|---|
| [src/attention/scaling.py](src/attention/scaling.py) | Attention scaling implementations |
| [src/models/transformer_encoder.py](src/models/transformer_encoder.py) | Transformer encoder implementation |
| [src/training/trainer.py](src/training/trainer.py) | Training implementation |
| [scripts/experiments/runner.py](scripts/experiments/runner.py) | Experiment runner for the staged experiment phases |
| [scripts/experiments/common.py](scripts/experiments/common.py) | Shared experiment utilities and model construction |
| [scripts/experiments/prepare_data.py](scripts/experiments/prepare_data.py) | Preparation of replica-scale datasets |
| [scripts/experiments/verify_mask.py](scripts/experiments/verify_mask.py) | Causal and padding mask verification |
| [scripts/experiments/aggregate.py](scripts/experiments/aggregate.py) | Aggregation of experiment results |
| [scripts/experiments/agg_stage3.py](scripts/experiments/agg_stage3.py) | Stage 3 result aggregation |
| [scripts/experiments/regime_law.py](scripts/experiments/regime_law.py) | Operating-point and regime-law analysis |
| [scripts/experiments/frontier_gpt2.py](scripts/experiments/frontier_gpt2.py) | Inference-time evaluation on GPT-2 and GPT-2-medium |
| [scripts/frontier/frontier_gpu_1b7b.py](scripts/frontier/frontier_gpu_1b7b.py) | Script for inference-time divisor evaluation on supported larger pretrained models |
| [results/aggregate.json](results/aggregate.json) | Aggregated experimental statistics |
| [results/results.jsonl](results/results.jsonl) | Experiment-level result records |
| [results/stage3_operatingpoint.json](results/stage3_operatingpoint.json) | Stage 3 operating-point results |
| [results/mask_verification.json](results/mask_verification.json) | Saved mask verification results |

## Experimental model and datasets

The primary encoder described in the manuscript uses a pre-layer-normalised Transformer architecture with two Transformer blocks, model width 128, four attention heads, head dimension 32, feed-forward dimension 256, maximum sequence length 128, GELU activation, and dropout 0.1. The reported training setup includes AdamW, learning rate \(3\times10^{-4}\), weight decay 0.01, gradient clipping at 1.0, and FP32 computation.

The programme also contains reduced-scale replica experiments. Their settings are not identical to the primary encoder configuration. Consult the relevant configuration files and experiment records when reproducing a particular stage.

The datasets used in the study are:

- **AG News:** multi-class news classification.
- **SST-2:** binary sentiment classification.
- **IMDB:** binary movie-review sentiment classification.
- **WikiText-2:** language modelling.

Dataset preparation scripts expect source data in the local data directory. The datasets themselves are not redistributed by this repository. Follow the original dataset providers' terms and instructions for obtaining them.

## Installation

Use Python 3.10 or later where possible. Install the dependencies from the repository root:

\`\`\`bash
git clone https://github.com/razahasan2000/Attention-Scaling.git
cd Attention-Scaling

python -m venv .venv
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

For a CPU-only PyTorch installation, install the appropriate CPU build of PyTorch before installing the remaining requirements, following the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Reproduction workflow

The commands below illustrate the repository's experiment workflow. Run them from the repository root. Dataset files must first be placed in the local paths expected by the preparation script.

### 1. Prepare replica-scale datasets

\`\`\`bash
python scripts/experiments/prepare_data.py
\`\`\`

### 2. Verify causal and padding masks

\`\`\`bash
python scripts/experiments/verify_mask.py
\`\`\`

### 3. Run experiment phases

The experiment runner supports named phases. For example:

\`\`\`bash
python scripts/experiments/runner.py --phase core --procs 2
python scripts/experiments/runner.py --phase conv --procs 2
python scripts/experiments/runner.py --phase grid --procs 2
python scripts/experiments/runner.py --phase lm --procs 2
python scripts/experiments/runner.py --phase gain --procs 2
\`\`\`

The runner records completed jobs and skips matching completed entries when resumed. Check the runner and configuration files for the exact jobs, seed ranges, and settings associated with each phase. The phases above are examples, not a claim that running only these commands reproduces every run in the 787-run programme.

### 4. Aggregate results

\`\`\`bash
python scripts/experiments/aggregate.py
python scripts/experiments/agg_stage3.py
\`\`\`

### 5. Generate figures

\`\`\`bash
python scripts/generate_figures.py
\`\`\`

### 6. Run tests

\`\`\`bash
pytest tests/
\`\`\`

The test suite includes checks for attention scaling, causal masking, configuration handling, data preparation, head-dimension grids, early stopping, gradient paths, and seed matching.

## Pretrained-model inference evaluations

The repository includes scripts that change the attention scaling divisor at inference time while retaining pretrained model weights.

For GPT-2 and GPT-2-medium:

\`\`\`bash
python scripts/experiments/frontier_gpt2.py \\
  --models gpt2,gpt2-medium \\
  --nseq 160 \\
  --seqlen 128
\`\`\`

For larger supported Hugging Face decoder models, the GPU-oriented script is:

\`\`\`bash
python scripts/frontier/frontier_gpu_1b7b.py --model gpt2-xl --nseq 500 --seqlen 256
\`\`\`

Additional model identifiers and options are documented in the script. Access to gated models may require Hugging Face authentication and acceptance of the model provider's terms.

**Scope note:** the larger-model script provides an evaluation pathway. Its presence does not mean that a full set of 1B–7B model experiments was completed as part of the reported study. The manuscript's reported pretrained-model evaluations should be distinguished from optional runs a user may execute with this script.

## Interpreting the operating-point analysis

The study considers the standard deviation of the raw attention logits before scaling. For a simplified comparison of two divisor coefficients \(b_1\) and \(b_2\), the theoretical analysis relates the divergence between the resulting attention distributions to the raw-logit variance and the number of attention positions.

This relationship is used as an operating-point analysis, not as a universal decision rule with fixed standard-deviation cut-offs. The realised effect should be interpreted using the relevant model, task, sequence length, and empirical diagnostics.

## Figures

The [figures directory](figures/) contains generated figures associated with the experiments. Selected figures include:

- [Experimental architecture](figures/fig_architecture.png)
- [Experimental design](figures/fig_design.png)
- [Main performance comparison](figures/fig_main_performance.png)
- [Paired differences](figures/fig_paired_differences.png)
- [Corrected language-modelling results](figures/fig_lm_corrected.png)
- [Operating-point regime analysis](figures/fig_regime_law.png)
- [Frontier evaluation](figures/fig_frontier.png)

## Publication and supplementary material

The manuscript and its supplementary material are intended for separate publication through the journal. They are deliberately **not included in this GitHub repository**.

Once the journal publication is available, this section can be updated with:
- the final article DOI and journal landing page;
- the journal-hosted supplementary-material link.

No DOI or journal supplementary URL is specified here until the final publication details are confirmed.

## Citation

If you use this code, please cite the associated article after its publication. The final bibliographic reference and DOI will be added to this section when available.

## Data and code availability

The code, experiment configurations, result records, verification outputs, and generated figures are available in this repository. Dataset files are obtained from their respective sources and are not redistributed here. The manuscript and supplementary material will be made available separately through the journal's publication platform.

## License

This repository is released under the [MIT License](LICENSE).
