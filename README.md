# Attention-Scaling

**A seed-matched empirical comparison of scaling functions in Transformer self-attention — with a quantitative operating-point law and a frontier-scale probe.**

Three parameter-free scaling functions for softmax self-attention are compared under a seed-matched paired design:

| Variant | Attention | Divisor $s$ (logits divided by $s$) |
|---|---|---|
| **Standard** | $\mathrm{softmax}(QK^\top / \sqrt{d_k})\,V$ | $\sqrt{d_k} = 5.66$ ($d_k = 32$) |
| **$1/d_k$** | $\mathrm{softmax}(QK^\top / d_k)\,V$ | $d_k = 32$ |
| **Temperature** | $\mathrm{softmax}(QK^\top / (\tau\sqrt{d_k}))\,V$, $\tau = 2$ | $2\sqrt{d_k} = 11.31$ |

Every seed initialises all variants from a **bit-identical state dict**, feeds identical batches in identical order, and shares the optimizer configuration — so the paired per-seed difference isolates the scaling function as the only manipulated factor.

## Headline findings (787 completed runs, three stages)

1. **Stage 1 — GPU scale (314 runs).** Standard scaling significantly beats both alternatives on 4-class AG News ($+0.31$ pp over $1/d_k$, $d_z = 1.52$, Holm $p < 0.001$) and is indistinguishable on binary sentiment (SST-2, IMDB). A sharper temperature ($\tau = 0.5$) looks best on sentiment.
2. **Stage 2 — replica completion (339 runs, CPU).** The AG News effect does **not** reproduce at a fivefold-reduced operating point; the $d_k \in \{32,64,128\}$ grid, train-to-convergence, and a corrected causal-mask WikiText-2 grid (25 seeds/arm, ppl $\approx 873$) are all variant-invariant. The original near-1 perplexities are diagnosed and reproduced-in-direction as a mask-leak artifact.
3. **Stage 3 — the operating-point programme (134 runs).** The nulls are regime facts, quantified by a power law:

$$\mathrm{KL}\big(P_{b_1}\,\|\,P_{b_2}\big) \;\approx\; \tfrac12 (b_1-b_2)^2\,\sigma^2\Bigl(1-\tfrac1n\Bigr)$$

   where $\sigma$ is the **raw pre-division logit SD** and $b_i = 1/c_i$ the inverse divisors. Validated by Monte Carlo; bite thresholds at $d_k{=}32$, $n{=}64$: KL $=0.01$ nats at $\sigma\approx0.98$, KL $=0.1$ nats at $\sigma\approx3.1$.
   - **Logit-gain sweep** (72 runs): forcing $\sigma$ from $0.33$ to $9.21$ moves the measured divisor contrast from $0.0009$ to $0.64$ nats and makes the 2017 default the *worst* variant at $g=32$ ($76.81\%$ vs $78.21\%$, $d_z=-1.76$).
   - **Regime-restored training** (18 runs): $12\times$ the step budget leaves $\sigma$ at $1.11$ for $d_k{=}32$ (inertia is real), but $d_k{=}128$ reaches $\sigma\approx5$ and the Stage-1 ordering re-emerges in direction.
   - **Frontier stabilisers** (20 runs): QK-Norm pins $\sigma_{\text{raw}}\approx6.0$ ($\sigma_{\text{post}}\approx1.05$ under the standard divisor) and re-equalises the variants; Gemma-2-style soft-capping bounds the logits but not the contrast.
   - **Pretrained decoder probe** (6 evaluations, no retraining): swapping *only* the divisor at inference in GPT-2 / GPT-2-medium ($\sigma_{\text{raw}}\approx45$–$63$, layer means):

     | Model | Standard | $\tau=2$ | $1/d_k$ |
     |---|---|---|---|
     | GPT-2 (124M) | 51.23 | 79.13 ($+54.5\%$) | **4,791.65 ($+9{,}253\%$)** |
     | GPT-2-medium (355M) | 38.77 | 49.06 ($+26.5\%$) | **1,585.70 ($+3{,}990\%$)** |

     Attention under $1/d_k$ collapses towards uniformity (per-layer attention entropy $1.2$–$3.5 \to 3.6$–$3.9$ nats). The 1B–7B extension ships as a ready-to-run GPU script (below).

**Bottom line.** $\sqrt{d_k}$ remains a sound default; the divisor's effect is an *operating-point-dependent* quantity — inert below $\sigma\approx1$, decisive far above it — and $\sigma$ is measurable in one forward pass.

## Repository structure

```
Attention-Scaling/
├── configs/                  # YAML experiment configurations (original protocol)
├── src/                      # Model / attention variants / trainer / evaluator
│   ├── attention/scaling.py  #   the scaling-function manipulation (one line)
│   ├── models/               #   pre-LN encoder implementation
│   ├── training/, evaluation/, analysis/
├── scripts/
│   ├── run_experiments.py    # Original Stage-1 runner (GPU protocol)
│   ├── run_ablations*.py, run_temp_ablation.py, run_model_size.py
│   ├── analyse_results.py, generate_figures.py, validate.py
│   ├── experiments/          # Self-contained Stage-2/3 suite (CPU-only, resume-safe)
│   │   ├── common.py         #   model, matched-seed infra, operating-point probe
│   │   ├── prepare_data.py   #   replica subsets + vocabularies (4 datasets)
│   │   ├── prepare_regime.py #   6,000-example regime-restored AG News
│   │   ├── runner.py         #   phases: lm, core, grid, conv, abl, lmc, lmleak,
│   │   │                     #           gain, bridge, regime, regime128, lmgain
│   │   ├── aggregate.py      #   Stage-2 paired statistics
│   │   ├── agg_stage3.py     #   Stage-3 aggregation (gain/bridge/lmgain/regime)
│   │   ├── regime_law.py     #   power-law Monte Carlo validation
│   │   ├── frontier_gpt2.py  #   pretrained-decoder divisor swap (CPU)
│   │   ├── verify_mask.py    #   V1–V5 causal-mask verification suite
│   │   └── diagnostics.py, clean_results.py, gen_perseed_tex.py
│   └── frontier/
│       └── frontier_gpu_1b7b.py   # 1B–7B GPU divisor-swap evaluation
├── tests/                    # Unit tests: seed matching, mask, scaling, gradients
├── results/                  # Raw + aggregated experimental data
│   ├── results.jsonl         #   467 append-only run records (Stages 2–3)
│   ├── aggregate.json        #   Stage-2 statistics used in the paper
│   ├── stage3_theory.json    #   power-law MC grid, entropy table, thresholds
│   ├── stage3_operatingpoint.json  # gain/bridge/lmgain/regime aggregates
│   ├── frontier_gpt2.json    #   GPT-2 / GPT-2-medium probe results
│   ├── diagnostics.json      #   attention/gradient diagnostics
│   ├── mask_verification.json
│   └── verification_checks.json    # output of scripts/verify_numbers.py
├── figures/                  # All 17 figures (300 DPI PNG, regenerated by scripts)
├── paper/                    # CURRENT manuscript (v4, three-stage programme)
│   ├── Attention-Scaling_Manuscript.pdf / .docx
│   ├── Attention-Scaling_Supplementary.pdf / .docx
│   ├── latex/                #   main.tex + sec_*.tex + supplementary.tex + refs.bib
│   └── source/               #   study_data.py (single source of truth), make_figures.py,
│                             #   docx generators, diagram sources
├── manuscript/, supplementary/  # Historical v1 documents (superseded by paper/)
├── reports/, artifacts/      # Historical execution / sync reports
├── README_V2.md              # Stage-2 completion notes (v3 experiment package)
├── requirements.txt
└── LICENSE                   # MIT
```

## Verification

The package ships with an independent verification layer (see
`reports/LLM_verification_report.md` for the full report):

- **Deterministic recomputation** — `scripts/verify_numbers.py` re-derives every headline
  statistic (run counts, paired statistics, law thresholds, frontier percentages, gain/regime/
  bridge quantities) from `results/` and cross-checks them against the rendered PDFs and this
  README: **58/58 checks pass**. Re-run with `python3 scripts/verify_numbers.py`
  (needs `pdftotext`; writes `results/verification_checks.json`).
- **Independent LLM review** — eight fresh-context reviews (manuscript ×3, supplementary ×2,
  standalone mathematical verification of the power law, code review of the scaling
  manipulation, README/abstract/conclusion consistency). Nine real issues were found and fixed
  (documents regenerated and re-verified); the remaining findings were dismissed with evidence.
  Raw reviewer outputs: `reports/llm_review/`.

## Quickstart (replica experiments, CPU-only)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scipy pandas pyarrow numpy

python scripts/experiments/prepare_data.py     # downloads + tokenises 4 datasets
python scripts/experiments/verify_mask.py      # causal-mask assertions (V1–V5)
python scripts/experiments/runner.py --phase core   --procs 2   # AG News, 25 seeds × 3
python scripts/experiments/runner.py --phase lm     --procs 2   # corrected causal LM
python scripts/experiments/runner.py --phase gain   --procs 2   # Stage-3 logit-gain sweep
python scripts/experiments/runner.py --phase regime --procs 2   # regime-restored, probes
python scripts/experiments/aggregate.py                         # Stage-2 statistics
python scripts/experiments/agg_stage3.py                        # Stage-3 statistics
```

The runner is **resume-safe**: completed `(phase, dataset, variant, d_k, seed, spec)`
keys in `results/results.jsonl` are skipped, so grids can be executed in stages.
Stage 3 also provides `regime128`, `bridge`, and `lmgain` phases.

## Frontier probe

CPU (GPT-2 124M / 355M, reproduces the paper's Table 12):

```bash
pip install transformers pandas pyarrow
python scripts/experiments/frontier_gpt2.py --models gpt2,gpt2-medium --nseq 160 --seqlen 128
```

GPU (1B–7B decoders, identical intervention + per-layer diagnostics):

```bash
python scripts/frontier/frontier_gpu_1b7b.py --model gpt2-xl --nseq 500 --seqlen 256
for M in EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b; do
  python scripts/frontier/frontier_gpu_1b7b.py --model $M --nseq 500 --seqlen 256
done
python scripts/frontier/frontier_gpu_1b7b.py --model meta-llama/Meta-Llama-3.2-8B --dtype bfloat16
```

Supports any HuggingFace decoder exposing a per-layer scaling attribute
(Llama / Qwen2 / Mistral / GPT-2), `norm_factor` (GPT-NeoX / Pythia), or
`scale_attn` (GPT-J); uses eager attention so the swap is exact.

## The single source of truth

Every number in the manuscript, figures, and Word versions is generated from
`paper/source/study_data.py` (mirrored to `study_data.json`). Figures are
regenerated with `paper/source/make_figures.py`; the LaTeX sources in
`paper/latex/` compile with [Tectonic](https://tectonic-typesetting.github.io)
(`tectonic main.tex`); the Word versions are built by
`paper/source/docx_part1.js` + `docx_part2.js` (manuscript) and
`docx_supplementary.js` (supplementary) with the `docx` npm package.

## Documentation

- `paper/Attention-Scaling_Manuscript.pdf` — the journal manuscript (26 pp, 13 figures/tables + Stage-3 programme)
- `paper/Attention-Scaling_Supplementary.pdf` — supplementary material S1–S18 (13 pp), including the power-law derivation (S17) and Stage-3 records (S18)
- `README_V2.md` — Stage-2 completion package notes
- `reports/` — execution and reproducibility reports from earlier stages

## Manifest (all completed runs)

| Grid | Design | Stage | Runs |
|---|---|---|---|
| Core classification | 3 variants × 25 seeds × 3 datasets | 1 | 225 |
| Temperature ablation | 4 τ × 3 seeds × 2 datasets | 1 | 24 |
| Extended budget | 3 variants × 5 seeds × 2 datasets, 25 epochs | 1 | 30 |
| Core LM (v1 protocol) | 3 variants, 25,000 steps | 1 | 25 |
| Focused ablations E1–E4 | single-seed arms, 5 epochs | 1 | 10 |
| Replica core classification | 3 variants × 25 seeds, AG News | 2 | 75 |
| Replica LM (corrected causal mask) | 3 variants × 25 seeds | 2 | 75 |
| Head-dimension grid | $d_k \in \{32,64,128\}$ × 3 variants × 12 seeds | 2 | 108 |
| Train-to-convergence | 3 variants × 5 seeds × 2 datasets | 2 | 30 |
| Replica ablations E1–E4 | 4 arms × 3 seeds × 2 datasets | 2 | 24 |
| Mask-artifact probes | leaky vs causal, 3 and 10 epochs | 2 | 27 |
| Logit-gain sweep | g ∈ {1,2,4,8,16,32} × 3 variants × 4 seeds | 3 | 72 |
| Regime-restored AG News | 6,000 train, 8 epochs, probes, $d_k \in \{32,128\}$ | 3 | 18 |
| Frontier-stabiliser bridge | QK-Norm / soft-cap arms × 4 seeds | 3 | 20 |
| LM logit gain | g ∈ {1,4} × 3 variants × 3 seeds, seq 128 | 3 | 18 |
| Frontier divisor swap | GPT-2 + GPT-2-medium, inference-time | 3 | 6 |
| **Total** | | | **787** |

## Citation

If you use this codebase or the results, please cite the manuscript in
`paper/` (preprint, 2026): *A Seed-Matched Empirical Comparison of Scaling
Functions in Transformer Self-Attention*, R. Hasan.

## License

MIT — see [LICENSE](LICENSE).
