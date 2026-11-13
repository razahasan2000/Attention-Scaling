# Attention-Scaling — v2 Experiment Package

This package completes the four v2 work streams requested for the study:

1. **Complete the planned matrix** (all remaining runs of the preregistered grid).
2. **Fix and re-run language modelling** — standard causal (autoregressive) masking on WikiText-2, full 25 matched seeds per variant.
3. **Head-dimension grid** — d_k ∈ {32, 64, 128} with parameter-count invariance.
4. **Train to convergence** — early stopping on validation loss, full learning curves.

> **Where things stand:** every code change is implemented, unit-tested
> (38/38 tests passing) and smoke-validated end-to-end on real WikiText-2
> data in this package. The training matrix itself (~1,300 runs, roughly
> 15–25 GPU-hours or 110–185 CPU-hours) must be executed on your machine —
> the runner below resumes automatically, so you can run it in stages.

---

## 1. What was fixed (and why the v1 LM numbers were invalid)

### 1.1 The language-modelling leakage bug (critical)

`src/models/transformer_encoder.py` built only a **key-side padding mask**
`(batch, 1, 1, seq_len)` — there was **no causal mask**. Under next-token
training, position *i* could attend to tokens *i+1 … n* and simply copy the
answer. This is why the v1 records show WikiText-2 perplexities of ~1.09:
near-perfect copying, not language modelling.

**v2 fix:** `TransformerEncoder` now defaults to a standard lower-triangular
causal mask for `task_type="language_modelling"`, ANDed with the padding mask
(`forward(..., causal=None)`; overridable via `set_causal()` /
`causal=` argument). Classification keeps bidirectional attention.

### 1.2 The clamp/mask ordering bug

`src/attention/scaling.py` applied `masked_fill(-inf)` **before** the
±50 logit clamp, so masked positions leaked weight `exp(-50) ≈ 2e-22`.
**v2 fix:** clamp first, then mask → masked positions get **exactly zero**
weight. Verified by an exact assertion in `tests/test_causal_mask.py`.

### 1.3 Missing data pipeline

`src/data/` was absent from the repository (the runner could not even
import). `src/data/prepare.py` is reconstructed to the manuscript's S2 spec
and the pinned API in `tests/test_data_prepare.py`: regex `[a-z0-9']+`
lowercase tokenisation for classification, whitespace + `<eos>` for LM,
vocab ≤ 18,855 at min-frequency 2 from training data only, seeded
10,000-example training subsets, SST-2 official GLUE validation split,
label maps preserved, right-padding to 128, LM stream chopped into
128-token windows. Verified on the real WikiText-2 download
(vocab 18,738; 4,403/1,649/1,859 train/val/test windows).

### 1.4 Run-ID collision (legacy flaw)

v1 used the same run_id for 2-epoch core runs and 25-epoch extended runs.
v2 uses explicit, collision-free ids (`_ext25`, `_lm25k_causal`, `_dk{N}`,
`_conv`, `_len{L}`, `_size{S}`, `_abl5`) and stores a `grid` field in every
record, so resume and analysis are unambiguous.

## 2. What was added

| Component | File | Notes |
|---|---|---|
| Causal masking | `src/models/transformer_encoder.py` | default for LM; tests pin zero future influence |
| Exact masking | `src/attention/scaling.py` | clamp-before-mask; NaN guard |
| Data pipeline | `src/data/prepare.py` | was missing; matches S2 spec |
| Early stopping | `src/training/trainer.py` | patience on val loss, best-state restore, `early_stop_info` in every record; learning curves in `training_history` |
| Convergence budget | `configs/training_budget_config.yaml` | max 30 epochs, patience 5, checkpoints 2/5/10/15/20/25/30 |
| LR compensation | `src/training/trainer.py` | per-group LR (W_q, W_k × √d_k) for the dk_lrcomp ablation arms |
| Master runner | `scripts/run_all_v2.py` | 7 stages, dry-run manifest, resume, per-stage selection |
| Paper finalisation | `scripts/finalise_paper.py` | completion audit, paired stats (t, Wilcoxon, Holm, d_z), manifest-table removal, figure regeneration hook |
| Tests | `tests/test_causal_mask.py`, `tests/test_early_stopping.py`, `tests/test_dk_grid.py` | 13 new tests, incl. a trained leak test |
| Smoke | `scripts/smoke_v2.py` | end-to-end on synthetic data |

## 3. The v2 matrix (1,303 runs)

| Stage | Design | Runs |
|---|---|---|
| `extended25` | 3 variants × 25 seeds × {sst2, imdb}, 25 epochs | 150 |
| `lm_causal` | **causal** LM, 3 variants × 25 matched seeds, 25k steps | 75 |
| `dk32` / `dk64` / `dk128` | 3 variants × 25 seeds × 3 datasets, d_model fixed at 128 (n_heads = 4/2/1) | 675 |
| `converged` | core matrix to convergence (early stop patience 5) | 225 |
| `seqlen` | lengths {64, 256} × 3 variants × 3 seeds × 3 datasets | 54 |
| `modelsize` | {small, medium} × 3 variants × 3 seeds × 3 datasets | 54 |
| `ablations` | E1–E4 completion (multiscale_v1 / dk_lrcomp, 5 epochs) | 70 |
| **Total** | | **1,303** |

**Head-dimension design note:** d_model stays 128, so Q/K/V/O are always
`Linear(128, 128)` and the parameter count is **identical** across d_k ∈
{32, 64, 128} (asserted by `tests/test_dk_grid.py`). Only the per-head split
— and therefore d_k, the quantity in the 2017 variance argument — changes.
The d_k grid runs at the 2-epoch core operating point by default (directly
comparable with the existing AG News result); pass `--protocol conv` to run
it under the convergence protocol instead.

**Power note:** the v1 LM arm had 25/75 runs under a leaky objective; the
`lm_causal` stage re-runs all 75 runs under the corrected objective so the
paired tests have full 25-seed power.

## 4. How to run

```bash
pip install -r requirements.txt          # adds nothing new; torch >= 2.1

# 0. Inspect the plan (writes results/v2_manifest.json, runs nothing)
python scripts/run_all_v2.py --dry-run

# 1. Tests (38) + smoke
python -m pytest tests/ -q
python scripts/smoke_v2.py

# 2. Run stages (each resumes automatically; re-run to continue)
python scripts/run_all_v2.py --stages extended25 lm_causal
python scripts/run_all_v2.py --stages dkgrid              # 2-epoch protocol
python scripts/run_all_v2.py --stages dkgrid --protocol conv
python scripts/run_all_v2.py --stages converged
python scripts/run_all_v2.py --stages seqlen modelsize ablations

# Useful flags: --device cpu|cuda|auto   --variants standard dk
#               --limit 5                # smoke-scale the real matrix
```

Records land in `results/per_seed/{run_id}.json` (original schema **plus**
`grid`, `arm`, `d_k`, `n_heads`, `protocol`, `early_stop_info`), so the
existing `scripts/analyse_results.py` keeps working.

## 5. Compute estimates

Measured on this package's 2-core CPU reference (298 ms/step, batch 64 ×
seq 128); a modern single GPU (RTX 3060-class or better) is typically
8–15× faster on this model size.

| Stage | Runs | Per run | Stage total (CPU ref) | Stage total (GPU est.) |
|---|---|---|---|---|
| extended25 | 150 | ~20 min (25 ep) | ~50 h | ~4–6 h |
| lm_causal | 75 | ~31 min (25k steps) | ~39 h | ~3–5 h |
| dkgrid (core) | 675 | ~1.6 min | ~18 h | ~1.5–2.5 h |
| dkgrid (conv) | 675 | ~8 min avg | ~90 h | ~7–11 h |
| converged | 225 | ~8 min avg | ~30 h | ~2.5–4 h |
| seqlen + modelsize | 108 | 1.6–7 min | ~5 h | ~0.5 h |
| ablations | 70 | ~4 min | ~5 h | ~0.5 h |
| **Total** | **1,303** | | **≈ 110–185 h** | **≈ 15–25 h** |

Recommendation: run `dkgrid` and `converged` on GPU; everything else fits in
a couple of GPU-evenings. The runner is resumable at single-run granularity.

## 6. Finalising the paper once runs complete

```bash
python scripts/finalise_paper.py --results-dir results --tex-dir manuscript
# review results/statistical_tests/v2_completion.json + v2_summary.json, then:
python scripts/finalise_paper.py --remove-manifest --tex-dir manuscript
python scripts/generate_figures.py results/statistical_tests figures
```

- `finalise_paper.py` recomputes paired t / Wilcoxon / Holm / d_z for every
  grid and writes `v2_summary.json`.
- `--remove-manifest` deletes the "planned versus completed" table (marked
  by `%%MANIFEST-SECTION-START/END` in `sec_results.tex`), as requested for
  the final text.
- The script never invents numbers: prose updates take values from
  `v2_summary.json`, then recompile the manuscript.

## 7. Validation evidence (already executed in this package)

- `python -m pytest tests/ -q` → **38 passed** (13 new tests included).
- `scripts/smoke_v2.py` → all four v2 paths pass; the LM control reproduces
  the v1 pathology and the fix removes it:
  - bidirectional control (v1 behaviour): test perplexity **1.01**
  - causal (v2): perplexity **746** on random synthetic text (≈ uniform
    baseline ln(500) → e^6.2 ≈ 495, as expected for unpredictable targets).
- Real WikiText-2 pipeline: vocab 18,738, 4,403 train windows; 120-step
  causal run → val perplexity 2,702 → 1,902 (healthy early LM curve).

## 8. Transparency

`V2_CHANGES.patch` (in this package) contains the complete diff of every
change against the upstream repository commit `4e8bbfe`.
