# Supplementary Materials — Attention Scaling Study

Companion to `manuscript/manuscript_final.md`. All numbers trace to per-seed records in
`results/results/per_seed/` (525 JSON files).

---

## S1. Per-seed record schema

Each run record (e.g., `sst2_temp_tau0.5_seed_1001_2epoch.json`) contains:

| Field | Meaning |
|---|---|
| `run_id` | `{dataset}_{variant}[_tau{τ}]_seed_{seed}_{budget}` (τ omitted for the τ=2.0 core-grid arms) |
| `dataset`, `task`, `variant`, `seed` | Experimental factors |
| `config` | d_model, n_heads, n_layers, d_ff, dropout, scaling, τ, max_seq_len, vocab_size |
| `training_history` | Per-epoch (cls) / per-5k-step (LM): train_loss, val_loss, val_accuracy, val_macro_f1, epoch_time_s, examples/s |
| `test_accuracy`, `test_macro_f1`, `test_loss` / `test_perplexity` | Final-epoch evaluation |
| `checkpoint_history` | Paths of saved checkpoints |
| `status`, `timestamp` | Completion marker and UTC timestamp |

## S2. Matched-design guarantees

1. **Initialisation.** Before each run: `random.seed(s)`, `np.random.seed(s)`, `torch.manual_seed(s)`,
   `torch.cuda.manual_seed_all(s)`, then `create_model(variant, ...)`. Scaling variants add no
   parameters and no RNG consumption → bit-identical init across variants per seed.
   Verified by `tests/test_seed_match.py` (`torch.allclose` equality assertions).
2. **Data order.** Epoch permutations drawn from `torch.Generator(cpu).manual_seed(s)`.
3. **Data identity.** Prepared datasets are cached once (`results/data/{dataset}/data.npz`) and
   reused across all variants; tokenisation is deterministic (fixed seed 42 for subsampling).
4. **τ is a forward-pass constant.** `temp` applies `logits / τ` after the base `1/√d_k`;
   no parameters change → the τ sweep is a pure treatment.
5. **Logit clamp** [−50, 50] and padding masks identical across all arms.

## S3. Statistical pipeline (`scripts/analyse_completion.py`)

- Records grouped by arm: `standard`, `dk`, `temp_tau{0.5,1.0,1.5,2.0}` (τ parsed from record config).
- Pairing: strict inner join on seed (never positional).
- Tests per comparison: paired t; exact sign-flip permutation (10,000 draws, seed 42);
  Wilcoxon signed-rank; exact sign test; Shapiro–Wilk on paired differences.
- p-value floor: permutation p reported as (b+1)/(n+1) (Phipson & Smyth, 2010).
- Multiplicity: Holm step-down **with monotonicity (running-max) step** across all
  comparisons-vs-standard within each dataset (family of 2 for core grid, 4 with the τ sweep).
- Effect size: Cohen's d_z = mean(diff)/sd(diff), ddof=1. 95% t-CIs on paired mean differences.

## S4. Reproducibility checklist

| Item | Status | Where |
|---|---|---|
| Preregistered configuration (YAML) | ✅ | `configs/` (model, experiment, seed, budget, statistical) |
| Bit-identical init across variants | ✅ unit-tested | `tests/test_seed_match.py` |
| Data-order matching per seed | ✅ | `src/training/trainer.py` (seeded Generator) |
| All runs logged as per-seed JSON | ✅ 525 records | `results/results/per_seed/` |
| Run ledger (crash-resumable) | ✅ | `results/results/completion_ledger.json` |
| Statistical outputs (all tests) | ✅ | `results/results/completion_statistical_tests/` |
| Publication figures from data | ✅ 7 figures (PNG+PDF) | `figures/`, `scripts/generate_figures_real.py` |
| Random seeds recorded | ✅ per record | per-seed JSON |
| Environment documented | ✅ | records' config + reports |
| Placeholder/illustrative figures removed | ✅ (3 deleted) | `figures/` |
| Phantom claims replaced with executed runs | ✅ | τ-ablation 225 runs |
| Failed/invalid experiments excluded & disclosed | ✅ | §8 of manuscript (E1–E4, extended grid) |
| Deviations from preregistration disclosed | ✅ | manuscript §8 (AG News τ post-hoc; Holm family) |
| Reference register with verification URLs | ✅ 44 verified | `../reference_verification_register.md` |

## S5. Deviations and known issues (disclosure)

1. **AG News τ-sweep** executed as a post-hoc extension (preregistered scope: SST-2, IMDB).
2. **LR scheduler window** mis-set (epochs×100) in the shared trainer; applies identically to all
   arms → paired comparisons valid; absolute numbers are budget-specific.
3. **SST-2 test partition** = GLUE dev split (872 examples); official test labels are unlabeled.
4. **Test evaluation** uses final-epoch weights (best checkpoints saved, not reloaded).
5. **Extended 25-epoch grid**: only seeds 1004–1005 (12 runs) genuinely trained 25 epochs; the
   remaining `*_25epoch.json` core files are 2-epoch runs with a hardcoded run-ID suffix. Excluded
   from conclusions.
6. **E1–E4 ablations**: 70 records exist but the intended treatments (multiscale per-head scaling,
   LR compensation) were never applied due to an implementation defect. Excluded from conclusions;
   re-implementation recommended before any use.
7. **Historical (pilot-era) artifacts** with different configurations are superseded and excluded.

## S6. Provenance of every headline number

| Claim | Source records | Statistic |
|---|---|---|
| AG News std 91.23±0.27 | 25 × `agnews_standard_seed_*.json` | `completion_statistical_tests/agnews_*.json` |
| AG News std > dk (dz 1.52, p_holm 4.0e-7) | 25 seed-pairs | same |
| SST-2 τ=0.5 81.08±0.55 (perm 0.010) | 25 × `sst2_temp_tau0.5_seed_*.json` | `sst2_*.json` |
| IMDB τ=0.5 82.29±0.21 (perm <1e-4) | 25 × `imdb_temp_tau0.5_seed_*.json` | `imdb_*.json` |
| WT2 dk 1.0811±0.0032 (p_holm 8.1e-8) | 25 × `wikitext2_dk_seed_*.json` | `wikitext2_*.json` |
| WT2 τ=2 1.0860±0.0050 (p_holm 5.8e-5) | 25 × `wikitext2_temp_seed_*.json` | `wikitext2_*.json` |
| Efficiency (<2%, identical params) | training_history epoch_time_s | `fig_efficiency.png` |

## S7. Rebuild instructions

```bash
cd attention_scaling_fresh_study
# statistics
python scripts/analyse_completion.py --results-dir results/results/per_seed \
       --output-dir results/results/completion_statistical_tests
# figures
python scripts/generate_figures_real.py --results-dir results/results/per_seed --output-dir figures
```
