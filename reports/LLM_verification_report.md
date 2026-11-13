# Independent Verification Report

**Date:** 2026-11-13
**Scope:** full v4 package — manuscript (PDF + DOCX), supplementary material (PDF + DOCX),
figures, `results/` data, experiment code, and README claims.
**Method:** two independent layers, neither of which shared context with the work that produced
the documents:

1. **Deterministic recomputation** (`scripts/verify_numbers.py`) — every headline statistic is
   independently recomputed from the raw per-seed records and JSON aggregates and compared with
   the numbers printed in the manuscript, the supplementary material, and the README.
2. **Fresh-context LLM review** — eight independent reviews by a separate LLM instance
   (`glm-4-plus` via API, no shared conversation state, reviewer persona, strict-JSON findings):
   manuscript in three section-aligned chunks, supplementary in two chunks, a standalone
   mathematical verification of the operating-point power law, a code-level review of the
   scaling manipulation, and a README/abstract/conclusion consistency cross-check.
   Raw reviewer outputs are archived in `reports/llm_review/`.

## 1. Deterministic checks — 59/59 PASS

| Category | Checks | Result |
|---|---|---|
| Run counts (467 JSONL records; Stage 2 = 339; Stage 3 = 128 + 6 frontier evals = 134; 314 + 339 + 134 = 787) | 4 | PASS |
| Stage-2 paired statistics recomputed from per-seed data (core AG News, LM; dk & temp arms) | 4 | PASS |
| Operating-point law thresholds recomputed from the formula (σ = 0.9795 → 0.98; 3.0974 → 3.1; Δβ = 0.145527) | 5 | PASS |
| Independent Monte-Carlo validation of the law itself (20,000 replicates at σ = 2: KL 0.04151 vs law 0.04169) | 1 | PASS |
| Frontier percentages recomputed (GPT-2 +9,253.21 % / +54.46 %; medium +3,990.02 % / +26.54 %) and values matched to documents | 16 | PASS |
| Entropy ranges (pooled per-layer H: 1.23–3.46 → 3.61–3.87 nats vs claim 1.2–3.5 → 3.6–3.9) | 2 | PASS |
| Layer-mean σ_raw (GPT-2 62.6, medium 44.8 vs claim 45–63, qualified as layer means) | 2 | PASS |
| Gain sweep (KL 0.0009 → 0.6418; acc 76.812 vs 78.212; d_z = −1.759; σ 0.3262 → 9.2123) | 6 | PASS |
| Regime-restored training (σ = 1.1104 at d_k = 32; max σ = 4.588 at d_k = 128) | 2 | PASS |
| QK-Norm bridge (σ_raw pinned 5.94–6.15 across arms; σ_post = 1.051 under standard divisor; acc spread 0.119 pp) | 3 | PASS |
| LM headline (ppl 872.80 ≈ 873; paired |d_z| ≤ 0.01) | 2 | PASS |
| Causal-mask verification suite (V1–V5 invariants) | 1 | PASS |
| Documents contain headline numbers (incl. comma-formatted 4,791.65 / 1,585.70; math-italic τ = 0.5) | 9 | PASS |
| LaTeX integrity (16 graphics files exist; 32 cite keys all resolve in refs.bib) | 2 | PASS |

## 2. Independent mathematical verification

A standalone verification call (fresh context) was asked to derive the power law itself and
check the bite thresholds:

- **Derivation:** confirmed correct to second order (Taylor expansion of the softmax KL around
  equal inverse divisors, Gaussian logit assumption).
- **Thresholds:** recomputed σ = 0.98 (KL = 0.01 nats) and σ = 3.1 (KL = 0.1 nats) at
  d_k = 32, n = 64 — both match the paper exactly.
- **Stated caveats** (second-order validity, logit independence, large divisor gaps) are already
  acknowledged in Supplementary S17.

## 3. Code-level verification

Claims checked against `src/attention/scaling.py`, `src/models/transformer_encoder.py`,
`scripts/experiments/common.py`, `scripts/experiments/frontier_gpt2.py`:

| Claim | Verdict | Evidence |
|---|---|---|
| Only the divisor differs across the three arms | **Supported** | `common.py: VARIANTS = ("standard", "dk", "temp")`; the extra code paths (`multiscale_v1`, `dk_lrcomp`) are the documented Stage-1 ablation arms E1–E4, never used in the core comparison |
| Bit-identical matched initialization | **Supported** | `build_matched()` clones one reference state dict into every arm; cross-variant bit-identity is asserted in `tests/test_seed_match.py`; `strict=False` is deliberate and documented (Stage-3 QK-Norm affine params keep deterministic defaults) |
| Frontier probe changes only the divisor at inference | **Supported** | frozen weights, identical token stream; `scalings(d_head)` is the only intervention; raw logits are bit-identical across variants by construction |
| Causal mask prevents future attention | **Supported** | mask construction correct in encoder and experiment suite; V1–V5 assertions in `results/mask_verification.json` all hold |

## 4. Findings triage

Sixteen reviewer findings + one cross-document catch were triaged. **Nine real issues were fixed;
the remainder were dismissed with evidence (reviewer misreads of paired statistics, daggered
artifact tables, multi-panel tables, or text-extraction artifacts).**

### Fixed (documents regenerated and re-verified)

| # | Severity | Location | Issue | Fix |
|---|---|---|---|---|
| 1 | critical | Conclusions (§7) | `p_Holm < 0.05` contradicted the abstract's and Table 5's `p_Holm < 0.001` (ground truth: `study_data.PAIRED` agnews p_holm `<0.001`) | Conclusions now read `p_Holm < 0.001` (LaTeX + DOCX) |
| 2 | major | §4.12 / Fig. 15 discussion | Per-row entropy claim "1.4–2.9 → 3.6–3.9 nats"; true pooled per-layer range under the standard divisor is 1.23–3.46 | Corrected to "1.2–3.5 → 3.6–3.9" (manuscript, README) |
| 3 | major | §4.12 bridge paragraph | "QK-Norm pins the operating point at σ_post ≈ 1.05 regardless of the divisor" — data show σ_post = 1.051 / 0.192 / 0.538 under the three divisors; what is pinned is σ_raw ≈ 6.0 | Reworded: σ_raw ≈ 6.0 pinned regardless of divisor (σ_post ≈ 1.05 under the standard divisor) — manuscript, README, supplementary caption |
| 4 | minor | §4.11 / §4.12 | "σ_raw ≈ 63 and 45" not marked as layer means | Qualified "layer means" (manuscript, DOCX, README) |
| 5 | minor | §1 contribution (iv), §4.9 | Gradient ratios presented as "matching theory" although 1/d_k measures 0.26 vs theory 0.18 | Reworded to ordering- and magnitude-level confirmation, with the fixed-attention-map assumption stated as the source of the residual excess |
| 6 | minor | §3.6 | "Preregistered" used without definition | Added: grids/seeds/analysis fixed and committed before Stage-2 data collection; no external registry |
| 7 | minor | §3.9 | Mask-verification claim lacked a pointer | Added "(five exact float32 assertions plus deliberate leak probes; Supplementary S15)" |
| 8 | minor | Supplementary Table 18 | Raw p-values (n = 3) without a correction note | Caption now states: two-sided paired t tests, uncorrected (exploratory, n = 3 per arm) |
| 9 | minor | README | "binary sentiment" unnamed; σ range unqualified | Now "(SST-2, IMDB)"; "layer means" added |

### Dismissed with evidence

| Reviewer finding | Why dismissed |
|---|---|
| "d_z = 1.52 inconsistent with +0.31 pp" | Paired effect size: sd of within-seed differences = 0.204 pp; 0.31/0.204 = 1.52. Large d_z with small mean diffs is the expected signature of a well-matched paired design |
| "Holm correction unclear/invalid" | Family is defined (two comparisons per dataset, step-down, α = 0.05); adjusted p-values are reported in the tables (p_Holm columns) |
| "Temperature divisor s = τ·d_k = 64" | Text-extraction artifact: the PDF renders s = τ√32 ≈ 11.3137, which is correct |
| "LM n = 15/7/3 inconsistent with 25 seeds" | Those are the daggered, superseded v1 leaky-mask runs; the footnote points to the corrected 25-seed protocol (S14) |
| "Stage-2 count 339 vs manifest 288" | The manifest's Stage-2 rows sum to 339 (75+75+108+30+24+27); the reviewer omitted two rows |
| "Duplicate seeds in Table 13" | Per-seed tables list each seed once per row across variant columns |
| "Table 22: measured KL 0.0010 vs law 0.0039 at σ = 0.5" | Cross-row misread: the actual row is KL 0.0026 vs law 0.00261 (0.4 % agreement) |
| "Table 25 does not state seed count" | Caption states "3 matched seeds, mean over seeds" |
| "Perplexities 1.08–1.09 are impossible" | This is the paper's own diagnosed artifact (mask leak), retained and explained, not an unnoticed error |
| "Extra scaling variants violate claim (a)" | Unused-in-core ablation arms, documented in supplementary S8 |
| "`tril` vs `triu` style point" | Cosmetic; behaviour identical |

## 5. Outcome

All four documents (manuscript PDF/DOCX, supplementary PDF/DOCX) were regenerated after the
fixes, the deterministic suite was re-run to **59/59 PASS** against the regenerated PDFs, and
rendered-page inspection confirmed clean layout. The verification tooling ships with the repo
(`scripts/verify_numbers.py`; raw reviewer outputs in `reports/llm_review/`) so the checks can
be re-run by any reader.
