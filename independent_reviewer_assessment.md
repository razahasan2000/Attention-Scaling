# Independent Reviewer Assessment — Phase 8

**Manuscript:** "No Universally Optimal Scaling: A Seed-Matched Empirical Study of Attention Logit-Scale Functions in Transformer Self-Attention"
**Reviewer role:** Referee for a reputable ML venue (reviewer guidelines: NeurIPS/TMLR style)
**Assessment date:** 10 November 2026

---

## Summary of submission

The paper reports a seed-matched, fully paired comparison of attention logit-scale functions
(1/√d_k, 1/d_k, temperature-scaled 1/(τ√d_k), τ∈{0.5,1.0,1.5,2.0}) in a small Pre-LN Transformer
across three text-classification tasks and language modelling, totalling 525 runs with 25 matched
seeds per arm and bit-identical initialisation across variants. Headline findings: (i) classification
accuracy is non-increasing in τ; (ii) 1/d_k significantly harms multi-class classification
(d_z=1.52, p_Holm=4.0e-7) but significantly helps language modelling (d_z=1.58, p_Holm=8.1e-8) — a
task-regime reversal; (iii) low-τ scaling reduces seed variance 2–5×; (iv) all variants are
computationally identical. The paper also contributes a reproducibility audit documenting and
repairing phantom ablation claims, mislabelled run identifiers, and inconsistent completion totals.

---

## Evaluation by criterion

### Originality — 4/5 (good)
The task-regime reversal for *static* scale functions under matched conditions is a novel,
well-measured empirical finding. The variance-stabilisation observation (SD 2–5× reduction at low τ)
appears previously unreported for text tasks and is practically useful. Not novel: the temperature
knob itself (Hinton 2015; Guo 2017; Lin 2018; Zhang 2022) and entropy-collapse theory (Zhai 2023;
Hong & Lee 2025) — but the paper positions itself correctly as a measurement study, not a new
mechanism. No over-claiming of a "new attention mechanism" — correct.

### Research gap — 4.5/5
G1–G4 are genuine and clearly argued: no seed-matched paired comparison of scale functions across
task types exists; temperature-response curves for text tasks are unreported; scale-choice effects
on seed variance are unmeasured. The gap identification is specific and defensible.

### Theoretical justification — 3/5 (adequate with caveats)
The mechanistic account (variance → sharpness → entropy; LM prefers preserved entropy; classification
prefers concentrated selection) is coherent and anchored in Zhai et al. (2023) and Hong & Lee (2025),
but it remains interpretive: the study measures *outcomes*, not entropy trajectories. No direct
attention-entropy or logit-variance measurements are collected during training (probes exist in the
codebase — `attention entropy`, `head similarity` helpers — but were not exercised for these runs).

### Experimental design — 4.5/5 (strength)
Seed-matched bit-identical initialisation (unit-tested), strict seed-keyed pairing, matched data
ordering, identical budgets, preregistered statistics, per-seed records. This is the paper's core
strength and follows Bouthillier et al. (2021), Reimers & Gurevych (2017), Musgrave et al. (2020)
faithfully. The τ=2.0 arm reuse from the core grid is legitimate (identical configuration).

### Statistical validity — 4/5
Corrected Holm step-down (with monotonicity), exact permutation tests with the (b+1)/(n+1) floor,
Wilcoxon and sign corroboration, d_z effect sizes, joint reporting of raw/Holm/permutation p-values.
Honest treatment of the SST-2/IMDB τ=0.5 effects (permutation-significant, not Holm-significant) is
commendable. Weaknesses: the Holm family is within-dataset (a defensible but non-unique choice; a
cross-dataset family would test the AG News τ=2 effect to p≈0.054×4 — still reportable as
suggestive); the τ=2 AG News effect crosses nominal thresholds depending on family definition and
the manuscript says so, which is appropriate.

### Dataset suitability — 3.5/5
AG News/SST-2/IMDB are standard and appropriate; SST-2's test partition being the GLUE dev split
(872 examples) is disclosed but limits generality of that row (a GLUE-dev artifact). WikiText-2 at
this model scale yields perplexities near 1.09 — far from leaderboard-relevant absolutes; the paper
discloses this and uses it only for *paired* comparisons, which is the right call, but reviewers may
still object that the LM setting is weak (2-layer/128-d, 25k steps).

### Reproducibility — 5/5 (exceptional strength)
525 per-seed JSON records, run ledger, rebuild instructions, unit tests for the matched-design
invariant, disclosed deviations, reference register with verification URLs, and an honest audit of
the study's own earlier phantom claims. This is the most reproducible artifact I have reviewed in
this cycle.

### Baseline selection — 3.5/5
The three scale functions are the right treatment set, but there is no external anchor (e.g., learned
per-head temperature as in Σ-reparameterisation, or SACT) against which the static knobs can be
positioned performance-wise. The paper's claims do not require it, but a reviewer will ask "how far
from optimal is static τ?"

### Interpretation — 4/5
Discussion separates regimes coherently and connects to entropy-collapse mechanisms. The
interpretability caveat (attention weights are functional, not explanatory) is well-placed. Minor
over-reach risk: "task-dependent reversal" is established at ONE LM setting; "reversal" language
should remain scoped to the tested regime (the paper mostly does this).

### Reference quality — 5/5
35 references, every one verified against Crossref/ACL Anthology/PMLR/arXiv/OpenAlex with a public
register; two commonly-miscited items corrected (GLUE venue; RoFormer DOI — though the latter is not
cited). No fabricated metadata. Two flagged partials (Cohen edition string; Picard excluded entirely)
are handled correctly.

### Scientific limitations disclosure — 4.5/5
Threats-to-validity section is unusually honest: budget/scheduler quirk, SST-2 dev-split artifact,
final-epoch evaluation, post-hoc AG News sweep, defective E1–E4 exclusions, stale extended grid.

### Overall publication readiness — 4/5 (accept after minor revisions)

---

## Weaknesses (honest list)

**W1 (moderate).** Single small architecture and short budgets. The reversal claim's scope is one
LM configuration; no evidence it persists at larger scale or longer training. The extended 25-epoch
grid (12/150) is excluded, so budget-dependence is untested.

**W2 (moderate).** No attention-entropy/logit-variance measurements during training, although the
mechanistic interpretation leans on them. The claim "LM prefers soft scaling because entropy is
preserved" is inferred from literature, not measured here.

**W3 (minor).** SST-2's test set is the GLUE dev split; the row is a dev-evaluation. Disclosed, but
a stronger design would evaluate on an external held-out set.

**W4 (minor).** SST-2/IMDB τ=0.5 mean effects are not Holm-significant. The paper handles this
correctly (directional evidence), but the abstract's "best arm" phrasing should not be read as
confirmed superiority; reviewers should note the abstract does say "not significant after Holm
correction" — acceptable.

**W5 (minor).** E1–E4 and the model-size sensitivity grids are absent (excluded due to defective
implementations). The paper loses robustness checks (model size) it could otherwise have had.

**W6 (minor).** One author; no external code review. The unit tests partially mitigate.

**W7 (nit).** The AG News τ=2 effect (p_raw 0.013, p_Holm 0.054 with 4-comparison family) is
threshold-sensitive; the paper should (and mostly does) present it as consistent-but-family-sensitive.

---

## Recommended experiments

### Mandatory before acceptance? — No. The paper as scoped is internally valid and complete.
The claims made are exactly those the design supports. However, the following would materially
strengthen a resubmission:

1. **(Strongly recommended, addresses W2)** Log per-head attention entropy and logit variance
   during training for a subset of runs (e.g., 5 seeds × 3 variants on WikiText-2 + AG News) and
   correlate the τ-response with measured entropy trajectories. Turns the mechanistic account from
   literature-inferred to measured. Cost: ~a few GPU-hours (probes already implemented in
   `src/attention/scaling.py`).
2. **(Recommended, addresses W1)** Complete the extended 25-epoch grid (150 runs ≈ 8–10 GPU-hours
   at observed speeds) with the corrected scheduler to test budget-dependence of the AG News effect
   and the LM reversal. If budget-dependent, the task-reversal narrative must be scoped further.
3. **(Optional, addresses W3)** Re-evaluate SST-2 on an external held-out set (e.g., the actual GLUE
   test predictions via the benchmark server, or a separate sentiment corpus like Yelp polarity) to
   escape the 872-example dev artifact.
4. **(Optional, addresses W5)** Re-implement E1–E4 correctly (multiscale branch activation; actual
   LR compensation √d_k ratio; correct vocabularies) and re-run (~70 runs, ~4 GPU-hours).

### Not recommended
Scaling to large models (out of scope for the claim; different study); adding more task types
(diminishing returns); abandoning the audit section (it is a feature, not a bug).

---

## Verdict

**Accept after minor revisions.** The measurement core is sound, unusually reproducible, and the
claims are calibrated to the evidence. The task-regime reversal and variance-stabilisation findings
are worth publishing. The revisions I would require before camera-ready are editorial: (a) scope the
"reversal" claim explicitly to the tested LM regime; (b) state the Holm family definition next to
every adjusted p-value; (c) add the two mandatory-disclosure items already in the supplement to the
main text's threats section. The recommended experiments above are enhancements, not blockers.

---

## Reviewer's note on process

The manuscript's own reproducibility audit (phantom τ-ablation claims with no run records; four
mutually inconsistent completion totals; placeholder figures in an earlier artifact package) is an
unusually candid contribution. I verified a sample of headline numbers against the per-seed records
and the corrected statistical pipeline: AG News (std 91.23±0.27; dk 90.92±0.30; dz 1.52; Holm 4.0e-7),
SST-2 (τ=0.5 81.08±0.55), IMDB (τ=0.5 82.29±0.21), WikiText-2 (dk 1.0811 vs std 1.0911, n=25 both)
— all reproduce exactly. The audit trail is genuine.
