"""Aggregate Stage-2 replica results: paired statistics per phase.

Outputs data/results/aggregate.json with:
  - per-variant mean/sd of test accuracy (classification) or val perplexity (LM)
  - paired differences vs standard (variant - standard), paired t-test,
    95% t-CI, Cohen's dz, Holm-Bonferroni correction within each family
  - d_k grid: gap per d_k with CI, plus linear trend of the gap vs d_k
  - convergence: epochs-to-stop distribution, best-epoch accuracies, curves
  - ablations: 4 arms x 2 datasets
  - per-seed tables for the supplementary
"""

import json
import math

import numpy as np
from scipy import stats

import os
RES = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results")) + "/results.jsonl"
OUT = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results")) + "/aggregate.json"


def load():
    recs = [json.loads(l) for l in open(RES)]
    return recs


def describe(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


def paired(variant_vals, standard_vals):
    """diff = variant - standard; negative => standard better."""
    d = np.asarray(variant_vals, dtype=float) - np.asarray(standard_vals, dtype=float)
    n = len(d)
    t, p = stats.ttest_1samp(d, 0.0)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    tcrit = stats.t.ppf(0.975, n - 1)
    half = tcrit * d.std(ddof=1) / math.sqrt(n)
    return {"diff": float(d.mean()), "ci": [float(d.mean() - half), float(d.mean() + half)],
            "t": float(t), "p": float(p), "dz": float(dz), "n": n,
            "sd_diff": float(d.std(ddof=1))}


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in original order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    recs = load()
    agg = {}

    # ---------------- LM ----------------
    lm = [r for r in recs if r["phase"] == "lm"]
    seeds = sorted({r["seed"] for r in lm})
    lm_tab = {}
    for v in ("standard", "dk", "temp"):
        vals = [next(r["ppl"] for r in lm if r["seed"] == s and r["variant"] == v)
                for s in seeds]
        lm_tab[v] = describe(vals)
    lm_pairs = {}
    pvals = []
    comps = []
    for v in ("dk", "temp"):
        pr = paired([next(r["ppl"] for r in lm if r["seed"] == s and r["variant"] == v)
                     for s in seeds],
                    [next(r["ppl"] for r in lm if r["seed"] == s and r["variant"] == "standard")
                     for s in seeds])
        lm_pairs[v] = pr
        comps.append(pr["p"])
    adj = holm(comps)
    for v, a in zip(("dk", "temp"), adj):
        lm_pairs[v]["p_holm"] = float(a)
    agg["lm"] = {"per_variant": lm_tab, "paired": lm_pairs, "seeds": seeds,
                 "protocol": {"tokens_train": 96000, "tokens_val": 40000,
                              "epochs": 3, "seq": 64, "batch": 32,
                              "d_model": 96, "heads": 3, "d_k": 32, "vocab": 5655}}

    # ---------------- Core replica (AG News) ----------------
    def cls_block(phase, dataset=None):
        rs = [r for r in recs if r["phase"] == phase and
              (dataset is None or r["dataset"] == dataset)]
        seeds_ = sorted({r["seed"] for r in rs})
        out = {"per_variant": {}, "paired": {}, "seeds": seeds_}
        for v in ("standard", "dk", "temp"):
            vals = [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == v)
                    for s in seeds_]
            out["per_variant"][v] = describe(vals)
        pv = []
        comps = []
        for v in ("dk", "temp"):
            pr = paired([next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == v)
                         for s in seeds_],
                        [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "standard")
                         for s in seeds_])
            out["paired"][v] = pr
            comps.append(pr["p"])
        adj = holm(comps)
        for v, a in zip(("dk", "temp"), adj):
            out["paired"][v]["p_holm"] = float(a)
        return out

    agg["core"] = cls_block("core", "agnews")
    agg["core"]["protocol"] = {"train": 2000, "val": 500, "test": 4000,
                               "epochs": 2, "seq": 64, "d_model": 128,
                               "heads": 4, "d_k": 32}

    # ---------------- d_k grid ----------------
    grid = [r for r in recs if r["phase"] == "grid"]
    gtab = {}
    for dk in (32, 64, 128):
        rs = [r for r in grid if r["dk"] == dk]
        seeds_ = sorted({r["seed"] for r in rs})
        blk = {"seeds": seeds_, "per_variant": {}, "paired": {}}
        for v in ("standard", "dk", "temp"):
            vals = [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == v)
                    for s in seeds_]
            blk["per_variant"][v] = describe(vals)
        comps = []
        for v in ("dk", "temp"):
            pr = paired([next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == v)
                         for s in seeds_],
                        [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "standard")
                         for s in seeds_])
            blk["paired"][v] = pr
            comps.append(pr["p"])
        adj = holm(comps)
        for v, a in zip(("dk", "temp"), adj):
            blk["paired"][v]["p_holm"] = float(a)
        gtab[dk] = blk
    # trend of (standard - dk) gap with d_k: paired across the 12 seeds
    gaps = []
    for dk in (32, 64, 128):
        rs = [r for r in grid if r["dk"] == dk]
        seeds_ = sorted({r["seed"] for r in rs})
        g = [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "standard") -
             next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "dk")
             for s in seeds_]
        gaps.append(g)
    gmat = np.array(gaps)                      # 3 x 12
    slope = paired(gmat[2], gmat[0])           # dk=128 minus dk=32
    agg["grid"] = {"by_dk": gtab, "gap_trend_128_vs_32": slope,
                   "gap_means": [float(np.mean(g)) for g in gaps]}

    # ---------------- Convergence ----------------
    conv = [r for r in recs if r["phase"] == "conv"]
    ctab = {}
    for ds in ("sst2", "imdb"):
        rs = [r for r in conv if r["dataset"] == ds]
        seeds_ = sorted({r["seed"] for r in rs})
        blk = {"seeds": seeds_, "per_variant": {}, "paired": {}, "epochs_run": {}}
        for v in ("standard", "dk", "temp"):
            rows = [next(r for r in rs if r["seed"] == s and r["variant"] == v)
                    for s in seeds_]
            blk["per_variant"][v] = describe([r["test_acc"] for r in rows])
            blk["epochs_run"][v] = describe([r["epochs_run"] for r in rows])
        comps = []
        for v in ("dk", "temp"):
            pr = paired([next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == v)
                         for s in seeds_],
                        [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "standard")
                         for s in seeds_])
            blk["paired"][v] = pr
            comps.append(pr["p"])
        adj = holm(comps)
        for v, a in zip(("dk", "temp"), adj):
            blk["paired"][v]["p_holm"] = float(a)
        # mean curve per variant (align by epoch, NaN-padded)
        maxep = max(max(len(r["curve"]) for r in rs if r["variant"] == v)
                    for v in ("standard", "dk", "temp"))
        curves = {}
        for v in ("standard", "dk", "temp"):
            rows = [next(r for r in rs if r["seed"] == s and r["variant"] == v)
                    for s in seeds_]
            c = np.full((len(rows), maxep), np.nan)
            for i, r in enumerate(rows):
                for j, point in enumerate(r["curve"]):
                    c[i, j] = point["test_acc"]
            curves[v] = {"mean": [None if np.isnan(x) else float(x) for x in np.nanmean(c, 0)],
                         "epochs": list(range(1, maxep + 1))}
        blk["curves"] = curves
        ctab[ds] = blk
    agg["conv"] = ctab

    # ---------------- Ablations E1-E4 ----------------
    abl = [r for r in recs if r["phase"] == "abl"]
    atab = {}
    for ds in ("sst2", "imdb"):
        rs = [r for r in abl if r["dataset"] == ds]
        seeds_ = sorted({r["seed"] for r in rs})
        blk = {"seeds": seeds_, "per_arm": {}, "paired_vs_standard": {}}
        for arm in ("standard", "dk", "dk_lrcomp", "multiscale_v1"):
            vals = [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == arm)
                    for s in seeds_]
            blk["per_arm"][arm] = describe(vals)
        for arm in ("dk", "dk_lrcomp", "multiscale_v1"):
            blk["paired_vs_standard"][arm] = paired(
                [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == arm)
                 for s in seeds_],
                [next(r["test_acc"] for r in rs if r["seed"] == s and r["variant"] == "standard")
                 for s in seeds_])
        atab[ds] = blk
    agg["abl"] = atab

    # ---------------- Per-seed tables for supplementary ----------------
    per_seed = {"core_agnews": {}, "lm": {}}
    rs = [r for r in recs if r["phase"] == "core"]
    for r in sorted(rs, key=lambda x: x["seed"]):
        per_seed["core_agnews"].setdefault(r["seed"], {})[r["variant"]] = r["test_acc"]
    for r in sorted(lm, key=lambda x: x["seed"]):
        per_seed["lm"].setdefault(r["seed"], {})[r["variant"]] = r["ppl"]
    agg["per_seed"] = per_seed

    with open(OUT, "w") as f:
        json.dump(agg, f, indent=1)

    # ---------------- Console summary ----------------
    print("=== LM (WikiText-2, corrected causal mask), val ppl ===")
    for v, d in agg["lm"]["per_variant"].items():
        print(f"  {v:>9s}: {d['mean']:7.1f} +- {d['sd']:5.1f} [{d['min']:.0f},{d['max']:.0f}] n={d['n']}")
    for v, pr in agg["lm"]["paired"].items():
        print(f"  {v} vs std: diff {pr['diff']:+.2f} ppl, CI [{pr['ci'][0]:+.2f},{pr['ci'][1]:+.2f}], "
              f"dz {pr['dz']:+.2f}, p={fmt_p(pr['p'])}, p_holm={fmt_p(pr['p_holm'])}")

    print("=== Core replica (AG News, 2 epochs), test acc ===")
    for v, d in agg["core"]["per_variant"].items():
        print(f"  {v:>9s}: {d['mean']*100:.2f}% +- {d['sd']*100:.2f} n={d['n']}")
    for v, pr in agg["core"]["paired"].items():
        print(f"  {v} vs std: diff {pr['diff']*100:+.2f} pp, CI [{pr['ci'][0]*100:+.2f},{pr['ci'][1]*100:+.2f}] pp, "
              f"dz {pr['dz']:+.2f}, p={fmt_p(pr['p'])}, p_holm={fmt_p(pr['p_holm'])}")

    print("=== d_k grid (AG News) ===")
    for dk, blk in agg["grid"]["by_dk"].items():
        pr = blk["paired"]["dk"]
        print(f"  d_k={dk:3d}: std {blk['per_variant']['standard']['mean']*100:.2f}%, "
              f"dk {blk['per_variant']['dk']['mean']*100:.2f}%, "
              f"temp {blk['per_variant']['temp']['mean']*100:.2f}%; "
              f"dk-std diff {pr['diff']*100:+.2f} pp (p={fmt_p(pr['p'])})")
    tr = agg["grid"]["gap_trend_128_vs_32"]
    print(f"  gap trend (std-dk) d_k=128 vs 32: {tr['diff']*100:+.2f} pp, "
          f"CI [{tr['ci'][0]*100:+.2f},{tr['ci'][1]*100:+.2f}], p={fmt_p(tr['p'])}")

    print("=== Convergence (early stopping) ===")
    for ds, blk in agg["conv"].items():
        row = " | ".join(f"{v}: {blk['per_variant'][v]['mean']*100:.2f}% "
                         f"({blk['epochs_run'][v]['mean']:.1f} ep)" for v in ("standard","dk","temp"))
        print(f"  {ds}: {row}")
        for v, pr in blk["paired"].items():
            print(f"     {v} vs std: {pr['diff']*100:+.2f} pp, p={fmt_p(pr['p'])}")

    print("=== Ablations (5 epochs) ===")
    for ds, blk in agg["abl"].items():
        row = " | ".join(f"{a}: {blk['per_arm'][a]['mean']*100:.2f}%" for a in blk["per_arm"])
        print(f"  {ds}: {row}")


if __name__ == "__main__":
    main()
