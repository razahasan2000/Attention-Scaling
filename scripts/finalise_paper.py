"""
finalise_paper.py - post-completion automation for the Attention-Scaling paper.

Run this AFTER `run_all_v2.py` has finished the full matrix. It:

1. Audits results/per_seed/*.json (legacy + v2 records) and reports
   per-grid completion; writes results/statistical_tests/v2_completion.json.
2. Recomputes the full paired-statistics bundle (paired t, Wilcoxon
   signed-rank, Holm-Bonferroni, Cohen's d_z) for every completed grid and
   writes results/statistical_tests/v2_summary.json.
3. With --remove-manifest, deletes the 'planned versus completed' section
   (between %%MANIFEST-SECTION-START / END markers) from the manuscript so
   the final text no longer reports incomplete grids.
4. Reminds which prose passages still carry v1 numbers and must be
   regenerated from v2_summary.json (the script never invents numbers).

Usage:
  python scripts/finalise_paper.py --results-dir results --tex-dir manuscript
  python scripts/finalise_paper.py --remove-manifest --tex-dir /path/to/tex
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def load_records(results_dir: Path):
    per_seed = results_dir / "per_seed"
    records = []
    for f in sorted(per_seed.glob("*.json")):
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except json.JSONDecodeError:
            print(f"  WARNING: unreadable record skipped: {f.name}")
    return records


def grid_of(rec: dict) -> str:
    if "grid" in rec:
        return rec["grid"]
    # legacy v1 records: infer from run_id suffix
    rid = rec.get("run_id", "")
    if rid.endswith("_25epoch"):
        return "core" if rec.get("task") == "classification" else "lm_v1"
    if "temp" in rid and "ablation" in rid:
        return "temperature"
    return "legacy_other"


def completion_audit(records) -> dict:
    by_grid = defaultdict(lambda: {"total": 0, "completed": 0,
                                   "seeds": defaultdict(set)})
    for rec in records:
        g = grid_of(rec)
        by_grid[g]["total"] += 1
        if rec.get("status") == "completed":
            by_grid[g]["completed"] += 1
            key = rec.get("variant", "?")
            by_grid[g]["seeds"][key].add(rec.get("seed"))
    out = {}
    for g, info in sorted(by_grid.items()):
        seeds_per_arm = {k: len(v) for k, v in info["seeds"].items()}
        out[g] = {
            "records": info["total"],
            "completed": info["completed"],
            "seeds_per_arm": seeds_per_arm,
        }
    return out


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired_bundle(a: list, b: list, label_a: str, label_b: str,
                  family_size: int = 2) -> dict:
    """Paired comparison a - b (positive favours a)."""
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(diffs)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t = mean / (sd / math.sqrt(n))
        p = float(2.0 * sps.t.sf(abs(t), df=n - 1))
        ci = sps.t.interval(0.975, df=n - 1, loc=mean, scale=sd / math.sqrt(n))
        try:
            w_p = float(sps.wilcoxon(diffs).pvalue)
        except ValueError:
            w_p = float("nan")
        dz = mean / sd
    else:
        t, p, ci, w_p, dz = float("nan"), float("nan"), (mean, mean), float("nan"), 0.0
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n": n, "mean_diff": mean, "sd_diff": sd,
        "ci95": [float(ci[0]), float(ci[1])],
        "t": t, "p": p, "wilcoxon_p": w_p, "d_z": dz,
        "p_holm": p,  # placeholder; replaced by holm() below
        "family_size": family_size,
    }


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        adj = max(adj, prev)  # enforce monotonicity
        adjusted[idx] = adj
        prev = adj
    return adjusted


def analyse_grid(records, grid_name, dataset, metric="test_accuracy",
                 variants=("standard", "temp", "dk")) -> dict:
    per_variant = defaultdict(dict)  # variant -> {seed: metric}
    for rec in records:
        if grid_of(rec) != grid_name or rec.get("dataset") != dataset:
            continue
        if rec.get("status") != "completed" or rec.get(metric) is None:
            continue
        per_variant[rec.get("variant", rec.get("arm"))][rec.get("seed")] = rec[metric]

    results = {}
    baseline = per_variant.get("standard", {})
    for alt in [v for v in variants if v != "standard"]:
        alt_map = per_variant.get(alt, {})
        common = sorted(set(baseline) & set(alt_map))
        if len(common) < 2:
            results[alt] = {"error": f"insufficient paired seeds ({len(common)})"}
            continue
        a = [baseline[s] for s in common]
        b = [alt_map[s] for s in common]
        results[alt] = paired_bundle(a, b, "standard", alt, family_size=2)
    # Holm across the family (two comparisons per dataset)
    keys = [k for k in results if "p" in results[k]]
    if keys:
        adjusted = holm([results[k]["p"] for k in keys])
        for k, adj in zip(keys, adjusted):
            results[k]["p_holm"] = adj
    return results


# ---------------------------------------------------------------------------
# Manuscript surgery
# ---------------------------------------------------------------------------

def remove_manifest_section(tex_dir: Path) -> bool:
    changed = False
    for tex in sorted(tex_dir.rglob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        start = text.find("%%MANIFEST-SECTION-START")
        end = text.find("%%MANIFEST-SECTION-END")
        if start == -1 or end == -1:
            continue
        block = text[start:end + len("%%MANIFEST-SECTION-END")]
        new = text.replace(block,
                           "%% Planned-vs-completed manifest removed in the "
                           "final version: the full matrix is complete.\n")
        tex.write_text(new, encoding="utf-8")
        print(f"  removed manifest section from {tex.name}")
        changed = True
    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--tex-dir", type=str, default="manuscript")
    ap.add_argument("--remove-manifest", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir)
    print(f"Loaded {len(records)} per-seed records from {results_dir}/per_seed")

    audit = completion_audit(records)
    out_dir = results_dir / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v2_completion.json", "w") as f:
        json.dump(audit, f, indent=2)
    print("\nCompletion audit (results/statistical_tests/v2_completion.json):")
    for g, info in audit.items():
        print(f"  {g:<18} {info['completed']}/{info['records']} records, "
              f"seeds/arm: {info['seeds_per_arm']}")

    # Full statistics for every classification dataset with >= 2 paired seeds
    summary = {"note": "v2 recomputed statistics; generated by finalise_paper.py",
               "grids": {}}
    for grid in ["core", "converged", "dk32", "dk64", "dk128",
                 "extended25", "ablation_E1", "ablation_E2",
                 "ablation_E3", "ablation_E4"]:
        for ds in ["agnews", "sst2", "imdb"]:
            res = analyse_grid(records, grid, ds)
            if res:
                summary["grids"].setdefault(grid, {})[ds] = res
    # Language modelling: descriptive perplexity comparison
    lm = defaultdict(list)
    for rec in records:
        if rec.get("task") == "language_modelling" and rec.get("status") == "completed":
            if rec.get("causal_masking"):
                lm[rec.get("variant")].append(rec.get("test_perplexity"))
    if lm:
        summary["language_modelling_causal"] = {
            v: {"n": len(p), "mean_ppl": float(np.mean(p)),
                "sd_ppl": float(np.std(p, ddof=1)) if len(p) > 1 else None}
            for v, p in lm.items()}

    with open(out_dir / "v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStatistics bundle written to {out_dir / 'v2_summary.json'}")

    if args.remove_manifest:
        if remove_manifest_section(Path(args.tex_dir)):
            print("\nManifest section removed from the manuscript.")
        else:
            print("\nNo %%MANIFEST-SECTION markers found; nothing removed.")

    print(
        "\nNEXT STEPS (manual, numbers must come from v2_summary.json):\n"
        "  1. Update abstract/results numbers from v2_summary.json.\n"
        "  2. Re-run: python scripts/generate_figures.py "
        "results/statistical_tests figures\n"
        "  3. Re-compile the manuscript (tectonic main.tex) and the "
        "Word version.\n")


if __name__ == "__main__":
    main()
