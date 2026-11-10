"""
Seed-keyed statistical analysis for the completion batch.

Improvements over the original analyse_results.py (documented in the audit):
  - pairs runs strictly BY SEED (inner join), never by list position
  - groups the 'temp' variant by its tau value (record config / run-id) so the
    temperature ablation arm set is analysed as {tau0.5, tau1.0, tau1.5, tau2.0}
  - corrected Holm step-down with running-max monotonicity
  - adds Shapiro-Wilk on paired diffs, exact sign-flip permutation test (10k)
  - reports Wilcoxon, sign test, dz, CI as before
  - works on a mixed per_seed directory containing 2-epoch core records and
    new completion records (filtered by status == 'completed')

Outputs one JSON per dataset under --output-dir:
  {dataset}_completion_statistical_analysis.json
plus a combined summary_table_completion.json.
"""

import json
import os
import re
import math
import argparse
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def load_all_records(results_dir: str) -> list:
    records = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename)) as f:
                records.append(json.load(f))
    return records


def arm_key(record: dict) -> Optional[str]:
    """Arm name: variant, with temp split into temp_tau{X} by config tau."""
    variant = record.get('variant')
    if variant not in ('standard', 'temp', 'dk'):
        return None
    if variant == 'temp':
        tau = record.get('config', {}).get('tau', record.get('tau'))
        if tau is None:
            m = re.search(r'_tau([\d.]+)_', record.get('run_id', ''))
            tau = float(m.group(1)) if m else 2.0
        return f"temp_tau{tau:g}"
    return variant


def gather(records: list, dataset: str, kind: str) -> dict:
    """Return {arm: {seed: metric_value}} for a dataset."""
    out = defaultdict(dict)
    for r in records:
        if r.get('status') != 'completed' or r.get('dataset') != dataset:
            continue
        arm = arm_key(r)
        if arm is None:
            continue
        metric_key = 'test_accuracy' if kind == 'classification' else 'test_perplexity'
        if metric_key == 'test_accuracy':
            if metric_key not in r:
                continue
        elif metric_key not in r:
            continue
        else:
            pass
        out[arm][r['seed']] = r[metric_key]
    return out


def paired_test(base: dict, other: dict, better: str = 'higher') -> dict:
    """base/other: {seed: value}. better='higher' -> positive diff means other better."""
    seeds = sorted(set(base) & set(other))
    if not seeds:
        return {'n_pairs': 0}
    b = np.array([base[s] for s in seeds], float)
    o = np.array([other[s] for s in seeds], float)
    return _stats_from_diffs(b, o, seeds, better)


def _stats_from_diffs(b: np.ndarray, o: np.ndarray, seeds: list, better: str) -> dict:
    n = len(seeds)
    d = (o - b) if better == 'higher' else (b - o)
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    if se and se > 0:
        t = md / se
        p = float(2 * (1 - stats.t.cdf(abs(t), n - 1)))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ci = [md - tcrit * se, md + tcrit * se]
    else:
        t, p, ci = 0.0, 1.0, [md, md]
    dz = md / sd if sd and sd > 0 else 0.0
    # permutation: sign-flip test on mean diff (exact via normal approx 10k)
    rng = np.random.default_rng(42)
    perms = rng.choice([-1.0, 1.0], size=(10000, n))
    null = (perms * d).mean(axis=1)
    perm_p = float((np.abs(null) >= abs(md) - 1e-15).mean())
    # wilcoxon
    try:
        w, wp = stats.wilcoxon(d)
    except ValueError:
        w, wp = np.nan, np.nan
    # shapiro on diffs
    try:
        sh, shp = stats.shapiro(d)
    except Exception:
        sh, shp = np.nan, np.nan
    # sign test
    npos = int((d > 0).sum()); nneg = int((d < 0).sum()); nz = int((d == 0).sum())
    nh = npos + nneg
    sign_p = float(2 * min(stats.binom.cdf(npos, nh, .5),
                           stats.binom.cdf(nh - npos, nh, .5))) if nh else 1.0
    return {
        'n_pairs': n, 'seeds': seeds,
        'mean_diff': md, 'sd_diff': sd, 'se_diff': se,
        'ci95': ci, 'cohen_dz': float(dz), 't': float(t), 'p_value': p,
        'perm_p_10k': perm_p, 'wilcoxon_p': (float(wp) if wp == wp else None),
        'shapiro_W': (float(sh) if sh == sh else None),
        'shapiro_p': (float(shp) if shp == shp else None),
        'sign_test_p': sign_p,
        'n_pos': npos, 'n_neg': nneg, 'n_zero': nz,
    }


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(p * (m - i), 1.0))
        out[k] = run
    return out


def analyse_dataset(records: list, dataset: str, kind: str, baseline: str = 'standard') -> dict:
    arms = gather(records, dataset, kind)
    if not arms:
        return {}
    # descriptives
    desc = {}
    for arm, sv in sorted(arms.items()):
        v = np.array(list(sv.values()), float)
        n = len(v)
        if n < 2:
            desc[arm] = {'n': n, 'mean': float(v.mean()) if n else None}
            continue
        md_, sd_ = float(v.mean()), float(v.std(ddof=1))
        se_ = sd_ / math.sqrt(n); tcrit = float(stats.t.ppf(0.975, n - 1))
        desc[arm] = {'n': n, 'mean': md_, 'sd': sd_,
                     'ci95': [md_ - tcrit * se_, md_ + tcrit * se_]}
    # comparisons vs baseline, seed-keyed
    better = 'lower' if kind == 'lm' else 'higher'
    comps = {}
    for arm in sorted(arms):
        if arm == baseline:
            continue
        common = sorted(set(arms[baseline]) & set(arms[arm]))
        comps[f"{arm}_vs_{baseline}"] = _stats_from_diffs(
            np.array([arms[baseline][s] for s in common], float),
            np.array([arms[arm][s] for s in common], float),
            common, better)
    adj = holm({k: v['p_value'] for k, v in comps.items()})
    for k in comps:
        comps[k]['p_value_holm'] = adj.get(k)
    return {'metric': 'perplexity' if kind == 'lm' else 'accuracy',
            'descriptives': desc, 'paired_comparisons': comps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/results/per_seed')
    ap.add_argument('--output-dir', default='results/results/completion_statistical_tests')
    ap.add_argument('--datasets', default='agnews,sst2,imdb,wikitext2')
    args = ap.parse_args()

    records = load_all_records(args.results_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        kind = 'lm' if ds == 'wikitext2' else 'classification'
        res = analyse_dataset(records, ds, kind)
        if res:
            summary[ds] = res
            with open(os.path.join(args.output_dir, f'{ds}_completion_statistical_analysis.json'), 'w') as f:
                json.dump(res, f, indent=2)
            print(f"=== {ds} ({res['metric']}) ===")
            for arm, d in res['descriptives'].items():
                print(f"  {arm:12} n={d['n']:2d} mean={d['mean']:.4f} sd={d.get('sd', float("nan")):.4f}")
            for comp, d in res['paired_comparisons'].items():
                print(f"  {comp:26} diff={d['mean_diff']:+.5f} dz={d['cohen_dz']:+.3f} "
                      f"t={d['t']:+.2f} p={d['p_value']:.4g} holm={d['p_value_holm']:.4g} "
                      f"perm={d['perm_p_10k']:.4f} n={d['n_pairs']}")
    with open(os.path.join(args.output_dir, 'summary_table_completion.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {args.output_dir}")


if __name__ == '__main__':
    main()
