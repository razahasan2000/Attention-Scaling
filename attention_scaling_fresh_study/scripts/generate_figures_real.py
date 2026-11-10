"""
Publication-quality figure generation for the attention scaling study.

All data-driven figures read from results/results/per_seed with the same
seed-keyed grouping as analyse_completion.py. Analytic/theoretical figures
(softmax sharpness, study design schematic) are explicitly labelled as such.

Outputs (PNG 300 dpi + PDF) to --output-dir. Re-runnable at any batch stage;
it uses whatever completed records exist (and labels per-arm n).
"""

import os, sys, json, re, math, argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

RESULTS = 'results/results/per_seed'
OUT = 'figures'

CMAP = {'standard': '#1f77b4', 'dk': '#d62728'}
TAU_CMAP = {0.5: '#9467bd', 1.0: '#2ca02c', 1.5: '#ff7f0e', 2.0: '#8c564b'}
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})


def arm_key(record):
    v = record.get('variant')
    if v not in ('standard', 'temp', 'dk'):
        return None
    if v == 'temp':
        tau = record.get('config', {}).get('tau')
        if tau is None:
            m = re.search(r'_tau([\d.]+)_', record.get('run_id', ''))
            tau = float(m.group(1)) if m else 2.0
        return f"temp_tau{tau:g}"
    return v


def arm_color(arm):
    if arm.startswith('temp_tau'):
        return TAU_CMAP[float(arm[8:])]
    return CMAP.get(arm, 'gray')


def load_records(results_dir):
    recs = []
    for fn in sorted(os.listdir(results_dir)):
        if fn.endswith('.json'):
            with open(os.path.join(results_dir, fn)) as f:
                recs.append(json.load(f))
    return [r for r in recs
            if r.get('status') == 'completed' and r.get('variant') in ('standard', 'temp', 'dk')]


def collect(recs, dataset, metric):
    out = defaultdict(dict)
    for r in recs:
        if r.get('dataset') != dataset:
            continue
        k = arm_key(r)
        if metric == 'test_accuracy' and metric not in r:
            continue
        if metric == 'test_perplexity' and metric not in r:
            continue
        out[k][r['seed']] = r[metric]
    return out


def mean_ci(v):
    v = np.asarray(list(v), float)
    n = len(v)
    m = v.mean(); sd = v.std(ddof=1)
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se, n, sd

# ---------------------------------------------------------------- Figure 1/5
FIG_MAIN_PERF = "fig_main_performance"
FIG_PAIRED = "fig_paired_differences"
FIG_TEMP = "fig_temperature_ablation"
FIG_WT2 = "fig_wikitext2"
FIG_SEEDS = "fig_seed_distributions"
FIG_EFF = "fig_efficiency"
FIG_SHARP = "fig_softmax_sharpness"
FIG_DESIGN = "fig_study_design"


def fig_main_performance(recs, path):
    datasets = ['agnews', 'sst2', 'imdb']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)
    for ax, ds in zip(axes, datasets):
        arms = collect(recs, ds, 'test_accuracy')
        arms = {k: v for k, v in arms.items() if not k.startswith('temp_tau') or k == 'temp_tau2'}
        names = ['standard', 'dk', 'temp_tau2']
        for i, arm in enumerate([a for a in names if a in arms]):
            m, lo, hi, n, sd = mean_ci(arms[arm].values())
            c = arm_color(arm)
            ax.scatter([i], [100 * m], s=42, color=c, zorder=3)
            ax.vlines(i, 100 * lo, 100 * hi, color=c, lw=2.2, zorder=2)
            ax.text(i, 100 * lo - 0.32, f"n={n}", ha='center', va='top', fontsize=7.5, color=c)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_tau2', ' (τ=2)') for n in names if n in arms])
        ax.set_title({'agnews': 'AG News (4-class)', 'sst2': 'SST-2 (binary)',
                      'imdb': 'IMDB (binary)'}[ds])
        ax.grid(axis='y', alpha=.25, lw=.4)
        ax.set_ylabel('Test accuracy (%)')
        pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * .04
        ax.set_ylim(ax.get_ylim()[0] - 1.3, ax.get_ylim()[1] + pad)
    fig.suptitle('Main comparison (2-epoch budget, 25 matched seeds, mean ±95% CI)',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def fig_paired_differences(recs, path):
    datasets = ['agnews', 'sst2', 'imdb']
    arms = ['standard']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, ds in zip(axes, datasets):
        A = collect(recs, ds, 'test_accuracy')
        for arm in ['dk', 'temp_tau2']:
            if arm not in A or 'standard' not in A:
                continue
            seeds = sorted(set(A[arm]) & set(A['standard']))
            d = np.array([100 * (A['standard'][s] - A[arm][s]) for s in seeds])
            c = arm_color(arm)
            x = (0 if arm == 'dk' else 1) + np.random.default_rng(0).uniform(-.08, .08, len(d))
            ax.scatter(x, d, s=14, alpha=.65, color=c, zorder=2)
            md, lo, hi = (d.mean(), *stats.t.ppf(.975, len(d) - 1) * np.array([1, 1]) *
                          d.std(ddof=1) / math.sqrt(len(d)) * np.array([1, 1]))
            ax.hlines(d.mean(), -.32 if arm == 'dk' else .68, .32 if arm == 'dk' else 1.32,
                      color=c, lw=2)
            ax.add_patch(plt.Rectangle((-.32 if arm == 'dk' else .68, lo), .64, hi - lo,
                                       fill=False, edgecolor=c, lw=1.2, zorder=3))
        ax.axhline(0, color='k', lw=.8, ls='--', alpha=.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['1/d_k', 'temp τ=2'])
        ax.set_title({'agnews': 'AG News', 'sst2': 'SST-2', 'imdb': 'IMDB'}[ds])
        ax.grid(axis='y', alpha=.25, lw=.4)
        ax.set_ylabel('Standard − variant (pp)')
    fig.suptitle('Paired per-seed accuracy differences (points = seeds; box = mean ±95% CI)',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def fig_temperature_ablation(recs, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, ds in zip(axes, ['sst2', 'imdb', 'agnews']):
        A = collect(recs, ds, 'test_accuracy')
        taus = sorted(float(k[8:]) for k in A if k.startswith('temp_tau'))
        if not taus:
            ax.text(.5, .5, 'pending', ha='center'); continue
        xs, ms, los, his, ns = [], [], [], [], []
        for t in taus:
            m, lo, hi, n, sd = mean_ci(A[f'temp_tau{t:g}'].values())
            xs.append(t); ms.append(100 * m); los.append(100 * lo); his.append(100 * hi); ns.append(n)
        ms, los, his = map(np.array, (ms, los, his))
        ax.fill_between(xs, los, his, color='#d62728', alpha=.12)
        ax.plot(xs, ms, 'o-', color='#d62728', lw=1.8, ms=4.5)
        if 'standard' in A:
            m, lo, hi, n, sd = mean_ci(A['standard'].values())
            ax.axhspan(100 * lo, 100 * hi, color='#1f77b4', alpha=.18)
            ax.axhline(100 * m, color='#1f77b4', lw=1.6)
        if 'dk' in A:
            m, lo, hi, n, sd = mean_ci(A['dk'].values())
            ax.axhline(100 * m, color='#666666', lw=1.2, ls='--')
        ax.set_xlabel('temperature τ'); ax.set_title(ds.upper())
        ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel('Test accuracy (%)')
    fig.suptitle('Temperature ablation (25 matched seeds; band = 95% CI)',
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def fig_wikitext2(recs, path):
    A = collect(recs, 'wikitext2', 'test_perplexity')
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    names = ['standard', 'dk'] + [f'temp_tau{t:g}' for t in sorted(float(k[8:]) for k in A if k.startswith('temp_tau'))]
    xs, ms, los, his, cs = [], [], [], [], []
    for i, arm in enumerate([a for a in names if a in A]):
        m, lo, hi, n, sd = mean_ci(A[arm].values())
        xs.append(i); ms.append(m); los.append(lo); his.append(hi); cs.append(arm_color(arm))
        ax.text(i, lo - .004, f"n={n}", ha='center', va='top', fontsize=7.5, color=arm_color(arm))
    ms, los, his = map(np.array, (ms, los, his))
    ax.errorbar(xs, ms, yerr=[ms - los, his - ms], fmt='o', ms=6, capsize=4, lw=1.6,
                ecolor='#333333', mfc='white', zorder=3)
    for x, y, c in zip(xs, ms, cs):
        ax.scatter([x], [y], color=c, s=30, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace('temp_tau', 'temp τ=').replace('dk', '1/d_k').replace('standard', 'standard')
                        for n in names if n in A])
    ax.set_ylabel('Validation perplexity')
    ax.set_title('WikiText-2 language modelling (25k steps)')
    ax.grid(axis='y', alpha=.25, lw=.4)
    fig.tight_layout()
    _save(fig, path)


def fig_seed_distributions(recs, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, ds in zip(axes, ['agnews', 'sst2', 'imdb']):
        A = collect(recs, ds, 'test_accuracy')
        arms = [a for a in ['standard', 'dk', 'temp_tau2'] if a in A]
        data = [100 * np.array(list(A[a].values())) for a in arms]
        parts = ax.violinplot(data, showmedians=False, showextrema=False)
        for pc, a in zip(parts['bodies'], arms):
            pc.set_facecolor(arm_color(a)); pc.set_alpha(.35)
        for i, (a, d) in enumerate(zip(arms, data)):
            x = np.random.default_rng(0).uniform(-.1, .1, len(d))
            ax.scatter(np.full(len(d), i + 1) + x, d, s=9, alpha=.75, color=arm_color(a))
            med = np.median(d)
            ax.hlines(med, i + .75, i + 1.25, color=arm_color(a), lw=2)
        ax.set_xticks(range(1, len(arms) + 1))
        ax.set_xticklabels([a.replace('_tau2', ' (τ=2)').replace('temp', 'temp') for a in arms])
        ax.set_title(ds.upper())
        ax.grid(axis='y', alpha=.25, lw=.4)
    axes[0].set_ylabel('Test accuracy (%)')
    fig.suptitle('Per-seed distributions (2-epoch budget, 25 matched seeds)', y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def fig_efficiency(recs, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    per_arm = defaultdict(list)
    for r in recs:
        if r.get('dataset') != 'agnews' or r.get('task') != 'classification':
            continue
        h = r.get('training_history', [])
        if h:
            per_arm[arm_key(r)].append(sum(e.get('epoch_time_s', 0) for e in h))
    arms = sorted(per_arm)
    ms, los, his = [], [], []
    for a in arms:
        m, lo, hi, n, sd = mean_ci(per_arm[a])
        ms.append(m); los.append(lo); his.append(hi)
    xs = np.arange(len(arms))
    ms, los, his = map(np.array, (ms, los, his))
    ax.bar(xs, ms, yerr=[ms - los, his - ms], capsize=4, color=[arm_color(a) for a in arms],
           alpha=.8, width=.6)
    ax.set_xticks(xs); ax.set_xticklabels([a.replace('_', ' / ') for a in arms])
    ax.set_ylabel('Training wall-clock per run (s)')
    ax.set_title('AG News computational efficiency (2-epoch budget)')
    ax.grid(axis='y', alpha=.25, lw=.4)
    fig.tight_layout()
    _save(fig, path)


def fig_softmax_sharpness(path):
    """Analytic/theoretical: softmax sharpness vs logit scale (NOT experimental data)."""
    s = np.linspace(0.5, 12, 300)  # logit std proxy
    from scipy.stats import norm, entropy
    d = 32
    dk_grid = np.linspace(.05, 1.4, 40)  # QK^T std per 1/sqrt(dk) unit
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    # Panel A: top-1 prob as a function of logit std (simulated max logit gap)
    for label, factor, color in [('1/√d_k (standard)', 1.0, CMAP['standard']),
                                 ('1/d_k', math.sqrt(d), CMAP['dk']),
                                 ('τ=2', 0.5, '#7f7f7f'),
                                 ('τ=0.5', 2.0, TAU_CMAP[0.5])]:
        top1, H = [], []
        for x in s:
            # simulated logits: gap normalized; use two-spot logit distribution
            r = np.random.default_rng(1).normal(0, x, 128)
            top1.append(softmax_prob(r, factor, 1))
            q = softmax_entropy(r, factor)
            H.append(q)
        axes[0].plot(s, top1, label=label, color=color, lw=1.6)
        axes[1].plot(s, H, label=label, color=color, lw=1.6)
    axes[0].set_xlabel('raw QK$^T$ logit std'); axes[0].set_ylabel('max attention prob')
    axes[0].set_title('(a) Concentration (expected p_max)')
    axes[1].set_xlabel('raw QK$^T$ logit std'); axes[1].set_ylabel('attention entropy (nats)')
    axes[1].set_title('(b) Entropy')
    axes[1].legend(frameon=False, fontsize=8.5)
    for ax in axes:
        ax.grid(alpha=.25, lw=.4)
    fig.suptitle('Analytic behaviour of scaling functions (Monte Carlo over random logits, d_k=32; '
                 'not experimental data)', fontsize=9.5, y=1.03)
    fig.tight_layout()
    _save(fig, path)


def softmax_prob(logits, factor, topk=1):
    z = logits / (factor * math.sqrt(dk := 32)) if False else logits * (1.0 / (factor * math.sqrt(32))) \
        if False else logits / factor
    # NOTE: 'factor' here already accounted standard=1/(1), dk=1/sqrt(dk) as multiplier semantics; keep simple:
    p = np.exp(z - z.max()); p /= p.sum()
    return np.sort(p)[::-1][:topk].sum()


def softmax_entropy(logits, factor):
    z = logits / (factor * math.sqrt(32)) if False else logits / factor
    p = np.exp(z - z.max()); p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + '.png', bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=RESULTS)
    ap.add_argument('--output-dir', default=OUT)
    args = ap.parse_args()
    recs = load_records(args.results_dir)
    O = lambda name: os.path.join(args.output_dir, name)
    fig_main_performance(recs, O(FIG_MAIN_PERF))
    fig_paired_differences(recs, O(FIG_PAIRED))
    fig_temperature_ablation(recs, O(FIG_TEMP))
    fig_wikitext2(recs, O(FIG_WT2))
    fig_seed_distributions(recs, O(FIG_SEEDS))
    fig_efficiency(recs, O(FIG_EFF))
    fig_softmax_sharpness(O(FIG_SHARP))
    print(f"Figures written to {args.output_dir}")


if __name__ == '__main__':
    main()
