"""
Publication-quality figures for the Attention-Scaling manuscript.
Style: charts-skill rules (Paul Tol colorblind-safe palette, no top/right
spines, faint y-grid, constrained_layout, legend outside or non-overlapping,
300 DPI PNG, white background).

All numbers come from study_data.py (canonical source of truth).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_data import (CORE, CORE_N, PAIRED, LM, TEMP_ABL, TEMP_ABL_N,
                        EXTENDED, DIAGNOSTICS, EFFICIENCY, SCALING,
                        STAGE2, MANIFEST2)

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures"))
os.makedirs(OUT, exist_ok=True)

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#E5E7EB",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
})

# Variant colours (Paul Tol colorblind-safe; consistent across ALL figures)
C_STD, C_TMP, C_DK = "#0077BB", "#009988", "#EE7733"
VARIANT_COLORS = {"standard": C_STD, "temp": C_TMP, "temp2": C_TMP, "dk": C_DK}
VARIANT_LABELS = {"standard": "Standard (1/√d$_k$)",
                  "temp": "Temp (τ = 2)",
                  "temp2": "Temp (τ = 2)",
                  "dk": "1/d$_k$"}
G900, G700, G500, G400, G300, G200 = "#111827", "#374151", "#6B7280", "#9CA3AF", "#D1D5DB", "#E5E7EB"
SIG = "#CC3311"   # significant accent
NS = G400         # non-significant

DATASET_NAMES = {"agnews": "AG News (4-class)", "sst2": "SST-2 (binary)",
                 "imdb": "IMDB (binary)"}
VARIANT_ORDER = ["standard", "temp", "dk"]


def clean_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, alpha=0.12, color=G300, linewidth=0.7)
        ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Main performance (mean ± SD across 25 seeds), 3 panels
# ════════════════════════════════════════════════════════════════════════════
def fig_main_performance():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), constrained_layout=True)
    ylims = {"agnews": (90.0, 92.0), "sst2": (79.0, 83.5), "imdb": (80.5, 84.0)}
    for ax, ds in zip(axes, ["agnews", "sst2", "imdb"]):
        lo, hi = ylims[ds]
        for i, v in enumerate(VARIANT_ORDER):
            m = CORE[ds][v]["mean"] * 100
            sd = CORE[ds][v]["sd"] * 100
            ax.errorbar(i, m, yerr=sd, fmt="o", markersize=6.5,
                        color=VARIANT_COLORS[v], ecolor=VARIANT_COLORS[v],
                        elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
            # label above the upper whisker (zero overlap with the cap)
            ax.annotate(f"{m:.2f}", (i, m + sd), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom",
                        fontsize=8, color=G700)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Std", "τ = 2", "1/d$_k$"])
        ax.set_xlim(-0.55, 2.55)
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad * 0.35, hi + pad)
        ax.set_title(DATASET_NAMES[ds])
        clean_axis(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    save(fig, "fig_main_performance.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: Paired differences with 95% CIs (forest plot)
# ════════════════════════════════════════════════════════════════════════════
def fig_paired_differences():
    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    yticks, ylabels, bold_flags = [], [], []
    y = 0.0
    for ds in ["agnews", "sst2", "imdb"]:
        yticks.append(y)
        ylabels.append(DATASET_NAMES[ds])
        bold_flags.append(True)
        y -= 0.85
        for r in PAIRED[ds]:
            diff = r["diff"] * 100
            lo, hi = r["ci"][0] * 100, r["ci"][1] * 100
            color = SIG if r["sig"] else NS
            ls = "-" if r["sig"] else "--"
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.8, linestyle=ls,
                    solid_capstyle="round", zorder=3)
            ax.plot([lo, lo], [y - 0.09, y + 0.09], color=color, linewidth=1.4, zorder=3)
            ax.plot([hi, hi], [y - 0.09, y + 0.09], color=color, linewidth=1.4, zorder=3)
            ax.plot(diff, y, "o", color=color, markersize=5.5, zorder=4)
            star = r["sig"].replace("*", "\u2605")
            tag = f"\u0394 = {diff:+.2f} pp" + (f"  {star}" if star else "")
            # stats column at fixed x, right of the widest whisker (0.39)
            ax.text(0.50, y, tag, fontsize=8.2,
                    color=color if r["sig"] else G500,
                    va="center", ha="left")
            yticks.append(y)
            ylabels.append("    " + (VARIANT_LABELS["dk"] if "dk" in r["comparison"]
                                     else "Temp (\u03c4 = 2)"))
            bold_flags.append(False)
            y -= 1.0
        y -= 0.55
    ax.axvline(0, color=G400, linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(0.465, color=G200, linewidth=1.0, linestyle="-", zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for lab, bold in zip(ax.get_yticklabels(), bold_flags):
        if bold:
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_ylim(y + 0.6, 0.72)
    ax.set_xlabel("Paired difference, baseline \u2212 variant (percentage points)\n"
                  "95% CI; positive = standard scaling better", fontsize=8.5)
    ax.set_xlim(-0.45, 0.95)
    clean_axis(ax, grid=False)
    ax.xaxis.grid(True, alpha=0.12, color=G300, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_paired_differences.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5: Temperature ablation (τ ∈ {0.5, 1.0, 1.5, 2.0}, n = 3 seeds)
# ════════════════════════════════════════════════════════════════════════════
def fig_temperature_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    for ax, ds, yl in zip(axes, ["sst2", "imdb"], ylims := (None, None)):
        taus = [r["tau"] for r in TEMP_ABL[ds]]
        means = [r["mean"] * 100 for r in TEMP_ABL[ds]]
        sds = [r["sd"] * 100 for r in TEMP_ABL[ds]]
        ax.errorbar(taus, means, yerr=sds, fmt="o-", color=C_TMP, ecolor=C_TMP,
                    elinewidth=1.4, capsize=3.5, capthick=1.4, markersize=6,
                    markerfacecolor="white", markeredgewidth=1.8, linewidth=1.8,
                    zorder=3)
        # reference: core standard scaling (τ = 1, 25 seeds)
        ref = CORE[ds]["standard"]["mean"] * 100
        ax.axhline(ref, color=C_STD, linewidth=1.2, linestyle="--", alpha=0.85,
                   zorder=2)
        ax.annotate(f"standard, 25 seeds ({ref:.2f})",
                    (0.02, ref), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, -11),
                    fontsize=7.6, color=C_STD, ha="left")
        ax.set_xticks(taus)
        ax.set_xlabel("Softmax temperature τ")
        ax.set_title("SST-2" if ds == "sst2" else "IMDB")
        lo = min(means) - 0.55
        hi = max(means) + 0.55
        ax.set_ylim(lo, hi)
        clean_axis(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    save(fig, "fig_temperature_ablation.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6: Extended-budget convergence (5 seeds, checkpoints at 2/5/10/25)
# ════════════════════════════════════════════════════════════════════════════
def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    styles = {"standard": (C_STD, "o"), "temp": (C_TMP, "s"), "dk": (C_DK, "^")}
    for ax, ds in zip(axes, ["sst2", "imdb"]):
        ep = EXTENDED[ds]["epochs"]
        for v in ["standard", "temp", "dk"]:
            c, mk = styles[v]
            y = np.array(EXTENDED[ds][v]) * 100
            ax.plot(ep, y, color=c, marker=mk, markersize=5.5, linewidth=1.8,
                    markerfacecolor="white", markeredgewidth=1.5,
                    label=VARIANT_LABELS[v], zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(ep)
        ax.set_xticklabels([str(e) for e in ep])
        ax.minorticks_off()
        ax.set_xlabel("Training epochs (log scale)")
        ax.set_title("SST-2" if ds == "sst2" else "IMDB")
        lo = min(min(EXTENDED[ds][v]) for v in ["standard", "temp", "dk"]) * 100 - 0.5
        hi = max(max(EXTENDED[ds][v]) for v in ["standard", "temp", "dk"]) * 100 + 0.5
        ax.set_ylim(lo, hi)
        clean_axis(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8,
                   handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_training_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7: Gradient norms vs theory (epoch 1, batch 0)
# ════════════════════════════════════════════════════════════════════════════
def fig_gradient_ratios():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    gn = DIAGNOSTICS["grad_norms"]
    gn_key = {"standard": "standard", "temp": "temp2", "dk": "dk"}
    # (a) absolute norms
    ax = axes[0]
    x = np.arange(3)
    w = 0.26
    for i, v in enumerate(VARIANT_ORDER):
        vals = [gn[gn_key[v]][p] for p in ["Q", "K", "V"]]
        ax.bar(x + (i - 1) * w, vals, width=w * 0.92, color=VARIANT_COLORS[v],
               label=VARIANT_LABELS[v], zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["∂L/∂Q", "∂L/∂K", "∂L/∂V"])
    ax.set_ylabel("Gradient norm (epoch 1)")
    ax.set_title("(a) Absolute gradient norms")
    ax.set_ylim(0, 0.055)
    ax.legend(fontsize=7.2, frameon=False, loc="upper right", ncol=1,
              handlelength=1.4)
    clean_axis(ax)

    # (b) observed vs theoretical ratio
    ax = axes[1]
    x = np.arange(2)
    w = 0.3
    obs = {"temp2": [gn["temp2"]["Q"] / gn["standard"]["Q"],
                     gn["temp2"]["K"] / gn["standard"]["K"]],
           "dk": [gn["dk"]["Q"] / gn["standard"]["Q"],
                  gn["dk"]["K"] / gn["standard"]["K"]]}
    for i, v in enumerate(["temp2", "dk"]):
        ax.bar(x + (i - 0.5) * w, obs[v], width=w * 0.9, color=VARIANT_COLORS[v],
               zorder=3, edgecolor="white", linewidth=0.4,
               label=VARIANT_LABELS[v])
        for xi, val in zip(x + (i - 0.5) * w, obs[v]):
            ax.annotate(f"{val:.2f}", (xi, val), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.6, color=G700)
    # theoretical lines
    for v, yy in [("temp2", SCALING["standard"] / SCALING["temp2"]),
                  ("dk", SCALING["standard"] / SCALING["dk"])]:
        ax.axhline(yy, color=VARIANT_COLORS[v], linewidth=1.1, linestyle="--",
                   alpha=0.9, zorder=2)
        ax.annotate(f"theory 1/s = {yy:.2f}", (1.46, yy), fontsize=7.4,
                    color=VARIANT_COLORS[v], va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["Queries Q", "Keys K"])
    ax.set_ylabel("Ratio vs standard")
    ax.set_title("(b) Observed vs theoretical 1/s")
    ax.set_ylim(0, 0.66)
    clean_axis(ax)
    save(fig, "fig_gradient_ratios.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 8: Attention diagnostics (entropy, max weight, head cosine)
# ════════════════════════════════════════════════════════════════════════════
def fig_head_diversity():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), constrained_layout=True)
    panels = [
        ("entropy", "Mean attention entropy", "nats", DIAGNOSTICS["entropy"], (2.0, 2.62)),
        ("max_weight", "Max attention weight", "prob.", DIAGNOSTICS["max_weight"], (0.20, 0.355)),
        ("head_cosine", "Inter-head cosine sim.", "cos", DIAGNOSTICS["head_cosine"], (0.15, 0.48)),
    ]
    diag_key = {"standard": "standard", "temp": "temp2", "dk": "dk"}
    for ax, (key, title, ylab, data, ylim) in zip(axes, panels):
        vals = [data[diag_key[v]] for v in VARIANT_ORDER]
        bars = ax.bar(range(3), vals, width=0.62, zorder=3,
                      color=[VARIANT_COLORS[v] for v in VARIANT_ORDER],
                      edgecolor="white", linewidth=0.4)
        for b, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=G700)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Std", "τ = 2", "1/d$_k$"])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab, fontsize=8)
        clean_axis(ax)
    save(fig, "fig_head_diversity.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9: Efficiency (epoch time, inference latency)
# ════════════════════════════════════════════════════════════════════════════
def fig_efficiency():
    eff_key = {"standard": "standard", "temp": "temp2", "dk": "dk"}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), constrained_layout=True)
    # (a) epoch time
    ax = axes[0]
    x = np.arange(2)
    w = 0.26
    datasets = [("SST-2", EFFICIENCY["epoch_time_sst2"]),
                ("AG News", EFFICIENCY["epoch_time_agnews"])]
    for i, v in enumerate(VARIANT_ORDER):
        vals = [d[1][eff_key[v]] for d in datasets]
        bars = ax.bar(x + (i - 1) * w, vals, width=w * 0.9,
                      color=VARIANT_COLORS[v], zorder=3,
                      edgecolor="white", linewidth=0.4,
                      label=VARIANT_LABELS[v])
        for b, val in zip(bars, vals):
            ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 2.5),
                        ha="center", fontsize=7.4, color=G700)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in datasets])
    ax.set_ylabel("Training time per epoch (s)")
    ax.set_title("(a) Epoch wall-clock time")
    ax.set_ylim(0, 72)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left", handlelength=1.4)
    clean_axis(ax)

    # (b) inference latency
    ax = axes[1]
    vals = [EFFICIENCY["infer_latency_s_1k"][eff_key[v]] for v in VARIANT_ORDER]
    bars = ax.bar(range(3), vals, width=0.6, zorder=3,
                  color=[VARIANT_COLORS[v] for v in VARIANT_ORDER],
                  edgecolor="white", linewidth=0.4)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 2.5),
                    ha="center", fontsize=8, color=G700)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Std", "τ = 2", "1/d$_k$"])
    ax.set_ylabel("Inference latency, 1k samples (s)")
    ax.set_title("(b) Inference latency")
    ax.set_ylim(0, 1.05)
    clean_axis(ax)
    save(fig, "fig_efficiency.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure for LM perplexity (small forest, used in Results LM subsection)
# ════════════════════════════════════════════════════════════════════════════
def fig_lm_perplexity():
    fig, ax = plt.subplots(figsize=(4.6, 1.9), constrained_layout=True)
    items = [("standard", LM["standard"]), ("temp2", LM["temp2"]), ("dk", LM["dk"])]
    ys = [2, 1, 0]
    for y, (v, d) in zip(ys, items):
        half = 1.96 * d["sd"] / np.sqrt(d["n"])
        lo, hi = d["ppl"] - half, d["ppl"] + half
        ax.errorbar(d["ppl"], y, xerr=half, fmt="o", markersize=6.5,
                    color=VARIANT_COLORS[v], ecolor=VARIANT_COLORS[v],
                    elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
        n = d["n"]
        ax.annotate(f"{d['ppl']:.4f}  (n={n})", (d["ppl"], y),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=G700)
    ax.set_yticks(ys)
    ax.set_yticklabels([VARIANT_LABELS["standard"], VARIANT_LABELS["temp2"],
                        VARIANT_LABELS["dk"]])
    ax.set_xlabel("Validation perplexity (95% CI), 25,000 steps")
    ax.set_xlim(1.075, 1.101)
    ax.invert_yaxis()
    clean_axis(ax, grid=False)
    ax.xaxis.grid(True, alpha=0.12, color=G300, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_lm_perplexity.png")


# ════════════════════════════════════════════════════════════════════════════
# Stage-2 Figure 10: replica core (AG News) means and paired differences
# ════════════════════════════════════════════════════════════════════════════
def fig_stage2_core():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) means +- sd
    ax = axes[0]
    for i, v in enumerate(VARIANT_ORDER):
        d = STAGE2["core_agnews"][v]
        m, sd = d["mean"] * 100, d["sd"] * 100
        ax.errorbar(i, m, yerr=sd, fmt="o", markersize=6.5,
                    color=VARIANT_COLORS[v], ecolor=VARIANT_COLORS[v],
                    elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
        ax.annotate(f"{m:.2f}", (i, m + sd), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", fontsize=8, color=G700)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Std", "τ = 2", "1/d$_k$"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(74.5, 82.0)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("(a) Replica core, AG News (n = 25)")
    clean_axis(ax)
    # (b) paired diffs forest
    ax = axes[1]
    rows = STAGE2["core_agnews_paired"]
    ys = [1, 0]
    for y, r in zip(ys, rows):
        diff = r["diff"] * 100
        lo, hi = r["ci"][0] * 100, r["ci"][1] * 100
        color = NS
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.8, linestyle="--",
                solid_capstyle="round", zorder=3)
        ax.plot([lo, lo], [y - 0.07, y + 0.07], color=color, linewidth=1.4, zorder=3)
        ax.plot([hi, hi], [y - 0.07, y + 0.07], color=color, linewidth=1.4, zorder=3)
        ax.plot(diff, y, "o", color=color, markersize=5.5, zorder=4)
        tag = (f"$\\Delta$ = {diff:+.3f} pp,  $d_z$ = {r['dz']:+.2f},  "
               f"$p_{{Holm}}$ = {r['p_holm']:.2f}")
        ax.text(0.052, y, tag, fontsize=8, color=G500, va="center", ha="left")
    ax.axvline(0, color=G400, linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(0.048, color=G200, linewidth=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(["1/d$_k$ vs Std", "Temp (τ = 2) vs Std"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-0.055, 0.16)
    ax.set_xlabel("Paired difference, variant − standard (pp)\n"
                  "95% CI; Stage 1 full-scale effect was +0.31 pp for 1/d$_k$",
                  fontsize=8.5)
    ax.set_title("(b) Within-seed differences")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid=False)
    ax.xaxis.grid(True, alpha=0.12, color=G300, linewidth=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig_stage2_core.png")


# ════════════════════════════════════════════════════════════════════════════
# Stage-2 Figure 11: head-dimension grid (d_k in {32, 64, 128})
# ════════════════════════════════════════════════════════════════════════════
def fig_dk_grid():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    dks = [32, 64, 128]
    # (a) accuracy vs d_k
    ax = axes[0]
    for v in VARIANT_ORDER:
        m = [STAGE2["grid"][str(dk)][v][0] * 100 for dk in dks]
        sd = [STAGE2["grid"][str(dk)][v][1] * 100 for dk in dks]
        ax.errorbar(dks, m, yerr=sd, fmt="o-", color=VARIANT_COLORS[v],
                    ecolor=VARIANT_COLORS[v], elinewidth=1.2, capsize=3,
                    capthick=1.2, markersize=5.5, markerfacecolor="white",
                    markeredgewidth=1.5, linewidth=1.8,
                    label=VARIANT_LABELS[v], zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dks)
    ax.set_xticklabels([str(d) for d in dks])
    ax.minorticks_off()
    ax.set_xlabel("Head dimension $d_k$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(76.5, 81.5)
    ax.set_title("(a) Accuracy vs $d_k$ (AG News, n = 12)")
    ax.legend(fontsize=7.4, frameon=False, loc="lower left", handlelength=1.5)
    clean_axis(ax)
    # (b) 1/d_k - std gap with CI
    ax = axes[1]
    for i, (v, mk) in enumerate([("dk", "o"), ("temp", "s")]):
        diffs = [STAGE2["grid"][str(dk)][f"{v}_minus_std"]["diff"] * 100 for dk in dks]
        los = [STAGE2["grid"][str(dk)][f"{v}_minus_std"]["ci"][0] * 100 for dk in dks]
        his = [STAGE2["grid"][str(dk)][f"{v}_minus_std"]["ci"][1] * 100 for dk in dks]
        off = (i - 0.5) * 3.5
        ax.errorbar(np.array(dks) + off, diffs, yerr=[np.array(diffs) - np.array(los),
                    np.array(his) - np.array(diffs)], fmt=mk, color=VARIANT_COLORS[v],
                    ecolor=VARIANT_COLORS[v], elinewidth=1.4, capsize=3.5,
                    capthick=1.4, markersize=5.5, zorder=3,
                    label=VARIANT_LABELS[v])
    ax.axhline(0, color=G400, linewidth=1.0, linestyle=":", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dks)
    ax.set_xticklabels([str(d) for d in dks])
    ax.minorticks_off()
    ax.set_xlabel("Head dimension $d_k$")
    ax.set_ylabel("Paired diff, variant − standard (pp)")
    ax.set_ylim(-0.09, 0.11)
    ax.set_title("(b) Gap vs $d_k$ (95% CI)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left", handlelength=1.5)
    clean_axis(ax)
    save(fig, "fig_dk_grid.png")


# ════════════════════════════════════════════════════════════════════════════
# Stage-2 Figure 12: corrected LM perplexity + mask-artifact probes
# ════════════════════════════════════════════════════════════════════════════
def fig_lm_corrected():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) corrected ppl forest (n = 25)
    ax = axes[0]
    arts = STAGE2["lm_corrected"]
    ys = [2, 1, 0]
    items = [("standard", "standard"), ("temp", "temp"), ("dk", "dk")]
    for y, (key, v) in zip(ys, items):
        d = arts[key]
        half = 1.96 * d["sd"] / np.sqrt(d["n"])
        ax.errorbar(d["ppl"], y, xerr=half, fmt="o", markersize=6.5,
                    color=VARIANT_COLORS[v], ecolor=VARIANT_COLORS[v],
                    elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
        ax.annotate(f"{d['ppl']:.1f}", (d["ppl"], y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=G700)
    ax.axvline(1.09, color=SIG, linewidth=1.1, linestyle="--", zorder=2)
    ax.annotate("v1 reported\n(leaky mask): 1.08–1.09", (1.09, 2.62),
                fontsize=7.4, color=SIG, ha="left", va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels(["Std", "τ = 2", "1/d$_k$"])
    ax.set_xlabel("Validation perplexity (95% CI), corrected causal mask")
    ax.set_xlim(830, 915)
    ax.set_ylim(-0.7, 2.75)
    ax.set_title("(a) Corrected LM, 25 seeds")
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid=False)
    ax.xaxis.grid(True, alpha=0.12, color=G300, linewidth=0.7)
    ax.set_axisbelow(True)
    # (b) artifact probes
    ax = axes[1]
    art = STAGE2["lm_artifact"]
    groups = [("3 epochs", art["causal_3ep"]["standard"], art["leaky_3ep"]["standard"]),
              ("10 epochs", np.mean(list(art["causal_10ep"].values())),
               np.mean(list(art["leaky_10ep"].values())))]
    x = np.arange(2)
    w = 0.32
    for i, (lab, col, mk) in enumerate([("causal mask", C_STD, "o"),
                                        ("leaky (v1 bug)", SIG, "x")]):
        vals = [g[1 + i] for g in groups]
        ax.bar(x + (i - 0.5) * w, vals, width=w * 0.9, color=col, alpha=0.92,
               zorder=3, edgecolor="white", linewidth=0.4,
               label=lab if i == 0 else lab)
        for xi, val in zip(x + (i - 0.5) * w, vals):
            ax.annotate(f"{val:.0f}", (xi, val), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=7.8, color=G700)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Validation perplexity (lower = better)")
    ax.set_ylim(0, 1010)
    ax.set_title("(b) Mask-artifact probes (mean of 3 seeds)")
    ax.legend(fontsize=7.4, frameon=False, loc="upper right", handlelength=1.4)
    clean_axis(ax)
    save(fig, "fig_lm_corrected.png")


# ════════════════════════════════════════════════════════════════════════════
# Stage-2 Figure 13: train-to-convergence learning curves (early stopping)
# ════════════════════════════════════════════════════════════════════════════
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    styles = {"standard": (C_STD, "o"), "temp": (C_TMP, "s"), "dk": (C_DK, "^")}
    curves = STAGE2["conv"]["curves"]
    for ax, ds in zip(axes, ["sst2", "imdb"]):
        n_ep = len(curves[ds]["standard"])
        for v in ["standard", "temp", "dk"]:
            c, mk = styles[v]
            y = np.array(curves[ds][v]) * 100
            ax.plot(range(1, n_ep + 1), y, color=c, marker=mk, markersize=5,
                    linewidth=1.8, markerfacecolor="white", markeredgewidth=1.4,
                    label=VARIANT_LABELS[v], zorder=3)
        stop = STAGE2["conv"]["stop_epochs"][ds]
        ax.axvline(stop, color=G400, linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(f"early stop\n(epoch {stop})", (stop, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7.4,
                    color=G500, ha="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("SST-2" if ds == "sst2" else "IMDB")
        lo = min(min(curves[ds][v]) for v in curves[ds]) * 100 - 1.5
        hi = max(max(curves[ds][v]) for v in curves[ds]) * 100 + 1.5
        ax.set_ylim(lo, hi)
        ax.set_xticks(range(1, n_ep + 1))
        clean_axis(ax)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8,
                   handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_convergence.png")


if __name__ == "__main__":
    fig_main_performance()
    fig_paired_differences()
    fig_temperature_ablation()
    fig_training_curves()
    fig_gradient_ratios()
    fig_head_diversity()
    fig_efficiency()
    fig_lm_perplexity()
    fig_stage2_core()
    fig_dk_grid()
    fig_lm_corrected()
    fig_convergence()
    print("All figures done.")
