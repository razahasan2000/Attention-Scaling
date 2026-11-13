"""Aggregate Stage-3 operating-point phases (gain / bridge / lmgain / regime).

Writes data/results/stage3_operatingpoint.json with:
  gain      : per-gain accuracy means, paired std-vs-dk/temp stats, probe means
  bridge    : frontier-stabiliser arm summary
  lmgain    : LM perplexity per gain with probe means
  regime    : per-epoch sigma_raw / entropy / KL trajectories (if run)
  frontier  : GPT-2 inference-time divisor swap (if run)
Usage: python3 agg_stage3.py
"""
import json
import os

import numpy as np
from scipy import stats as st

RESULTS = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results"))
V = ("standard", "dk", "temp")
SEEDS4 = (1001, 1002, 1003, 1004)
SEEDS3 = (1001, 1002, 1003)


def paired(recs, phase, sel, a="standard", b="dk", key="test_acc"):
    """Paired a-b stats over matched seeds for a phase selection."""
    rs = {(r["variant"], r["seed"]): r for r in recs
          if r["phase"] == phase and sel(r)}
    seeds = sorted(s for (v, s) in rs if v == a)
    d = np.array([rs[(a, s)][key] - rs[(b, s)][key] for s in seeds])
    t, p = st.ttest_1samp(d, 0.0)
    sd = np.std(d, ddof=1)
    return {"mean_diff": round(float(d.mean()), 5),
            "sd_diff": round(float(sd), 5),
            "dz": round(float(d.mean() / sd), 3) if sd > 0 else 0.0,
            "t": round(float(t), 3), "p": round(float(p), 4), "n": len(seeds)}


def probe_mean(recs, phase, sel):
    pr = [r["probe"] for r in recs if r["phase"] == phase and sel(r)]
    out = {}
    for k in ("sigma_raw", "entropy_own", "maxw_own", "kl_std_dk",
              "kl_std_temp", "sigma_post_own"):
        vals = [p[k] for p in pr if k in p]
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    return out


def main():
    recs = [json.loads(l) for l in open(f"{RESULTS}/results.jsonl")]
    out = {"phases": {}}

    # ---- gain sweep -------------------------------------------------------
    gain = {}
    for g in (1, 2, 4, 8, 16, 32):
        rs = {(r["variant"], r["seed"]): r for r in recs
              if r["phase"] == "gain" and r["gain"] == g}
        accs = {v: float(np.mean([rs[(v, s)]["test_acc"] for s in SEEDS4]))
                for v in V}
        gain[f"g{g}"] = {
            "acc": {v: round(100 * accs[v], 3) for v in V},
            "paired_std_dk": paired(recs, "gain", lambda r, g=g: r["gain"] == g),
            "paired_std_temp": paired(recs, "gain", lambda r, g=g: r["gain"] == g,
                                      b="temp"),
            "probe_std": probe_mean(recs, "gain", lambda r, g=g: r["gain"] == g
                                    and r["variant"] == "standard"),
        }
    out["phases"]["gain"] = gain

    # ---- bridge (frontier stabilisers) ------------------------------------
    bridge = {}
    for spec in ("qk_std", "qk_dk", "qk_temp", "cap3_std_g32", "cap3_dk_g32"):
        rs = [r for r in recs if r["phase"] == "bridge" and r["spec"] == spec]
        bridge[spec] = {
            "acc": round(100 * float(np.mean([r["test_acc"] for r in rs])), 3),
            "probe": probe_mean(recs, "bridge", lambda r, s=spec: r["spec"] == s),
        }
    out["phases"]["bridge"] = bridge

    # ---- lmgain ------------------------------------------------------------
    lmgain = {}
    for g in (1, 4):
        for v in V:
            rs = [r for r in recs if r["phase"] == "lmgain"
                  and r["gain"] == g and r["variant"] == v]
            ppls = [r["ppl"] for r in rs]
            e = {"ppl_mean": round(float(np.mean(ppls)), 2),
                 "ppl_sd": round(float(np.std(ppls, ddof=1)), 2)}
            e.update(probe_mean(recs, "lmgain",
                                lambda r, g=g, v=v: r["gain"] == g
                                and r["variant"] == v))
            lmgain[f"g{g}_{v}"] = e
    out["phases"]["lmgain"] = lmgain

    # ---- regime / regime128 (per-epoch trajectories) -----------------------
    for phase in ("regime", "regime128"):
        rs = [r for r in recs if r["phase"] == phase]
        if not rs:
            continue
        tr = {}
        for v in V:
            runs = [r for r in rs if r["variant"] == v]
            eps = {}
            for r in runs:
                for c in r.get("curve", []):
                    eps.setdefault(c["epoch"], []).append(c)
            tr[v] = []
            for ep in sorted(eps):
                cells = eps[ep]
                row = {"epoch": ep,
                       "val_acc": round(float(np.mean([c["val_acc"] for c in cells])), 4),
                       "test_acc": round(float(np.mean([c["test_acc"] for c in cells])), 4)}
                pr = [c.get("probe", {}) for c in cells]
                for k in ("sigma_raw", "entropy_own", "kl_std_dk", "kl_std_temp"):
                    vals = [p[k] for p in pr if k in p]
                    if vals:
                        row[k] = round(float(np.mean(vals)), 4)
                tr[v].append(row)
        # final-epoch paired differences
        fin = {}
        for a, b in (("standard", "dk"), ("standard", "temp")):
            d = []
            for s in SEEDS3:
                ra = next((r for r in rs if r["variant"] == a and r["seed"] == s), None)
                rb = next((r for r in rs if r["variant"] == b and r["seed"] == s), None)
                if ra and rb and ra.get("curve") and rb.get("curve"):
                    d.append(ra["test_acc"] - rb["test_acc"])
            if len(d) >= 3:
                t, p = st.ttest_1samp(d, 0.0)
                fin[f"{a}_minus_{b}"] = {"mean_pp": round(100 * float(np.mean(d)), 3),
                                         "p": round(float(p), 4), "n": len(d)}
        out["phases"][phase] = {"trajectory": tr, "final_paired": fin,
                                "n_runs": len(rs)}

    # ---- frontier (GPT-2 inference-time divisor swap) ----------------------
    fp = f"{RESULTS}/frontier_gpt2.json"
    if os.path.exists(fp):
        out["phases"]["frontier"] = json.load(open(fp))

    with open(f"{RESULTS}/stage3_operatingpoint.json", "w") as f:
        json.dump(out, f, indent=1)

    # console summary
    print("GAIN SWEEP (AG News, 4 seeds, 2 epochs)")
    print(f"{'g':>3} | {'std%':>6} {'dk%':>6} {'tmp%':>6} | {'sigma_raw':>9} "
          f"{'H_own':>6} {'KL(sd||dk)':>10} {'dz(sd-dk)':>9} {'p':>6}")
    for g in (1, 2, 4, 8, 16, 32):
        e = gain[f"g{g}"]
        pr = e["paired_std_dk"]
        pb = e["probe_std"]
        print(f"{g:>3} | {e['acc']['standard']:6.2f} {e['acc']['dk']:6.2f} "
              f"{e['acc']['temp']:6.2f} | {pb.get('sigma_raw', 0):9.2f} "
              f"{pb.get('entropy_own', 0):6.3f} {pb.get('kl_std_dk', 0):10.4f} "
              f"{pr['dz']:9.2f} {pr['p']:6.3f}")
    print("\nBRIDGE")
    for s, e in bridge.items():
        print(f"  {s:>14}: acc={e['acc']:6.2f} sigma_post={e['probe'].get('sigma_post_own')} "
              f"H={e['probe'].get('entropy_own')} KL(sd||dk)={e['probe'].get('kl_std_dk')}")
    print("\nLMGAIN (WikiText-2, seq 128)")
    for k, e in lmgain.items():
        print(f"  {k:>14}: ppl={e['ppl_mean']:8.1f}+-{e['ppl_sd']:5.1f} "
              f"sigma_raw={e.get('sigma_raw')} KL(sd||dk)={e.get('kl_std_dk')}")
    for phase in ("regime", "regime128"):
        if phase in out["phases"]:
            print(f"\n{phase.upper()}: n={out['phases'][phase]['n_runs']} "
                  f"final_paired={out['phases'][phase]['final_paired']}")
    if "frontier" in out["phases"]:
        print("\nFRONTIER (GPT-2):", json.dumps(out["phases"]["frontier"], indent=1)[:800])


if __name__ == "__main__":
    main()
