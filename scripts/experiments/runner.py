"""Phase runner for the Stage-2/Stage-3 replica experiments.

Stage-2 phases:
  lm     : WikiText-2 causal LM, 25 seeds x 3 variants          (75 runs)
  core   : AG News 2-epoch core, 25 seeds x 3 variants          (75 runs)
  grid   : AG News d_k grid {32,64,128}, 12 seeds x 3 variants  (108 runs)
  conv   : SST-2+IMDB to convergence, 5 seeds x 3 variants      (30 runs)
  abl    : E1-E4 ablation arms, 3 seeds x 4 arms x 2 datasets   (24 runs)

Stage-3 operating-point phases (spec field selects the arm):
  gain       : AG News replica + logit gain g in {1,2,4,8}, 4 seeds x 3 variants (48)
  bridge     : AG News replica, g=8, frontier stabilisers (QK-Norm / soft-cap),
               4 seeds x 2 arms                                                  (8)
  regime     : agnews_big (6,000 train), 8 epochs, per-epoch probe, 3 seeds
               x 3 variants, d_k=32                                             (9)
  regime128  : as regime but d_k=128 (1 head)                                    (9)
  lmgain     : WikiText-2 causal LM, seq 128, gain {1,4}, 3 seeds x 3 variants   (18)

Usage: python3 runner.py --phase lm --procs 2
Resume-safe: completed (phase, dataset, variant, dk, seed, spec) keys are skipped.
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C

PROC = C.PROC_DIR
CLS_KW = dict(d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
              max_len=64, task="classification")


def load_cls(name):
    d = np.load(f"{PROC}/{name}.npz")
    return (d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xt"], d["yt"])


def run_one(job):
    """job: (phase, dataset, variant, dk, seed, spec)"""
    phase, ds, variant, dk, seed, spec = job
    C.set_threads()
    t0 = time.time()
    rec = {"phase": phase, "dataset": ds, "variant": variant, "dk": dk,
           "seed": seed, "arm": "", "spec": spec}

    if phase in ("core", "grid"):
        Xtr, ytr, Xva, yva, Xt, yt = load_cls(ds if phase == "core" else "agnews")
        ncls = int(max(ytr.max(), yt.max())) + 1
        n_heads = (128 // dk) if dk else 4
        models = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                                 n_heads=n_heads, n_layers=2, d_ff=256,
                                 dropout=0.1, max_len=64, task="classification",
                                 n_classes=ncls)
        res = C.train_cls(models[variant], Xtr, ytr, Xva, yva, seed,
                          epochs=2, bs=32, lr=3e-4, Xte=Xt, yte=yt)
        rec.update(res)

    elif phase == "gain":
        # Logit-gain sweep: g multiplies raw q.k logits before the divisor,
        # moving the softmax operating point while holding data/init fixed.
        g = int(spec[1:])
        Xtr, ytr, Xva, yva, Xt, yt = load_cls("agnews")
        models = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                                 n_heads=4, n_layers=2, d_ff=256,
                                 dropout=0.1, max_len=64, task="classification",
                                 n_classes=4, logit_gain=g)
        res = C.train_cls(models[variant], Xtr, ytr, Xva, yva, seed,
                          epochs=2, bs=32, lr=3e-4, Xte=Xt, yte=yt)
        res["probe"] = C.probe_summary(
            C.probe(models[variant], torch.from_numpy(Xva[:256]).long()))
        rec["gain"] = g
        rec.update(res)

    elif phase == "bridge":
        # Frontier-stabiliser arms (see jobs_for for the spec list).
        Xtr, ytr, Xva, yva, Xt, yt = load_cls("agnews")
        if spec.startswith("qk"):
            g, kw = 1, {"qk_norm": True}
        else:
            parts = spec.split("_")          # cap3_<variant>_g32
            g, kw = int(parts[2][1:]), {"softcap": 3.0}
        m = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                            n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                            max_len=64, task="classification", n_classes=4,
                            logit_gain=g, **kw)[variant]
        res = C.train_cls(m, Xtr, ytr, Xva, yva, seed, epochs=2, bs=32,
                          lr=3e-4, Xte=Xt, yte=yt)
        res["probe"] = C.probe_summary(
            C.probe(m, torch.from_numpy(Xva[:256]).long()))
        rec["arm"] = spec
        rec["gain"] = g
        rec.update(res)

    elif phase in ("regime", "regime128"):
        # Regime-restored AG News: 6,000 train examples, 8 epochs (12x the
        # replica step budget), instrumented with per-epoch operating-point
        # probes. d_k = 32 (4 heads) or 128 (1 head).
        n_heads = 4 if phase == "regime" else 1
        Xtr, ytr, Xva, yva, Xt, yt = load_cls("agnews_big")
        m = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                            n_heads=n_heads, n_layers=2, d_ff=256, dropout=0.1,
                            max_len=64, task="classification",
                            n_classes=4)[variant]
        ids_probe = torch.from_numpy(Xva[:256]).long()
        res = C.train_cls(m, Xtr, ytr, Xva, yva, seed, epochs=8, bs=32,
                          lr=3e-4, Xte=Xt, yte=yt, record_curve=True,
                          probe_each_epoch=lambda mm: C.probe_summary(C.probe(mm, ids_probe)))
        rec.update(res)

    elif phase == "lmgain":
        # Causal LM at sequence length 128 with logit gain {1, 4}
        g = int(spec.split("_g")[1])
        tr = np.load(f"{PROC}/wikitext2_train.npy")
        va = np.load(f"{PROC}/wikitext2_valid.npy")
        model = C.build_matched(seed, [variant], vocab=8000, d_model=96,
                                n_heads=3, n_layers=2, d_ff=192, dropout=0.1,
                                max_len=128, task="lm",
                                logit_gain=g)[variant]
        ppl, nll = C.train_lm(model, tr, va, seed, epochs=3, bs=32, seq=128,
                              lr=3e-4)
        res = {"ppl": ppl, "val_nll": nll,
               "probe": C.probe_summary(
                   C.probe(model, torch.from_numpy(va[:16 * 128]).reshape(16, 128).long()))}
        rec["gain"] = g
        rec.update(res)

    elif phase == "conv":
        Xtr, ytr, Xva, yva, Xt, yt = load_cls(ds)
        ncls = int(max(ytr.max(), yt.max())) + 1
        model = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                                max_len=64, task="classification",
                                n_classes=ncls)[variant]
        res = C.train_cls(model, Xtr, ytr, Xva, yva, seed, epochs=20, bs=32,
                          lr=3e-4, Xte=Xt, yte=yt, early_stop=3,
                          record_curve=True)
        rec.update(res)

    elif phase == "abl":
        Xtr, ytr, Xva, yva, Xt, yt = load_cls(ds)
        ncls = int(max(ytr.max(), yt.max())) + 1
        # NOTE: for abl jobs the arm arrives in the `variant` slot (arm == "")
        lr = 3e-4 * (math.sqrt(32.0) if variant == "dk_lrcomp" else 1.0)
        model = C.build_matched(seed, [variant], vocab=12000, d_model=128,
                                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                                max_len=64, task="classification",
                                n_classes=ncls)[variant]
        rec["arm"] = variant
        res = C.train_cls(model, Xtr, ytr, Xva, yva, seed, epochs=5, bs=32,
                          lr=lr, Xte=Xt, yte=yt)
        rec.update(res)

    elif phase == "lmleak":
        # v1 failure-mode reproduction: identical LM but WITHOUT the causal
        # mask (bidirectional attention can read the target position).
        tr = np.load(f"{PROC}/wikitext2_train.npy")
        va = np.load(f"{PROC}/wikitext2_valid.npy")
        model = C.build_matched(seed, [variant], vocab=8000, d_model=96,
                                n_heads=3, n_layers=2, d_ff=192, dropout=0.1,
                                max_len=64, task="lm")[variant]
        model.causal_override = False
        ppl, nll = C.train_lm(model, tr, va, seed, epochs=3, bs=32, seq=64,
                              lr=3e-4)
        rec.update({"ppl": ppl, "val_nll": nll})

    elif phase == "lmc":
        # LM convergence probe: 10 epochs, causal or leaky per dataset tag
        leak = ds.endswith("_leak")
        tr = np.load(f"{PROC}/wikitext2_train.npy")
        va = np.load(f"{PROC}/wikitext2_valid.npy")
        model = C.build_matched(seed, [variant], vocab=8000, d_model=96,
                                n_heads=3, n_layers=2, d_ff=192, dropout=0.1,
                                max_len=64, task="lm")[variant]
        if leak:
            model.causal_override = False
        ppl, nll = C.train_lm(model, tr, va, seed, epochs=10, bs=32, seq=64,
                              lr=3e-4)
        rec.update({"ppl": ppl, "val_nll": nll, "leak": leak})

    elif phase == "lm":
        tr = np.load(f"{PROC}/wikitext2_train.npy")
        va = np.load(f"{PROC}/wikitext2_valid.npy")
        model = C.build_matched(seed, [variant], vocab=8000, d_model=96,
                                n_heads=3, n_layers=2, d_ff=192, dropout=0.1,
                                max_len=64, task="lm")[variant]
        ppl, nll = C.train_lm(model, tr, va, seed, epochs=3, bs=32, seq=64,
                              lr=3e-4)
        rec.update({"ppl": ppl, "val_nll": nll})

    rec["wall_s"] = round(time.time() - t0, 1)
    C.append_result(rec)
    return rec


def jobs_for(phase):
    J = []
    seeds25 = list(range(1001, 1026))
    if phase == "lm":
        for s in seeds25:
            for v in C.VARIANTS:
                J.append(("lm", "wikitext2", v, 32, s, ""))
    elif phase == "lmleak":
        for s in range(1001, 1004):
            for v in C.VARIANTS:
                J.append(("lmleak", "wikitext2", v, 32, s, ""))
    elif phase == "lmc":
        for s in range(1001, 1004):
            for tag in ("wikitext2", "wikitext2_leak"):
                for v in C.VARIANTS:
                    J.append(("lmc", tag, v, 32, s, ""))
    elif phase == "core":
        for s in seeds25:
            for v in C.VARIANTS:
                J.append(("core", "agnews", v, 32, s, ""))
    elif phase == "grid":
        for s in range(1001, 1013):
            for dk in (32, 64, 128):
                for v in C.VARIANTS:
                    J.append(("grid", "agnews", v, dk, s, ""))
    elif phase == "conv":
        for s in range(1001, 1006):
            for ds in ("sst2", "imdb"):
                for v in C.VARIANTS:
                    J.append(("conv", ds, v, 32, s, ""))
    elif phase == "abl":
        for s in range(1001, 1003 + 1):
            for ds in ("sst2", "imdb"):
                for arm in ("standard", "dk", "dk_lrcomp", "multiscale_v1"):
                    J.append(("abl", ds, arm, 32, s, ""))
    elif phase == "gain":
        for s in range(1001, 1005):
            for g in (1, 2, 4, 8, 16, 32):
                for v in C.VARIANTS:
                    J.append(("gain", "agnews", v, 32, s, f"g{g}"))
    elif phase == "bridge":
        # Frontier-stabiliser arms. QK-Norm pins the operating point at
        # sigma_post ~ 1.0 under the standard divisor, so all three divisors
        # are compared WITHIN the stabilised (frontier-like) regime:
        #   qk_std / qk_dk / qk_temp : per-head LayerNorm on q,k, gain 1
        #   cap3_std_g32 / cap3_dk_g32 : Gemma-2-style soft cap 3.0 at gain 32
        for s in range(1001, 1005):
            for spec in ("qk_std", "qk_dk", "qk_temp",
                         "cap3_std_g32", "cap3_dk_g32"):
                vmap = {"std": "standard", "dk": "dk", "temp": "temp"}
                v = vmap[spec.split("_")[1]]
                J.append(("bridge", "agnews", v, 32, s, spec))
    elif phase == "regime":
        for s in range(1001, 1004):
            for v in C.VARIANTS:
                J.append(("regime", "agnews_big", v, 32, s, "big"))
    elif phase == "regime128":
        for s in range(1001, 1004):
            for v in C.VARIANTS:
                J.append(("regime128", "agnews_big", v, 128, s, "big_dk128"))
    elif phase == "lmgain":
        for s in range(1001, 1004):
            for g in (1, 4):
                for v in C.VARIANTS:
                    J.append(("lmgain", "wikitext2", v, 32, s, f"s128_g{g}"))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="max jobs this call")
    ap.add_argument("--max-seconds", type=float, default=520.0,
                    help="stop submitting new jobs after this wall time")
    a = ap.parse_args()

    done = C.load_results()
    jobs = [j for j in jobs_for(a.phase)
            if (j[0], j[1], j[2], j[3], j[4], j[5]) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    if not jobs:
        print(f"[{a.phase}] nothing to do")
        return
    print(f"[{a.phase}] {len(jobs)} jobs, {a.procs} procs", flush=True)
    t0 = time.time()
    if a.procs <= 1:
        for j in jobs:
            if time.time() - t0 > a.max_seconds:
                print("  time budget reached; stopping submission", flush=True)
                break
            r = run_one(j)
            print(f"  {j[2]:>14s} dk={j[3]:<3d} seed={j[4]} -> "
                  f"{r.get('test_acc', r.get('ppl')):.4f} ({r['wall_s']}s)", flush=True)
    else:
        with Pool(a.procs) as p:
            outstanding, n_done = [], 0

            def drain_ready(block=False):
                nonlocal n_done
                progressed = False
                for ar in list(outstanding):
                    if block or ar.ready():
                        r = ar.get()
                        n_done += 1
                        acc = r.get("test_acc", r.get("ppl"))
                        print(f"  [{n_done}] {r['variant']:>14s} dk={r['dk']:<3d} "
                              f"seed={r['seed']} -> {acc:.4f} ({r['wall_s']}s)",
                              flush=True)
                        outstanding.remove(ar)
                        progressed = True
                return progressed

            for j in jobs:
                if time.time() - t0 > a.max_seconds:
                    print("  time budget reached; draining queue", flush=True)
                    break
                while len(outstanding) >= a.procs:
                    if not drain_ready():
                        time.sleep(0.4)
                outstanding.append(p.apply_async(run_one, (j,)))
            for ar in outstanding:
                r = ar.get()
                n_done += 1
                acc = r.get("test_acc", r.get("ppl"))
                print(f"  [{n_done}] {r['variant']:>14s} dk={r['dk']:<3d} "
                      f"seed={r['seed']} -> {acc:.4f} ({r['wall_s']}s)", flush=True)
    print(f"[{a.phase}] finished in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
