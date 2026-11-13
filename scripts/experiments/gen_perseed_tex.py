"""Generate LaTeX table rows for the Stage-2 per-seed supplementary tables."""

import json
from collections import defaultdict

import os
recs = [json.loads(l) for l in open(os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results")) + "/results.jsonl")]

by = defaultdict(dict)   # (phase, dataset, seed) -> {variant_or_dk_tag: value}


def key_var(r):
    return r["variant"] if r["phase"] != "grid" else f"{r['variant']}@{r['dk']}"


for r in recs:
    if r["phase"] in ("core", "grid", "conv", "abl", "lm"):
        by[(r["phase"], r.get("dataset", "agnews"), r["seed"])][key_var(r)] = (
            r["test_acc"] if "test_acc" in r else r["ppl"])

# ---- core AG News: 25 rows x [std, temp, dk] ----
print("% ---- S14 core replica (AG News) ----")
for s in sorted({k[2] for k in by if k[0] == "core"}):
    d = by[("core", "agnews", s)]
    print(f"{s} & {d['standard']:.4f} & {d['temp']:.4f} & {d['dk']:.4f} \\\\")

# ---- LM corrected: 25 rows ----
print("% ---- S14 LM corrected ----")
for s in sorted({k[2] for k in by if k[0] == "lm"}):
    d = by[("lm", "wikitext2", s)]
    print(f"{s} & {d['standard']:.1f} & {d['temp']:.1f} & {d['dk']:.1f} \\\\")

# ---- grid: 12 rows x [std32, tmp32, dk32, std64, tmp64, dk64, std128, tmp128, dk128] ----
print("% ---- S14 grid ----")
for s in sorted({k[2] for k in by if k[0] == "grid"}):
    d = by[("grid", "agnews", s)]
    cells = []
    for dk in (32, 64, 128):
        for v in ("standard", "temp", "dk"):
            cells.append(f"{d[f'{v}@{dk}']:.4f}")
    print(f"{s} & " + " & ".join(cells) + " \\\\")

# ---- conv: 5 rows x [std_sst, tmp_sst, dk_sst, std_imdb, tmp_imdb, dk_imdb] ----
print("% ---- S14 conv ----")
for s in sorted({k[2] for k in by if k[0] == "conv"}):
    cells = []
    for ds in ("sst2", "imdb"):
        d = by[("conv", ds, s)]
        for v in ("standard", "temp", "dk"):
            cells.append(f"{d[v]:.4f}")
    print(f"{s} & " + " & ".join(cells) + " \\\\")

# ---- abl: 3 rows x [std_sst, dk_sst, lrc_sst, ms_sst, std_imdb, dk_imdb, lrc_imdb, ms_imdb] ----
print("% ---- S14 ablations ----")
for s in sorted({k[2] for k in by if k[0] == "abl"}):
    cells = []
    for ds in ("sst2", "imdb"):
        d = by[("abl", ds, s)]
        for v in ("standard", "dk", "dk_lrcomp", "multiscale_v1"):
            cells.append(f"{d[v]:.4f}")
    print(f"{s} & " + " & ".join(cells) + " \\\\")
