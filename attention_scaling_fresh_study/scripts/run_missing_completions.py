"""
Completion runner for missing experiments in the attention scaling fresh study.

Fills, in priority order:
  1. Temperature ablation on SST-2 and IMDB (tau in {0.5, 1.0, 1.5}) with the full
     set of 25 matched seeds (1001-1025). The tau=2.0 arm already exists as the
     'temp' variant records of the core grid and is reused, not re-run.
  2. WikiText-2 LM completion so that all three variants have matched seeds
     1001-1025 (dk 1004-1025, temp 1008-1025, standard 1016-1025).
  3. Temperature ablation on AG News (tau in {0.5, 1.0, 1.5}) with 25 matched seeds.
     [Flagged as a post-hoc extension beyond the preregistered sst2/imdb scope.]

Design guarantees (identical to run_experiments.py):
  - same seed sequence: python/numpy/torch/cuda seeded before model creation
  - scaling variants add no parameters -> bit-identical init across variants/taus
  - identical data ordering via torch.Generator(seed) permutation
  - records saved to results/results/per_seed with unique run ids
  - skip-if-exists resume; per-run ledger for crash recovery
  - the SAME trainer implementation is used as the existing 225 core runs so new
    records are directly comparable (documented scheduler quirk applies uniformly)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments import (
    merge_configs, run_single_classification_experiment, run_single_lm_experiment,
)
from data.prepare import prepare_data, DataCache, save_prepared_data, load_prepared_data

LEADER = Path(__file__).resolve().parents[1] / "results" / "results" / "completion_ledger.json"


def ledger_update(update: dict):
    state = {}
    if LEADER.exists():
        with open(LEADER) as f:
            state = json.load(f)
    state.setdefault("runs", [])
    state.setdefault("started", {})
    # dedupe by run_id (last entry wins)
    rids = {r["run_id"]: r for r in state["runs"]}
    for r in update.get("runs_new", []):
        rids[r["run_id"]] = r
    state["runs"] = list(rids.values())
    for k, v in update.items():
        if k != "runs_new":
            state[k] = v
    state["ledger_updated"] = datetime.now().isoformat()
    LEADER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADER, "w") as f:
        json.dump(state, f, indent=2)


def per_seed_path(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / "results" / "per_seed" / f"{run_id}.json"


def run_if_missing_classification(config, prepared_data, device, output_dir,
                                  variant, dataset, seed, run_id, tau=None):
    path = per_seed_path(output_dir, run_id)
    if path.exists():
        try:
            with open(path) as f:
                rec = json.load(f)
            if rec.get("status") == "completed":
                print(f"  [{run_id}] SKIP (exists)")
                return rec
        except Exception:
            pass
    if tau is not None:
        config['model']['temperature'] = float(tau)
    print(f"  [{run_id}] RUNNING ...")
    rec = run_single_classification_experiment(
        variant=variant, dataset_name=dataset, seed=seed, config=config,
        prepared_data=prepared_data, device=device, output_dir=output_dir,
        run_id=run_id)
    ledger_update({"runs_new": [{"run_id": run_id, "seed": seed, "dataset": dataset,
                                 "variant": variant, "tau": tau,
                                 "test_accuracy": rec.get("test_accuracy"),
                                 "status": rec.get("status"), "ts": rec.get("timestamp")}]})
    return rec


def run_if_missing_lm(config, prepared_data, device, output_dir,
                      variant, seed, run_id):
    path = per_seed_path(output_dir, run_id)
    if path.exists():
        try:
            with open(path) as f:
                rec = json.load(f)
            if rec.get("status") == "completed":
                print(f"  [{run_id}] SKIP (exists)")
                return rec
        except Exception:
            pass
    print(f"  [{run_id}] RUNNING ...")
    rec = run_single_lm_experiment(
        variant=variant, dataset_name="wikitext2", seed=seed, config=config,
        prepared_data=prepared_data, device=device, output_dir=output_dir,
        run_id=run_id)
    ledger_update({"runs_new": [{"run_id": run_id, "seed": seed, "dataset": "wikitext2",
                                 "variant": variant,
                                 "test_perplexity": rec.get("test_perplexity"),
                                 "status": rec.get("status"), "ts": rec.get("timestamp")}]})
    return rec


CLASSIFICATION_RUNS = []   # (priority, dataset, tau, variant, run_fn)


def build_plan():
    plan = []

    def add(stage, dataset, taus, variants=None, seeds=None, lm=False):
        for tau in taus:
            for seed in (seeds or list(range(1001, 1026))):
                for variant in (variants or ["temp"]):
                    if lm:
                        run_id = f"wikitext2_{variant}_seed_{seed}_25000steps"
                        plan.append({"stage": stage, "kind": "lm", "dataset": "wikitext2",
                                     "variant": variant, "seed": seed, "run_id": run_id, "tau": None})
                    else:
                        vname = "temp" if not tau else "temp"
                        suffix = "25epoch" if tau == 2.0 else "2epoch"
                        if tau == 2.0:
                            run_id = f"{dataset}_{variant}_seed_{seed}_25epoch"
                        else:
                            run_id = f"{dataset}_{variant}_tau{tau}_seed_{seed}_2epoch"
                        plan.append({"stage": stage, "kind": "classification", "dataset": dataset,
                                     "variant": "temp", "seed": seed, "run_id": run_id, "tau": tau})

    # Stage 1: SST-2 temp ablation (tau 0.5, 1.0, 1.5) x 25 seeds
    add(1, "sst2", [0.5, 1.0, 1.5])
    # Stage 2: IMDB temp ablation x 25 seeds
    add(2, "imdb", [0.5, 1.0, 1.5])
    # Stage 3a: wikitext2 pairing completion to n=15
    for seed in range(1004, 1016):
        plan.append({"stage": "3a", "kind": "lm", "dataset": "wikitext2",
                     "variant": "dk", "seed": seed,
                     "run_id": f"wikitext2_dk_seed_{seed}_25000steps", "tau": None})
    for seed in range(1008, 1016):
        plan.append({"stage": "3a", "kind": "lm", "dataset": "wikitext2",
                     "variant": "temp", "seed": seed,
                     "run_id": f"wikitext2_temp_seed_{seed}_25000steps", "tau": None})
    # Stage 4: AG News temp ablation x 25 seeds (post-hoc extension)
    add(4, "agnews", [0.5, 1.0, 1.5])
    # Stage 3b: extend wikitext2 to full 25 matched seeds
    for variant, lo in [("dk", 1016), ("temp", 1016), ("standard", 1016)]:
        for seed in range(lo, 1026):
            plan.append({"stage": "3b", "kind": "lm", "dataset": "wikitext2",
                         "variant": variant, "seed": seed,
                         "run_id": f"wikitext2_{variant}_seed_{seed}_25000steps", "tau": None})
    # dedupe plan entries by run_id while preserving order; normalize stage names to str
    seen = set(); uniq = []
    for p in plan:
        p["stage"] = str(p["stage"])
        if p["run_id"] not in seen:
            seen.add(p["run_id"]); uniq.append(p)
    return uniq


def main():
    parser = argparse.ArgumentParser(description='Run missing/incomplete experiments')
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--stages', type=str, default='1,2,3a,4,3b',
                        help='Comma-separated stages to run, in order')
    args = parser.parse_args()

    config = merge_configs(args.config_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    plan = build_plan()
    counts = {}
    plan = [p for p in plan if p["stage"] in stages]
    print(f"Device: {device} | stages: {stages} | runs planned: {len(plan)}")

    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))
    ledger_update({"started": {"ts": datetime.now().isoformat(), "device": str(device),
                               "plan_counts": counts, "stages": stages}})

    prepared = {}
    errors = 0
    done = 0
    for p in plan:
        if errors >= 10:
            print("Too many errors, stopping.")
            break
        ds = p["dataset"]
        if ds not in prepared:
            dataset_config = config.get('dataset_configs', {}).get(ds, {})
            full_config = {
                'dataset': dataset_config,
                'training': config.get('training', {}),
                'data': config.get('data', {}),
                'model': config.get('model', {}),
            }
            data_dir = Path(args.output_dir) / 'data' / ds
            data_file = data_dir / 'data.npz'
            if data_file.exists():
                prepared[ds] = load_prepared_data(data_dir)
                print(f"=== Data loaded for {ds} ===")
            else:
                prepared[ds] = prepare_data(full_config, cache)
                save_prepared_data(prepared[ds], data_dir)
                print(f"=== Data prepared for {ds}, vocab={prepared[ds].vocab_size} ===")
        try:
            if p["kind"] == "lm":
                run_if_missing_lm(config, prepared[ds], device, args.output_dir,
                                  p["variant"], p["seed"], p["run_id"])
            else:
                run_if_missing_classification(config, prepared[ds], device, args.output_dir,
                                              p["variant"], ds, p["seed"], p["run_id"], tau=p["tau"])
            done += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors += 1
            ledger_update({"runs_new": [{"run_id": p["run_id"], "status": "failed",
                                         "error": str(e), "ts": datetime.now().isoformat()}]})
        # restore default tau between runs to avoid leakage
        config['model']['temperature'] = 2.0

    n_ok = 0
    for p in plan:
        path = per_seed_path(args.output_dir, p["run_id"])
        if path.exists():
            with open(path) as f:
                if json.load(f).get("status") == "completed":
                    n_ok += 1
    print(f"Batch complete: {n_ok}/{len(plan)} planned runs recorded as completed.")
    ledger_update({"summary": {"ts": datetime.now().isoformat(),
                               "completed": n_ok, "planned": len(plan), "errors": errors}})


if __name__ == '__main__':
    main()
