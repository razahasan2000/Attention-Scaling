"""
v2 master experiment runner for the Attention-Scaling study.

Completes the full planned matrix and the four v2 work streams, with
per-run resume (existing result JSONs are skipped):

Stages
------
  extended25   Extended budget: 3 variants x 25 seeds x {sst2, imdb},
               25 epochs, checkpoints at 2/5/10/15/20/25      (150 runs)
  lm_causal    WikiText-2 language modelling with STANDARD CAUSAL
               (autoregressive) masking, 3 variants x 25 matched seeds,
               25,000 steps                                     (75 runs)
  dkgrid       Head-dimension sensitivity d_k in {32, 64, 128}
               (d_model = 128 fixed, so n_heads in {4, 2, 1}):
               3 d_k x 3 variants x 25 seeds x 3 datasets      (675 runs)
               Protocol selectable: --protocol core (2 epochs, default,
               comparable operating point) or conv (train to convergence).
  converged    Core matrix re-trained to convergence with early stopping
               on validation loss (patience 5): 3 x 25 x 3     (225 runs)
  seqlen       Sequence-length sensitivity: 3 variants x 3 seeds x
               lengths {64, 256} x 3 datasets, 2 epochs         (54 runs)
  modelsize    Model-size sensitivity {small, baseline, medium} x
               3 variants x 3 seeds                             (27 runs)
  ablations    E1-E4 completion (multiscale_v1 / dk_lrcomp arms,
               5 epochs, seeds 0-based)                         (70 runs)

Usage
-----
  python scripts/run_all_v2.py --dry-run                 # manifest only
  python scripts/run_all_v2.py --stages lm_causal        # one stage
  python scripts/run_all_v2.py --stages dkgrid --protocol conv
  python scripts/run_all_v2.py --limit 3                 # smoke test

Every run writes results/per_seed/{run_id}.json in the original schema,
plus the extra fields: grid, arm, d_k, n_heads, protocol, early_stop_info.
"""

import os
import sys
import json
import math
import random
import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer_encoder import create_model
from data.prepare import prepare_data, DataCache, save_prepared_data
from training.trainer import train_classification_model, train_lm_model

REPO = Path(__file__).resolve().parent.parent

SEEDS_25 = list(range(1001, 1026))
SEEDS_3 = [1001, 1002, 1003]
CLASS_DATASETS = ['agnews', 'sst2', 'imdb']
VARIANTS = ['standard', 'temp', 'dk']
HEAD_DIMS = [32, 64, 128]
MODEL_SIZES = {
    'small': {'d_model': 64, 'n_heads': 2, 'n_layers': 2, 'd_ff': 128},
    'baseline': {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_ff': 256},
    'medium': {'d_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024},
}
ABLATIONS = {
    'E1': {'dataset': 'sst2', 'variants': ['standard', 'multiscale_v1'], 'seeds': list(range(10))},
    'E2': {'dataset': 'sst2', 'variants': ['standard', 'dk', 'dk_lrcomp'], 'seeds': list(range(7))},
    'E3': {'dataset': 'imdb', 'variants': ['standard', 'multiscale_v1'], 'seeds': list(range(7))},
    'E4': {'dataset': 'imdb', 'variants': ['standard', 'dk', 'dk_lrcomp'], 'seeds': list(range(5))},
}
SQRT_DK = math.sqrt(32)


def load_merged_config(config_dir: str) -> dict:
    merged = {}
    for fname in ['model_config.yaml', 'experiment_config.yaml', 'seed_config.yaml',
                  'training_budget_config.yaml', 'statistical_config.yaml']:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
    ds_path = os.path.join(config_dir, 'dataset_configs.yaml')
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            merged['dataset_configs'] = yaml.safe_load(f) or {}
    return merged


# ---------------------------------------------------------------------------
# Stage manifests (pure enumeration, no side effects)
# ---------------------------------------------------------------------------

def build_manifest(args) -> list:
    jobs = []
    v = args.variants or VARIANTS

    if 'extended25' in args.stages:
        for ds in ['sst2', 'imdb']:
            for variant in v:
                for seed in SEEDS_25:
                    jobs.append({
                        'run_id': f"{ds}_{variant}_seed_{seed}_ext25",
                        'kind': 'classification', 'grid': 'extended25',
                        'dataset': ds, 'variant': variant, 'seed': seed,
                        'protocol': 'ext25', 'overrides': {},
                    })

    if 'lm_causal' in args.stages:
        for variant in v:
            for seed in SEEDS_25:
                jobs.append({
                    'run_id': f"wikitext2_{variant}_seed_{seed}_lm25k_causal",
                    'kind': 'lm', 'grid': 'lm_causal',
                    'dataset': 'wikitext2', 'variant': variant, 'seed': seed,
                    'protocol': 'lm25k', 'overrides': {},
                })

    if 'dkgrid' in args.stages:
        protocol = 'conv' if args.protocol == 'conv' else 'core'
        suffix = '_conv' if protocol == 'conv' else '_25epoch'
        for dk in HEAD_DIMS:
            if 128 % dk != 0:
                continue
            n_heads = 128 // dk
            for ds in CLASS_DATASETS:
                for variant in v:
                    for seed in SEEDS_25:
                        jobs.append({
                            'run_id': f"{ds}_{variant}_dk{dk}_seed_{seed}{suffix}",
                            'kind': 'classification', 'grid': f'dk{dk}',
                            'dataset': ds, 'variant': variant, 'seed': seed,
                            'protocol': protocol,
                            'overrides': {'n_heads': n_heads},
                        })

    if 'converged' in args.stages:
        for ds in CLASS_DATASETS:
            for variant in v:
                for seed in SEEDS_25:
                    jobs.append({
                        'run_id': f"{ds}_{variant}_seed_{seed}_conv",
                        'kind': 'classification', 'grid': 'converged',
                        'dataset': ds, 'variant': variant, 'seed': seed,
                        'protocol': 'conv', 'overrides': {},
                    })

    if 'seqlen' in args.stages:
        for length in [64, 256]:
            for ds in CLASS_DATASETS:
                for variant in v:
                    for seed in SEEDS_3:
                        jobs.append({
                            'run_id': f"{ds}_{variant}_len{length}_seed_{seed}_25epoch",
                            'kind': 'classification', 'grid': f'len{length}',
                            'dataset': ds, 'variant': variant, 'seed': seed,
                            'protocol': 'core',
                            'overrides': {'max_len': length},
                        })

    if 'modelsize' in args.stages:
        for size_name, size_cfg in MODEL_SIZES.items():
            if size_name == 'baseline':
                continue  # baseline == core grid
            for ds in CLASS_DATASETS:
                for variant in v:
                    for seed in SEEDS_3:
                        jobs.append({
                            'run_id': f"{ds}_{variant}_size{size_name}_seed_{seed}_25epoch",
                            'kind': 'classification', 'grid': f'size_{size_name}',
                            'dataset': ds, 'variant': variant, 'seed': seed,
                            'protocol': 'core', 'overrides': dict(size_cfg),
                        })

    if 'ablations' in args.stages:
        for abl_id, abl in ABLATIONS.items():
            for variant in abl['variants']:
                for seed in abl['seeds']:
                    jobs.append({
                        'run_id': f"{abl['dataset']}_{variant}_seed_{seed}_abl5",
                        'kind': 'classification', 'grid': f'ablation_{abl_id}',
                        'dataset': abl['dataset'], 'variant': variant, 'seed': seed,
                        'protocol': 'abl5', 'overrides': {},
                    })

    # resume filter + limit
    per_seed_dir = REPO / 'results' / 'per_seed'
    pending, skipped = [], 0
    for job in jobs:
        if (per_seed_dir / f"{job['run_id']}.json").exists():
            skipped += 1
            continue
        pending.append(job)
    if args.limit and args.limit > 0:
        pending = pending[: args.limit]
    return pending, skipped, len(jobs)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DataPool:
    """Prepared-data pool keyed by (dataset, max_len, model-size-relevant)."""

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))
        self._pool = {}

    def get(self, dataset: str, max_len: int = None):
        key = (dataset, max_len)
        if key in self._pool:
            return self._pool[key]
        ds_cfg = copy.deepcopy(self.config['dataset_configs'][dataset])
        if max_len is not None:
            ds_cfg['max_len'] = max_len
        full_cfg = {
            'dataset': ds_cfg,
            'training': self.config.get('training', {}),
            'data': self.config.get('data', {}),
            'model': self.config.get('model', {}),
        }
        prepared = prepare_data(full_cfg, self.cache)
        out_dir = Path(self.output_dir) / 'data' / (
            f"{dataset}" if max_len is None else f"{dataset}_len{max_len}")
        save_prepared_data(prepared, out_dir)
        self._pool[key] = prepared
        return prepared


def run_job(job: dict, config: dict, pool: DataPool, device, output_dir: str) -> dict:
    set_seed(job['seed'])

    if job['kind'] == 'lm':
        prepared = pool.get('wikitext2')
        model = create_model(
            variant=job['variant'],
            tau=config['model'].get('temperature', 2.0),
            d_model=config['model'].get('d_model', 128),
            n_heads=config['model'].get('n_heads', 4),
            n_layers=config['model'].get('n_layers', 2),
            d_ff=config['model'].get('d_ff', 256),
            dropout=config['model'].get('dropout', 0.1),
            max_seq_len=prepared.max_len,
            vocab_size=prepared.vocab_size,
            task_type='language_modelling',  # causal masking automatic in v2
        )
        model.to(device)
        ckpt_dir = os.path.join(output_dir, 'checkpoints', job['run_id'])
        results = train_lm_model(model=model, prepared_data=prepared,
                                 config=config, device=device,
                                 output_dir=ckpt_dir, run_id=job['run_id'])
        test_metrics = results.get('test_metrics', {})
        record = {
            'run_id': job['run_id'], 'grid': job['grid'], 'arm': job['variant'],
            'dataset': 'wikitext2', 'task': 'language_modelling',
            'variant': job['variant'], 'seed': job['seed'],
            'protocol': job['protocol'], 'causal_masking': True,
            'config': {'d_model': config['model'].get('d_model', 128),
                       'n_heads': config['model'].get('n_heads', 4),
                       'n_layers': config['model'].get('n_layers', 2),
                       'd_ff': config['model'].get('d_ff', 256),
                       'scaling': job['variant'], 'max_seq_len': prepared.max_len,
                       'vocab_size': prepared.vocab_size},
            'training_history': results.get('training_history', []),
            'test_loss': test_metrics.get('val_loss'),
            'test_perplexity': test_metrics.get('val_perplexity'),
            'status': 'completed', 'timestamp': datetime.now().isoformat(),
        }
        return record

    # ---- classification ---------------------------------------------------
    overrides = job['overrides']
    max_len = overrides.get('max_len')
    prepared = pool.get(job['dataset'], max_len=max_len)

    model_cfg = copy.deepcopy(config['model'])
    for key in ('d_model', 'n_heads', 'n_layers', 'd_ff'):
        if key in overrides:
            model_cfg[key] = overrides[key]

    variant_alias = job['variant']
    per_head_scales = None
    lrcomp = None
    if job['variant'] == 'multiscale_v1':
        variant_alias = 'multiscale_v1'
        per_head_scales = config['model'].get('multiscale_v1', {}).get(
            'per_head_scales', [SQRT_DK, SQRT_DK, 32.0, 32.0])
    elif job['variant'] == 'dk_lrcomp':
        variant_alias = 'dk'
        lrcomp = {'ratio': SQRT_DK, 'modules': ['W_q', 'W_k']}

    run_cfg = copy.deepcopy(config)
    if lrcomp:
        run_cfg['lr_compensation'] = lrcomp
    if job['protocol'] == 'ext25':
        run_cfg['training_budgets']['classification_core'] = {
            'epochs': 25, 'checkpoint_epochs': [2, 5, 10, 15, 20, 25]}
    elif job['protocol'] == 'abl5':
        run_cfg['training_budgets']['classification_core'] = {
            'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}
    # protocol 'conv' needs no override: the '_conv' run_id suffix makes
    # train_classification_model select training_budgets.convergence.

    model = create_model(
        variant=variant_alias,
        tau=model_cfg.get('temperature', 2.0),
        d_model=model_cfg.get('d_model', 128),
        n_heads=model_cfg.get('n_heads', 4),
        n_layers=model_cfg.get('n_layers', 2),
        d_ff=model_cfg.get('d_ff', 256),
        dropout=model_cfg.get('dropout', 0.1),
        max_seq_len=prepared.max_len,
        vocab_size=prepared.vocab_size,
        task_type='classification',
        num_classes=config['dataset_configs'][job['dataset']].get('num_classes', 4),
        per_head_scales=per_head_scales,
    )
    model.to(device)

    ckpt_dir = os.path.join(output_dir, 'checkpoints', job['run_id'])
    results = train_classification_model(model=model, prepared_data=prepared,
                                         config=run_cfg, device=device,
                                         output_dir=ckpt_dir, run_id=job['run_id'])
    test_metrics = results.get('test_metrics', {})
    dk = model_cfg.get('d_model', 128) // model_cfg.get('n_heads', 4)
    record = {
        'run_id': job['run_id'], 'grid': job['grid'], 'arm': job['variant'],
        'dataset': job['dataset'], 'task': 'classification',
        'variant': job['variant'], 'seed': job['seed'],
        'protocol': job['protocol'],
        'd_k': dk, 'n_heads': model_cfg.get('n_heads', 4),
        'd_model': model_cfg.get('d_model', 128),
        'config': {'scaling': variant_alias,
                   'tau': model_cfg.get('temperature', 2.0),
                   'd_model': model_cfg.get('d_model', 128),
                   'n_heads': model_cfg.get('n_heads', 4),
                   'd_k': dk,
                   'n_layers': model_cfg.get('n_layers', 2),
                   'd_ff': model_cfg.get('d_ff', 256),
                   'max_seq_len': prepared.max_len,
                   'vocab_size': prepared.vocab_size},
        'training_history': results.get('training_history', []),
        'early_stop_info': results.get('early_stop_info', {}),
        'test_accuracy': test_metrics.get('accuracy'),
        'test_macro_f1': test_metrics.get('macro_f1'),
        'test_loss': test_metrics.get('val_loss'),
        'status': 'completed', 'timestamp': datetime.now().isoformat(),
    }
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v2 master experiment runner")
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--stages', type=str, nargs='+', default=[
        'extended25', 'lm_causal', 'dkgrid', 'converged', 'seqlen',
        'modelsize', 'ablations'],
        choices=['extended25', 'lm_causal', 'dkgrid', 'converged',
                 'seqlen', 'modelsize', 'ablations'])
    parser.add_argument('--protocol', type=str, default='core',
                        choices=['core', 'conv'],
                        help='protocol for the dkgrid stage')
    parser.add_argument('--variants', type=str, nargs='+', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0,
                        help='run at most N jobs (smoke testing)')
    args = parser.parse_args()

    os.chdir(REPO)  # results/, artifacts/, configs/ are repo-relative

    pending, skipped, total = build_manifest(args)
    print(f"v2 manifest: {total} jobs | already done: {skipped} | pending: {len(pending)}")
    from collections import Counter
    by_grid = Counter(j['grid'] for j in (pending if args.limit == 0 else pending))
    for grid, n in sorted(by_grid.items()):
        print(f"  {grid:<16} {n}")
    if args.dry_run:
        out = REPO / 'results' / 'v2_manifest.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'total': total, 'skipped_existing': skipped,
                       'pending': pending}, f, indent=2)
        print(f"Manifest written to {out}")
        return

    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    config = load_merged_config(args.config_dir)
    pool = DataPool(config, args.output_dir)
    per_seed_dir = REPO / 'results' / 'per_seed'
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    for i, job in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {job['run_id']} "
              f"(grid={job['grid']}, protocol={job['protocol']})")
        try:
            record = run_job(job, config, pool, device, args.output_dir)
        except Exception as e:  # noqa: BLE001 - keep the matrix running
            import traceback
            traceback.print_exc()
            record = {'run_id': job['run_id'], 'grid': job['grid'],
                      'arm': job['variant'], 'dataset': job['dataset'],
                      'variant': job['variant'], 'seed': job['seed'],
                      'status': 'failed', 'error': str(e),
                      'timestamp': datetime.now().isoformat()}
            failed += 1
        with open(per_seed_dir / f"{job['run_id']}.json", 'w') as f:
            json.dump(record, f, indent=2)
        if record.get('status') == 'completed':
            done += 1
            metric = record.get('test_accuracy', record.get('test_perplexity'))
            print(f"  -> done: {metric}")
        print(f"  progress: {done} ok, {failed} failed")

    print(f"\nFinished: {done} completed, {failed} failed, "
          f"{skipped} previously done. Re-run the same command to resume.")


if __name__ == '__main__':
    main()
