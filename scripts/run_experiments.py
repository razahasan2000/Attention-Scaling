"""
Main experiment runner for attention scaling fresh study.

Runs the full experiment grid with 25 matched seeds across all variants
and datasets, saving per-seed results and checkpoints.
"""

import os
import sys
import json
import time
import yaml
import argparse
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer_encoder import create_model
from data.prepare import prepare_data, DataCache, save_prepared_data, load_prepared_data
from training.trainer import train_classification_model, train_lm_model


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(configs_dir: str) -> dict:
    """Merge all configuration files into a single config dictionary."""
    config_files = [
        'model_config.yaml',
        'experiment_config.yaml',
        'seed_config.yaml',
        'training_budget_config.yaml',
        'statistical_config.yaml',
    ]

    merged = {}
    for fname in config_files:
        fpath = os.path.join(configs_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f)
                # Merge recursively
                for key, value in data.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value

    # Handle dataset_configs.yaml specially - it has datasets as top-level keys
    dataset_configs_path = os.path.join(configs_dir, 'dataset_configs.yaml')
    if os.path.exists(dataset_configs_path):
        with open(dataset_configs_path) as f:
            data = yaml.safe_load(f)
            merged['dataset_configs'] = data

    return merged


def save_run_record(run_dir: str, run_id: str, run_info: dict):
    """Save individual run record."""
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(run_dir, f"{run_id}.json")
    with open(run_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    return run_path


def run_single_classification_experiment(
    variant: str,
    dataset_name: str,
    seed: int,
    config: dict,
    prepared_data,
    device: torch.device,
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Run a single classification experiment (one variant, one dataset, one seed).
    """
    print(f"  [{run_id}] Starting...")

    # Set random seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model configuration
    model_config = config.get('model', {})
    # Get dataset config from dataset_configs
    dataset_config = config.get('dataset_configs', {}).get(dataset_name, {})
    model = create_model(
        variant=variant,
        tau=model_config.get('temperature', 2.0),
        d_model=model_config.get('d_model', 128),
        n_heads=model_config.get('n_heads', 4),
        n_layers=model_config.get('n_layers', 2),
        d_ff=model_config.get('d_ff', 256),
        dropout=model_config.get('dropout', 0.1),
        max_seq_len=prepared_data.max_len,
        vocab_size=prepared_data.vocab_size,
        task_type='classification',
        num_classes=dataset_config.get('num_classes', 4),
    )
    model.to(device)

    # Training
    checkpoint_dir = os.path.join(output_dir, 'checkpoints', run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        train_results = train_classification_model(
            model=model,
            prepared_data=prepared_data,
            config=config,
            device=device,
            output_dir=checkpoint_dir,
            run_id=run_id,
        )

        # Extract test metrics
        test_metrics = train_results.get('test_metrics', {})
        test_accuracy = test_metrics.get('accuracy', 0)
        test_macro_f1 = test_metrics.get('macro_f1', 0)
        test_loss = test_metrics.get('val_loss', 0)

        run_info = {
            'run_id': run_id,
            'dataset': dataset_name,
            'task': 'classification',
            'variant': variant,
            'seed': seed,
            'config': {
                'd_model': model_config.get('d_model', 128),
                'n_heads': model_config.get('n_heads', 4),
                'n_layers': model_config.get('n_layers', 2),
                'd_ff': model_config.get('d_ff', 256),
                'dropout': model_config.get('dropout', 0.1),
                'scaling': variant,
                'tau': model_config.get('temperature', 2.0),
                'max_seq_len': prepared_data.max_len,
                'vocab_size': prepared_data.vocab_size,
            },
            'training_history': train_results.get('training_history', []),
            'test_accuracy': test_accuracy,
            'test_macro_f1': test_macro_f1,
            'test_loss': test_loss,
            'checkpoint_history': train_results.get('checkpoint_history', []),
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"  [{run_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        run_info = {
            'run_id': run_id,
            'dataset': dataset_name,
            'task': 'classification',
            'variant': variant,
            'seed': seed,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }

    # Save run record
    per_seed_dir = os.path.join(output_dir, 'results', 'per_seed')
    save_run_record(per_seed_dir, run_id, run_info)

    print(f"  [{run_id}] Completed: test_acc={run_info.get('test_accuracy', 'N/A'):.4f}")
    return run_info


def run_single_lm_experiment(
    variant: str,
    dataset_name: str,
    seed: int,
    config: dict,
    prepared_data,
    device: torch.device,
    output_dir: str,
    run_id: str,
) -> dict:
    """
    Run a single language modelling experiment.
    """
    print(f"  [{run_id}] Starting...")

    # Set random seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model configuration
    model_config = config.get('model', {})
    model = create_model(
        variant=variant,
        tau=model_config.get('temperature', 2.0),
        d_model=model_config.get('d_model', 128),
        n_heads=model_config.get('n_heads', 4),
        n_layers=model_config.get('n_layers', 2),
        d_ff=model_config.get('d_ff', 256),
        dropout=model_config.get('dropout', 0.1),
        max_seq_len=prepared_data.max_len,
        vocab_size=prepared_data.vocab_size,
        task_type='language_modelling',
    )
    model.to(device)

    # Training
    checkpoint_dir = os.path.join(output_dir, 'checkpoints', run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        train_results = train_lm_model(
            model=model,
            prepared_data=prepared_data,
            config=config,
            device=device,
            output_dir=checkpoint_dir,
            run_id=run_id,
        )

        # Extract test metrics
        test_metrics = train_results.get('test_metrics', {})
        test_perplexity = test_metrics.get('val_perplexity', 0)
        test_loss = test_metrics.get('val_loss', 0)

        run_info = {
            'run_id': run_id,
            'dataset': dataset_name,
            'task': 'language_modeling',
            'variant': variant,
            'seed': seed,
            'config': {
                'd_model': model_config.get('d_model', 128),
                'n_heads': model_config.get('n_heads', 4),
                'n_layers': model_config.get('n_layers', 2),
                'd_ff': model_config.get('d_ff', 256),
                'dropout': model_config.get('dropout', 0.1),
                'scaling': variant,
                'tau': model_config.get('temperature', 2.0),
                'max_seq_len': prepared_data.max_len,
                'vocab_size': prepared_data.vocab_size,
            },
            'training_history': train_results.get('training_history', []),
            'test_perplexity': test_perplexity,
            'test_loss': test_loss,
            'checkpoint_history': train_results.get('checkpoint_history', []),
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"  [{run_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        run_info = {
            'run_id': run_id,
            'dataset': dataset_name,
            'task': 'language_modeling',
            'variant': variant,
            'seed': seed,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }

    # Save run record
    per_seed_dir = os.path.join(output_dir, 'results', 'per_seed')
    save_run_record(per_seed_dir, run_id, run_info)

    print(f"  [{run_id}] Completed: test_ppl={run_info.get('test_perplexity', 'N/A'):.2f}")
    return run_info


def run_core_grid(config: dict, device: torch.device, output_dir: str, 
                  seeds: list = None, variants: list = None, datasets: list = None) -> list:
    """
    Run the core grid: 3 variants × 4 datasets × 25 seeds = 300 runs.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results', 'per_seed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results', 'aggregated'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results', 'statistical_tests'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Get configuration
    seeds_config = config.get('seeds', {})
    primary_seeds = seeds if seeds is not None else seeds_config.get('primary', list(range(1001, 1026)))
    all_variants = config.get('model', {}).get('scaling_variants', ['standard', 'temp', 'dk'])
    variants = variants if variants is not None else all_variants

    # Dataset configurations
    classification_datasets = datasets if datasets is not None else ['agnews', 'sst2', 'imdb']
    lm_datasets = datasets if datasets is not None and 'wikitext2' in datasets else (['wikitext2'] if datasets is None else [])
    # If specific datasets requested, filter
    if datasets is not None:
        classification_datasets = [d for d in datasets if d in ['agnews', 'sst2', 'imdb']]
        lm_datasets = [d for d in datasets if d == 'wikitext2']

    # Data cache
    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))

    all_results = []
    run_count = 0
    total_runs = len(variants) * (len(classification_datasets) + len(lm_datasets)) * len(primary_seeds)

    print(f"Starting core grid: {len(classification_datasets)} classification + {len(lm_datasets)} LM datasets")
    print(f"Variants: {variants}")
    print(f"Seeds: {len(primary_seeds)} seeds ({primary_seeds[0]}-{primary_seeds[-1]})")
    print(f"Total runs: {total_runs}")

    # Run classification datasets
    for dataset_name in classification_datasets:
        print(f"\n=== Preparing {dataset_name} ===")
        # Build config for this dataset
        dataset_config = config.get('dataset_configs', {}).get(dataset_name, {})
        if dataset_config:
            # Wrap in the expected structure
            full_config = {
                'dataset': dataset_config,
                'training': config.get('training', {}),
                'data': config.get('data', {}),
                'model': config.get('model', {}),
            }
        else:
            full_config = config
        prepared_data = prepare_data(full_config, cache)

        # Save prepared data
        data_dir = os.path.join(output_dir, 'data', dataset_name)
        save_prepared_data(prepared_data, Path(data_dir))
        print(f"Prepared data saved to {data_dir}")

        for variant in variants:
            for seed in primary_seeds:
                run_count += 1
                run_id = f"{dataset_name}_{variant}_seed_{seed}_25epoch"

                # Check if already completed
                per_seed_dir = os.path.join(output_dir, 'results', 'per_seed')
                run_file = os.path.join(per_seed_dir, f"{run_id}.json")
                if os.path.exists(run_file):
                    print(f"  [{run_count}/{total_runs}] {run_id} - SKIP (exists)")
                    with open(run_file) as f:
                        all_results.append(json.load(f))
                    continue

                print(f"  [{run_count}/{total_runs}] {run_id}")

                run_info = run_single_classification_experiment(
                    variant=variant,
                    dataset_name=dataset_name,
                    seed=seed,
                    config=config,
                    prepared_data=prepared_data,
                    device=device,
                    output_dir=output_dir,
                    run_id=run_id,
                )
                all_results.append(run_info)

    # Run WikiText-2
    for dataset_name in lm_datasets:
        print(f"\n=== Preparing {dataset_name} ===")
        # Build config for this dataset
        dataset_config = config.get('dataset_configs', {}).get(dataset_name, {})
        if dataset_config:
            # Wrap in the expected structure
            full_config = {
                'dataset': dataset_config,
                'training': config.get('training', {}),
                'data': config.get('data', {}),
                'model': config.get('model', {}),
            }
        else:
            full_config = config
        prepared_data = prepare_data(full_config, cache)

        # Save prepared data
        data_dir = os.path.join(output_dir, 'data', dataset_name)
        save_prepared_data(prepared_data, Path(data_dir))
        print(f"Prepared data saved to {data_dir}")

        for variant in variants:
            for seed in primary_seeds:
                run_count += 1
                run_id = f"{dataset_name}_{variant}_seed_{seed}_25000steps"

                # Check if already completed
                per_seed_dir = os.path.join(output_dir, 'results', 'per_seed')
                run_file = os.path.join(per_seed_dir, f"{run_id}.json")
                if os.path.exists(run_file):
                    print(f"  [{run_count}/{total_runs}] {run_id} - SKIP (exists)")
                    with open(run_file) as f:
                        all_results.append(json.load(f))
                    continue

                print(f"  [{run_count}/{total_runs}] {run_id}")

                run_info = run_single_lm_experiment(
                    variant=variant,
                    dataset_name=dataset_name,
                    seed=seed,
                    config=config,
                    prepared_data=prepared_data,
                    device=device,
                    output_dir=output_dir,
                    run_id=run_id,
                )
                all_results.append(run_info)

    # Save aggregated results
    aggregated_path = os.path.join(output_dir, 'results', 'aggregated', 'all_results.json')
    os.makedirs(os.path.dirname(aggregated_path), exist_ok=True)
    with open(aggregated_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCore grid complete: {len(all_results)} runs saved to {aggregated_path}")
    return all_results


def run_extended_budget_25epoch(config: dict, device: torch.device, output_dir: str, seeds: list = None) -> list:
    """
    Run extended budget experiments (25 epochs) for SST-2 and IMDB.
    """
    budget_config = config.get('training_budgets', {}).get('extended_budget_25', {})
    epochs = budget_config.get('epochs', 25)
    checkpoint_epochs = budget_config.get('checkpoint_epochs', [2, 5, 10, 15, 20, 25])
    datasets = budget_config.get('datasets', ['sst2', 'imdb'])
    variants = budget_config.get('variants', ['standard', 'temp', 'dk'])
    seeds = seeds if seeds is not None else budget_config.get('seeds', [1001, 1002, 1003, 1004, 1005])

    print(f"\n=== Extended Budget (25 epochs) ===")
    print(f"Datasets: {datasets}")
    print(f"Variants: {variants}")
    print(f"Seeds: {seeds}")

    all_results = []
    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))

    for dataset_name in datasets:
        # Check if data already prepared
        data_dir = os.path.join(output_dir, 'data', dataset_name)
        if os.path.exists(os.path.join(data_dir, 'data.npz')):
            prepared_data = load_prepared_data(Path(data_dir))
        else:
            prepared_data = prepare_data(config.get('dataset_configs', {}).get(dataset_name, config), cache)
            save_prepared_data(prepared_data, Path(data_dir))

        for variant in variants:
            for seed in seeds:
                run_id = f"{dataset_name}_{variant}_seed_{seed}_25epoch"
                per_seed_dir = os.path.join(output_dir, 'results', 'per_seed')
                run_file = os.path.join(per_seed_dir, f"{run_id}.json")
                if os.path.exists(run_file):
                    print(f"  {run_id} - SKIP (exists)")
                    with open(run_file) as f:
                        all_results.append(json.load(f))
                    continue

                print(f"  Running {run_id}...")

                # Need to use the 25-epoch config
                # Temporarily override training budget
                original_budget = config.get('training_budgets', {}).get('classification_core', {})
                config['training_budgets']['classification_core'] = {
                    'epochs': epochs,
                    'checkpoint_epochs': checkpoint_epochs,
                }

                run_info = run_single_classification_experiment(
                    variant=variant,
                    dataset_name=dataset_name,
                    seed=seed,
                    config=config,
                    prepared_data=prepared_data,
                    device=device,
                    output_dir=output_dir,
                    run_id=run_id,
                )

                # Restore original budget
                config['training_budgets']['classification_core'] = original_budget

                all_results.append(run_info)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run attention scaling experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='Path to experiment configuration YAML')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Directory containing all config files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--grid', type=str, default='core',
                        choices=['core', 'extended', 'all'],
                        help='Which experiment grid to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without executing')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to run (default: all 25)')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        help='Variants to run (default: all 3)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to run (default: all 4)')

    args = parser.parse_args()

    # Load all configs
    config = merge_configs(args.config_dir)

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")

    if args.dry_run:
        print("DRY RUN - would execute experiments")
        return

    # Run experiments
    if args.grid == 'core':
        run_core_grid(config, device, args.output_dir, 
                      seeds=args.seeds, variants=args.variants, datasets=args.datasets)
    elif args.grid == 'extended':
        run_extended_budget_25epoch(config, device, args.output_dir, seeds=args.seeds)
    elif args.grid == 'all':
        run_core_grid(config, device, args.output_dir,
                      seeds=args.seeds, variants=args.variants, datasets=args.datasets)
        run_extended_budget_25epoch(config, device, args.output_dir, seeds=args.seeds)

    print("\nExperiment grid complete!")


if __name__ == '__main__':
    main()