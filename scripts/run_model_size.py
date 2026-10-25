"""
Run model size sensitivity experiments: small, baseline, medium configs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import yaml
from data.prepare import prepare_data, DataCache
from models.transformer_encoder import create_model
from training.trainer import train_classification_model, ClassificationTrainer


def merge_configs(configs_dir: str) -> dict:
    """Merge all configuration files into a single config dictionary."""
    config_files = [
        'model_config.yaml',
        'experiment_config.yaml',
        'seed_config.yaml',
        'training_budget_config.yaml',
        'statistical_config.yaml',
        'dataset_configs.yaml',
    ]

    merged = {}
    for fname in config_files:
        fpath = os.path.join(configs_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value
    return merged


def run_model_size_sensitivity(config, cache, device, output_dir, seeds):
    """Run model size sensitivity experiments on SST-2 and IMDB."""
    model_sizes = {
        'small': {'d_model': 64, 'n_heads': 2, 'n_layers': 2, 'd_ff': 128},
        'baseline': {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_ff': 256},
        'medium': {'d_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024},
    }
    datasets = ['sst2', 'imdb']
    variants = ['standard', 'temp', 'dk']
    
    for dataset_name in datasets:
        print(f"\n--- Model Size Sensitivity on {dataset_name.upper()} ---")
        dataset_config = config.get(dataset_name, {})
        
        for size_name, size_params in model_sizes.items():
            print(f"\n  Model size: {size_name} (d_model={size_params['d_model']}, heads={size_params['n_heads']}, layers={size_params['n_layers']})")
            for variant in variants:
                print(f"    Variant: {variant}")
                for seed in seeds:
                    # Prepare data for this dataset
                    dataset_config = config.get(dataset_name, {})
                    full_config = {
                        'dataset': config.get(dataset_name, {}),
                        'training': config.get('training', {}),
                        'data': config.get('data', {}),
                        'model': config.get('model', {}),
                    }
                    prepared_data = prepare_data({
                        'dataset': config.get(dataset_name, {}),
                        'training': config.get('training', {}),
                        'data': config.get('data', {}),
                        'model': config.get('model', {}),
                    }, cache, seed=seed)
                    
                    model = create_model(
                        variant=variant,
                        tau=2.0,
                        d_model=size_params['d_model'],
                        n_heads=size_params['n_heads'],
                        n_layers=size_params['n_layers'],
                        d_ff=size_params['d_ff'],
                        dropout=0.1,
                        max_seq_len=prepared_data.max_len,
                        vocab_size=prepared_data.vocab_size,
                        task_type='classification',
                        num_classes=dataset_config.get('num_classes', 2),
                    )
                    model.to(device)
                    
                    train_config = {
                        'learning_rate': 0.0003,
                        'weight_decay': 0.01,
                        'warmup_frac': 0.05,
                        'max_steps': 2 * 100,
                        'mixed_precision': False,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8,
                    }
                    
                    run_id = f"{dataset_name}_{variant}_{size_name}_seed_{seed}_2epoch"
                    
                    trainer = ClassificationTrainer(
                        model=model,
                        config={
                            'learning_rate': 0.0003,
                            'weight_decay': 0.01,
                            'warmup_frac': 0.05,
                            'max_steps': 2 * 100,
                            'mixed_precision': False,
                            'betas': (0.9, 0.999),
                            'eps': 1e-8,
                        },
                        device=device,
                        output_dir=os.path.join(output_dir, 'checkpoints'),
                        run_id=run_id,
                    )
                    
                    dataset_config_full = config.get(dataset_name, {})
                    batch_size = dataset_config.get('batch_size', 64)
                    eval_batch_size = dataset_config.get('eval_batch_size', 64)
                    
                    history = trainer.train(
                        prepared_data.X_train, prepared_data.y_train,
                        prepared_data.X_val, prepared_data.y_val,
                        2, batch_size, eval_batch_size,
                        [2], seed
                    )
                    
                    test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, eval_batch_size)
                    acc = test_metrics.get('accuracy', 'N/A')
                    print(f"    {size_name} {variant} seed={seed}: test_acc={acc:.4f}")


def main():
    config = merge_configs(r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\configs')
    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\results'
    
    # Model size sensitivity seeds from config
    seeds = [1001, 1002, 1003]
    
    print("=" * 60)
    print("Running Model Size Sensitivity Experiments (small, baseline, medium)")
    print("=" * 60)
    
    run_model_size_sensitivity(config, cache, device, output_dir, [1001, 1002, 1003])
    
    print("\n" + "=" * 60)
    print("Model size sensitivity experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()