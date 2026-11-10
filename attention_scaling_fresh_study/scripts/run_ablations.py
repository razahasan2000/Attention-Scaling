"""
Run ablation experiments E1-E4.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
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
                # Merge recursively
                for key, value in data.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value
    return merged


import torch
import numpy as np
from data.prepare import prepare_data, DataCache
from models.transformer_encoder import create_model
from training.trainer import train_classification_model, ClassificationTrainer


def run_ablation_e1_sst2_multiscale(config, cache, device, output_dir, seeds):
    """E1: SST-2 multiscale heads at 5 epochs"""
    dataset_config = config.get('sst2', {})
    full_config = {
        'dataset': dataset_config,
        'training': config.get('training', {}),
        'data': config.get('data', {}),
        'model': config.get('model', {}),
        'training_budgets': {
            'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}
        }
    }
    prepared_data = prepare_data(full_config, cache, seed=seeds[0])
    
    for seed in seeds:
        for variant in ['standard', 'multiscale_v1']:
            model_kwargs = {
                'variant': variant if variant != 'multiscale_v1' else 'standard',
                'tau': 2.0,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'd_ff': 256,
                'dropout': 0.1,
                'max_seq_len': prepared_data.max_len,
                'vocab_size': prepared_data.vocab_size,
                'task_type': 'classification',
                'num_classes': 2,
            }
            if variant == 'multiscale_v1':
                import math
                s = math.sqrt(128 // 4)
                model_kwargs['per_head_scales'] = [s, s, float(128 // 4), float(128 // 4)]
            
            model = create_model(**model_kwargs)
            model.to(device)
            
            run_id = f"sst2_{variant}_seed_{seed}_e1_5epoch"
            results = train_classification_model(
                model=model,
                prepared_data=prepared_data,
                config={
                    'dataset': config.get('sst2', {}),
                    'training': config.get('training', {}),
                    'data': config.get('data', {}),
                    'model': config.get('model', {}),
                    'training_budgets': {'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}},
                    'optimizer': config.get('optimizer', {}),
                    'hardware': config.get('hardware', {}),
                    'dataset_configs': config,
                },
                device=device,
                output_dir=os.path.join(output_dir, 'checkpoints'),
                run_id=run_id,
            )
            print(f"E1 {variant} seed {seed}: test_acc={results.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}")


def run_ablation_e2_sst2_lrcomp(config, cache, device, output_dir, seeds):
    """E2: SST-2 LR compensation for dk variant at 5 epochs"""
    dataset_config = config.get('sst2', {})
    full_config = {
        'dataset': dataset_config,
        'training': config.get('training', {}),
        'data': config.get('data', {}),
        'model': config.get('model', {}),
        'training_budgets': {
            'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}
        }
    }
    prepared_data = prepare_data({'dataset': config.get('sst2', {}), 'training': config.get('training', {}), 'data': config.get('data', {}), 'model': config.get('model', {})}, cache, seed=seeds[0])
    
    for seed in seeds:
        for variant_name in ['standard', 'dk', 'dk_lrcomp']:
            model = create_model(
                variant='dk' if 'dk' in variant_name else 'standard',
                tau=2.0,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=256,
                dropout=0.1,
                max_seq_len=128,
                vocab_size=20000,  # approximate, will be overridden by prepared_data
                task_type='classification',
                num_classes=2,
            )
            model.to(device)
            
            train_config = {
                'learning_rate': 0.0003,
                'weight_decay': 0.01,
                'warmup_frac': 0.05,
                'max_steps': 5 * 100,
                'mixed_precision': False,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
            }
            
            trainer = ClassificationTrainer(model, train_config, torch.device('cuda'), 'checkpoints', f'sst2_{variant_name}_seed_{seed}_e2')
            
            history = trainer.train(
                prepared_data.X_train, prepared_data.y_train,
                prepared_data.X_val, prepared_data.y_val,
                5, 64, 64,
                [1, 2, 3, 4, 5], seed
            )
            
            test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, 64)
            print(f"E2 {variant_name} seed {seed}: test_acc={test_metrics.get('accuracy', 'N/A'):.4f}")


def run_ablation_e3_imdb_multiscale(config, cache, device, output_dir, seeds):
    """E3: IMDB multiscale heads at 5 epochs"""
    dataset_config = config.get('imdb', {})
    full_config = {
        'dataset': dataset_config,
        'training': config.get('training', {}),
        'data': config.get('data', {}),
        'model': config.get('model', {}),
        'training_budgets': {
            'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}
        }
    }
    prepared_data = prepare_data(full_config, cache, seed=seeds[0])
    
    for seed in seeds:
        for variant in ['standard', 'multiscale_v1']:
            model_kwargs = {
                'variant': variant if variant != 'multiscale_v1' else 'standard',
                'tau': 2.0,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'd_ff': 256,
                'dropout': 0.1,
                'max_seq_len': prepared_data.max_len,
                'vocab_size': prepared_data.vocab_size,
                'task_type': 'classification',
                'num_classes': 2,
            }
            if variant == 'multiscale_v1':
                import math
                s = math.sqrt(128 // 4)
                model_kwargs['per_head_scales'] = [s, s, float(128 // 4), float(128 // 4)]
            
            model = create_model(**model_kwargs)
            model.to(device)
            
            run_id = f"imdb_{variant}_seed_{seed}_e3_5epoch"
            results = train_classification_model(
                model=model,
                prepared_data=prepared_data,
                config={
                    'dataset': config.get('imdb', {}),
                    'training': config.get('training', {}),
                    'data': config.get('data', {}),
                    'model': config.get('model', {}),
                    'training_budgets': {'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}},
                    'optimizer': config.get('optimizer', {}),
                    'hardware': config.get('hardware', {}),
                    'dataset_configs': config,
                },
                device=device,
                output_dir=os.path.join(output_dir, 'checkpoints'),
                run_id=run_id,
            )
            print(f"E3 {variant} seed {seed}: test_acc={results.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}")


def run_ablation_e4_imdb_lrcomp(config, cache, device, output_dir, seeds):
    """E4: IMDB LR compensation for dk variant at 5 epochs"""
    dataset_config = config.get('imdb', {})
    full_config = {
        'dataset': dataset_config,
        'training': config.get('training', {}),
        'data': config.get('data', {}),
        'model': config.get('model', {}),
        'training_budgets': {
            'classification_core': {'epochs': 5, 'checkpoint_epochs': [1, 2, 3, 4, 5]}
        }
    }
    prepared_data = prepare_data({'dataset': config.get('imdb', {}), 'training': config.get('training', {}), 'data': config.get('data', {}), 'model': config.get('model', {})}, cache, seed=seeds[0])
    
    for seed in seeds:
        for variant_name in ['standard', 'dk', 'dk_lrcomp']:
            model = create_model(
                variant='dk' if 'dk' in variant_name else 'standard',
                tau=2.0,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=256,
                dropout=0.1,
                max_seq_len=128,
                vocab_size=20000,
                task_type='classification',
                num_classes=2,
            )
            model.to(device)
            
            train_config = {
                'learning_rate': 0.0003,
                'weight_decay': 0.01,
                'warmup_frac': 0.05,
                'max_steps': 5 * 100,
                'mixed_precision': False,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
            }
            
            trainer = ClassificationTrainer(model, train_config, torch.device('cuda'), 'checkpoints', f'imdb_{variant_name}_seed_{seed}_e4')
            
            history = trainer.train(
                prepared_data.X_train, prepared_data.y_train,
                prepared_data.X_val, prepared_data.y_val,
                5, 64, 64,
                [1, 2, 3, 4, 5], seed
            )
            
            test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, 64)
            print(f"E4 {variant_name} seed {seed}: test_acc={test_metrics.get('accuracy', 'N/A'):.4f}")


def main():
    import yaml
    config = merge_configs(r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\configs')
    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\results'
    
    print("=" * 60)
    print("Running Ablation Experiments E1-E4")
    print("=" * 60)
    
    # E1: SST-2 multiscale (10 seeds)
    print("\n--- E1: SST-2 Multiscale (10 seeds) ---")
    run_ablation_e1_sst2_multiscale(config, cache, device, output_dir, list(range(10)))
    
    # E2: SST-2 LR compensation (7 seeds)
    print("\n--- E2: SST-2 LR Compensation (7 seeds) ---")
    run_ablation_e2_sst2_lrcomp(config, cache, device, output_dir, list(range(7)))
    
    # E3: IMDB multiscale (7 seeds)
    print("\n--- E3: IMDB Multiscale (7 seeds) ---")
    run_ablation_e3_imdb_multiscale(config, cache, device, output_dir, list(range(7)))
    
    # E4: IMDB LR compensation (5 seeds)
    print("\n--- E4: IMDB LR Compensation (5 seeds) ---")
    run_ablation_e4_imdb_lrcomp(config, cache, device, output_dir, list(range(5)))
    
    print("\n" + "=" * 60)
    print("All ablation experiments completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()