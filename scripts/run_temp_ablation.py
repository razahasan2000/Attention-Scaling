"""
Run temperature ablation experiments with τ = 0.5, 1.0, 1.5, 2.0 on SST-2 and IMDB.
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


def run_temp_ablation(config, cache, device, output_dir, seeds):
    """Run temperature ablation experiments on SST-2 and IMDB."""
    temps = [0.5, 1.0, 1.5, 2.0]
    datasets = ['sst2', 'imdb']
    
    for dataset_name in datasets:
        print(f"\n--- Temperature Ablation on {dataset_name.upper()} ---")
        dataset_config = config.get(dataset_name, {})
        full_config = {
            'dataset': config.get(dataset_name, {}),
            'training': config.get('training', {}),
            'data': config.get('data', {}),
            'model': config.get('model', {}),
        }
        prepared_data = prepare_data(full_config, cache, seed=seeds[0])
        
        for tau in temps:
            print("\n  Temperature tau = " + str(tau))
            for seed in seeds:
                model = create_model(
                    variant='temp',
                    tau=tau,
                    d_model=128,
                    n_heads=4,
                    n_layers=2,
                    d_ff=256,
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
                
                run_id = f"{dataset_name}_temp_tau{tau}_seed_{seed}_2epoch"
                
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
                batch_size = dataset_config_full.get('batch_size', 64)
                eval_batch_size = dataset_config_full.get('eval_batch_size', 64)
                
                history = trainer.train(
                    prepared_data.X_train, prepared_data.y_train,
                    prepared_data.X_val, prepared_data.y_val,
                    2, batch_size, eval_batch_size,
                    [2], seed
                )
                
                test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, eval_batch_size)
                acc = test_metrics.get('accuracy', 'N/A')
                print("    tau=" + str(tau) + " seed=" + str(seed) + ": test_acc=" + "{:.4f}".format(acc))


def main():
    config = merge_configs(r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\configs')
    cache = DataCache(config.get('data', {}).get('cache_dir', './artifacts/cache'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention_scaling_fresh_study\results'
    
    # Temperature ablation seeds from config
    seeds = [1001, 1002, 1003]
    
    print("=" * 60)
    print("Running Temperature Ablation Experiments (tau = 0.5, 1.0, 1.5, 2.0)")
    print("=" * 60)
    
    run_temp_ablation(config, cache, device, output_dir, [1001, 1002, 1003])
    
    print("\n" + "=" * 60)
    print("Temperature ablation experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()