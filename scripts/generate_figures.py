"""
Figure generation for attention scaling fresh study.

Generates publication-quality figures from statistical analysis results.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Any, Optional


# Style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

# Color palette
COLORS = {
    'standard': '#1f77b4',  # blue
    'temp': '#ff7f0e',      # orange
    'dk': '#2ca02c',        # green
    'multiscale_v1': '#d62728',  # red
    'multiscale_v2': '#9467bd',  # purple
    'dk_lrcomp': '#8c564b',      # brown
}

VARIANT_LABELS = {
    'standard': 'Standard ($1/\sqrt{d_k}$)',
    'temp': 'Temperature ($\\tau=2$)',
    'dk': '$1/d_k$',
    'multiscale_v1': 'Multi-scale v1',
    'multiscale_v2': 'Multi-scale v2',
    'dk_lrcomp': '$1/d_k$ + LR comp',
}


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all results files."""
    results = {}

    # Load per-variant stats
    per_variant_path = os.path.join(results_dir, 'per_variant_stats.json')
    if os.path.exists(per_variant_path):
        with open(per_variant_path) as f:
            results['per_variant'] = json.load(f)

    # Load paired comparisons
    paired_path = os.path.join(results_dir, 'paired_comparisons.json')
    if os.path.exists(paired_path):
        with open(paired_path) as f:
            results['paired'] = json.load(f)

    # Load full analysis
    analysis_path = os.path.join(results_dir, 'statistical_analysis.json')
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            results['analysis'] = json.load(f)

    return results


def plot_main_performance(results: Dict, output_path: str, datasets: List[str] = None):
    """
    Plot main performance comparison across datasets and variants.
    """
    if 'per_variant' not in results:
        print("No per_variant results found")
        return

    per_variant = results['per_variant']

    if datasets is None:
        datasets = ['agnews', 'sst2', 'imdb', 'wikitext2']

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 4), squeeze=False)
    axes = axes[0]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Find matching variant results for this dataset
        dataset_results = {}
        for variant, data in per_variant.items():
            if dataset in variant:
                # Extract variant name from key like "agnews_standard"
                base_variant = variant.replace(f"{dataset}_", "")
                dataset_results[base_variant] = data

        if not dataset_results:
            ax.text(0.5, 0.5, f'No data for {dataset}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset.upper())
            continue

        # Determine metric
        if 'mean_accuracy' in list(dataset_results.values())[0]:
            metric_key = 'mean_accuracy'
            err_key = 'se_accuracy'
            ylabel = 'Test Accuracy'
        elif 'mean_perplexity' in list(dataset_results.values())[0]:
            metric_key = 'mean_perplexity'
            err_key = 'se_perplexity'
            ylabel = 'Validation Perplexity'
        else:
            ax.text(0.5, 0.5, f'Unknown metric for {dataset}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        variants = list(dataset_results.keys())
        means = [dataset_results[v][metric_key] for v in variants]
        errors = [dataset_results[v][err_key] for v in variants]
        colors = [COLORS.get(v, '#333333') for v in variants]

        x_pos = np.arange(len(variants))
        bars = ax.bar(x_pos, means, yerr=errors, capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5,
                      alpha=0.8)

        # Add individual seed points
        for i, v in enumerate(variants):
            if 'per_seed_accuracy' in dataset_results[v]:
                seed_vals = dataset_results[v]['per_seed_accuracy']
                ax.scatter([i] * len(seed_vals), seed_vals, color='black', alpha=0.5, s=15, zorder=5)
            elif 'per_seed_perplexity' in dataset_results[v]:
                seed_vals = dataset_results[v]['per_seed_perplexity']
                ax.scatter([i] * len(seed_vals), seed_vals, color='black', alpha=0.5, s=15, zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(dataset.upper())

    plt.suptitle('Predictive Performance by Attention Variant', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved main performance plot to {output_path}")


def plot_paired_differences(results: Dict, output_path: str):
    """
    Plot paired differences with confidence intervals.
    """
    if 'paired' not in results:
        print("No paired comparisons found")
        return

    paired = results['paired']

    # Group by dataset
    dataset_comparisons = {}
    for comp_name, comp_data in paired.items():
        # Extract dataset from comparison name
        # e.g., "agnews_standard_vs_temp" or "standard_vs_temp" within dataset
        parts = comp_name.split('_')
        if len(parts) >= 2:
            dataset = parts[0]
        else:
            dataset = 'unknown'

        if dataset not in dataset_comparisons:
            dataset_comparisons[dataset] = []
        dataset_comparisons[dataset].append((comp_name, comp_data))

    n_datasets = len(dataset_comparisons)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), squeeze=False)
    axes = axes[0]

    for idx, (dataset, comparisons) in enumerate(dataset_comparisons.items()):
        ax = axes[idx]

        comp_names = []
        mean_diffs = []
        ci_lowers = []
        ci_uppers = []
        colors = []
        p_values = []

        for comp_name, comp_data in comparisons:
            variant = comp_data.get('variant', '')
            comp_names.append(VARIANT_LABELS.get(variant, variant))
            mean_diffs.append(comp_data['mean_diff'])
            ci_lowers.append(comp_data['ci_lower'])
            ci_uppers.append(comp_data['ci_upper'])
            colors.append(COLORS.get(variant, '#333333'))
            p_values.append(comp_data.get('p_value_holm', comp_data['p_value']))

        y_pos = np.arange(len(comp_names))

        # Plot confidence intervals
        for i, (mean, lower, upper, color) in enumerate(zip(mean_diffs, ci_lowers, ci_uppers, colors)):
            ax.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
            ax.plot(mean, i, 'o', color=color, markersize=8)

        # Add zero line
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        # Add significance markers
        for i, p in enumerate(p_values):
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            ax.text(0.98, i, sig, ha='right', va='center', transform=ax.transAxes,
                   fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(comp_names)
        ax.set_xlabel('Mean Difference (Baseline - Variant)')
        ax.set_title(f'{dataset.upper()} - Paired Differences (95% CI)')

    plt.suptitle('Seed-Matched Paired Differences vs Standard', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved paired differences plot to {output_path}")


def plot_training_curves(results: Dict, output_path: str, datasets: List[str] = None):
    """
    Plot training curves (validation accuracy/loss over epochs).
    """
    # This would need checkpoint data from run records
    # Placeholder for now
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Training curves plot - requires checkpoint data',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Training Curves (Placeholder)')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot to {output_path}")


def plot_seed_distributions(results: Dict, output_path: str):
    """
    Plot per-seed distributions for each variant/dataset.
    """
    if 'per_variant' not in results:
        print("No per_variant results found")
        return

    per_variant = results['per_variant']

    # Group by dataset
    datasets = {}
    for variant_key, data in per_variant.items():
        # Parse dataset from variant key
        for ds in ['agnews', 'sst2', 'imdb', 'wikitext2']:
            if variant_key.startswith(ds + '_'):
                if ds not in datasets:
                    datasets[ds] = {}
                variant = variant_key.replace(ds + '_', '')
                datasets[ds][variant] = data
                break

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 4), squeeze=False)
    axes = axes[0]

    for idx, (dataset, variants) in enumerate(datasets.items()):
        ax = axes[idx]

        variant_names = list(variants.keys())
        all_data = []

        for v in variant_names:
            if 'per_seed_accuracy' in variants[v]:
                all_data.append(variants[v]['per_seed_accuracy'])
            elif 'per_seed_perplexity' in variants[v]:
                all_data.append(variants[v]['per_seed_perplexity'])

        if not all_data:
            ax.text(0.5, 0.5, f'No seed data for {dataset}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Violin plot
        parts = ax.violinplot(all_data, positions=range(len(variant_names)),
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            color = COLORS.get(variant_names[i], '#333333')
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Overlay individual points
        for i, data in enumerate(all_data):
            x = np.random.normal(i, 0.05, size=len(data))
            ax.scatter(x, data, color='black', alpha=0.5, s=15, zorder=5)

        ax.set_xticks(range(len(variant_names)))
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variant_names], rotation=30, ha='right')
        ax.set_ylabel('Test Accuracy' if 'accuracy' in str(all_data[0]) else 'Perplexity')
        ax.set_title(f'{dataset.upper()} - Per-Seed Distribution')

    plt.suptitle('Per-Seed Distributions by Dataset and Variant', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved seed distributions plot to {output_path}")


def plot_gradient_ratios(results: Dict, output_path: str):
    """
    Plot gradient magnitude ratios relative to standard.
    """
    # Placeholder - requires gradient probe data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Gradient ratios plot - requires probe data',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Gradient Magnitude Ratios (Placeholder)')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved gradient ratios plot to {output_path}")


def plot_head_diversity(results: Dict, output_path: str):
    """
    Plot inter-head cosine similarity and attention entropy.
    """
    # Placeholder - requires head metrics probe data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Head diversity plot - requires probe data',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Head Diversity (Placeholder)')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved head diversity plot to {output_path}")


def plot_temperature_ablation(results: Dict, output_path: str):
    """
    Plot temperature sensitivity ablation results.
    """
    # Placeholder - requires temperature ablation data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Temperature ablation plot - requires ablation data',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Temperature Sensitivity (Placeholder)')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved temperature ablation plot to {output_path}")


def plot_efficiency(results: Dict, output_path: str):
    """
    Plot computational efficiency comparison.
    """
    # Placeholder - requires efficiency data from run records
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Efficiency plot - requires run records',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Computational Efficiency (Placeholder)')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved efficiency plot to {output_path}")


def generate_all_figures(results_dir: str, output_dir: str):
    """
    Generate all figures from analysis results.

    Args:
        results_dir: Directory containing statistical analysis results
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir)

    # Generate figures
    plot_main_performance(results, output_dir / 'fig_main_performance.png')
    plot_paired_differences(results, output_dir / 'fig_paired_differences.png')
    plot_training_curves(results, output_dir / 'fig_training_curves.png')
    plot_seed_distributions(results, output_dir / 'fig_seed_distributions.png')
    plot_gradient_ratios(results, output_dir / 'fig_gradient_ratios.png')
    plot_head_diversity(results, output_dir / 'fig_head_diversity.png')
    plot_temperature_ablation(results, output_dir / 'fig_temperature_ablation.png')
    plot_efficiency(results, output_dir / 'fig_efficiency.png')

    print(f"All figures generated in {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        results_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        results_dir = "results/statistical_tests"
        output_dir = "figures"

    generate_all_figures(results_dir, output_dir)