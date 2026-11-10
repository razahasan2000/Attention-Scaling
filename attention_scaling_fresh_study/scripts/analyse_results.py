"""
Statistical analysis runner for attention scaling fresh study.

Loads per-seed run records, computes statistics, and saves analysis results.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.analyze_results import (
    load_run_records,
    load_all_run_records,
    compute_per_variant_stats,
    compute_paired_comparisons,
    apply_holm_correction,
    run_statistical_analysis,
    compute_checkpoint_stats,
    save_analysis_results,
)


def load_results_from_dir(results_dir: str):
    """Load all per-seed results from a results directory."""
    per_seed_dir = os.path.join(results_dir, 'results', 'per_seed')
    if not os.path.exists(per_seed_dir):
        print(f"Results directory not found: {per_seed_dir}")
        return []

    records = load_run_records(per_seed_dir)
    print(f"Loaded {len(records)} completed run records from {per_seed_dir}")
    return records


def group_by_dataset(records: list) -> dict:
    """Group run records by dataset."""
    datasets = {}
    for record in records:
        dataset = record.get('dataset', 'unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(record)
    return datasets


def analyze_dataset(records: list, dataset_name: str, output_dir: str):
    """Analyze a single dataset's results."""
    print(f"\n=== Analyzing {dataset_name} ===")

    # Compute per-variant stats
    variant_stats = compute_per_variant_stats(records)
    print(f"Variants found: {list(variant_stats.keys())}")

    for variant, stats in variant_stats.items():
        if 'mean_accuracy' in stats:
            print(f"  {variant}: acc={stats['mean_accuracy']:.4f}±{stats['se_accuracy']:.4f} "
                  f"(n={stats['n']})")
        elif 'mean_perplexity' in stats:
            print(f"  {variant}: ppl={stats['mean_perplexity']:.2f}±{stats['se_perplexity']:.2f} "
                  f"(n={stats['n']})")

    # Run paired comparisons
    analysis = run_statistical_analysis(variant_stats, baseline_variant='standard')

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save per-variant stats
    with open(os.path.join(output_dir, f'{dataset_name}_per_variant_stats.json'), 'w') as f:
        json.dump(variant_stats, f, indent=2)

    # Save paired comparisons
    with open(os.path.join(output_dir, f'{dataset_name}_paired_comparisons.json'), 'w') as f:
        json.dump(analysis['paired_comparisons'], f, indent=2)

    # Save full analysis
    with open(os.path.join(output_dir, f'{dataset_name}_statistical_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print paired comparisons
    print(f"\nPaired comparisons vs standard:")
    for comp_name, comp_data in analysis['paired_comparisons'].items():
        p_raw = comp_data['p_value']
        p_holm = comp_data.get('p_value_holm', p_raw)
        metric = comp_data.get('metric', 'accuracy')
        diff = comp_data['mean_diff']
        ci = f"[{comp_data['ci_lower']:.4f}, {comp_data['ci_upper']:.4f}]"
        dz = comp_data['cohen_dz']
        sig = "***" if p_holm < 0.001 else "**" if p_holm < 0.01 else "*" if p_holm < 0.05 else "ns"
        print(f"  {comp_name}: diff={diff:.4f} {ci}, dz={dz:.2f}, p_raw={p_raw:.6f}, p_holm={p_holm:.6f} {sig}")

    return analysis


def generate_summary_table(all_analyses: dict, output_path: str):
    """Generate a summary table across all datasets."""
    print("\n=== Generating Summary Table ===")

    summary = {
        'datasets': {},
        'overall': {},
    }

    for dataset_name, analysis in all_analyses.items():
        variant_stats = analysis['per_variant_stats']
        paired = analysis['paired_comparisons']

        summary['datasets'][dataset_name] = {
            'variants': {},
            'comparisons': {},
        }

        for variant, stats in variant_stats.items():
            if 'mean_accuracy' in stats:
                summary['datasets'][dataset_name]['variants'][variant] = {
                    'metric': 'accuracy',
                    'mean': stats['mean_accuracy'],
                    'sd': stats['sd_accuracy'],
                    'se': stats['se_accuracy'],
                    'n': stats['n'],
                    'ci_95': [
                        stats['mean_accuracy'] - 1.96 * stats['se_accuracy'],
                        stats['mean_accuracy'] + 1.96 * stats['se_accuracy'],
                    ],
                }
            elif 'mean_perplexity' in stats:
                summary['datasets'][dataset_name]['variants'][variant] = {
                    'metric': 'perplexity',
                    'mean': stats['mean_perplexity'],
                    'sd': stats['sd_perplexity'],
                    'se': stats['se_perplexity'],
                    'n': stats['n'],
                    'ci_95': [
                        stats['mean_perplexity'] - 1.96 * stats['se_perplexity'],
                        stats['mean_perplexity'] + 1.96 * stats['se_perplexity'],
                    ],
                }

        for comp_name, comp_data in paired.items():
            summary['datasets'][dataset_name]['comparisons'][comp_name] = {
                'mean_diff': comp_data['mean_diff'],
                'ci_95': [comp_data['ci_lower'], comp_data['ci_upper']],
                'cohen_dz': comp_data['cohen_dz'],
                'p_value': comp_data['p_value'],
                'p_value_holm': comp_data.get('p_value_holm', comp_data['p_value']),
                'significant': comp_data.get('p_value_holm', comp_data['p_value']) < 0.05,
            }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary table saved to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for dataset_name, data in summary['datasets'].items():
        print(f"\n{dataset_name.upper()}:")
        for variant, stats in data['variants'].items():
            metric = stats['metric']
            print(f"  {variant}: {metric}={stats['mean']:.4f}±{stats['se']:.4f} (n={stats['n']})")

        print("  Comparisons vs standard:")
        for comp_name, comp in data['comparisons'].items():
            sig = "YES" if comp['significant'] else "NO"
            print(f"    {comp_name}: diff={comp['mean_diff']:.4f} [{comp['ci_95'][0]:.4f}, {comp['ci_95'][1]:.4f}], "
                  f"dz={comp['cohen_dz']:.2f}, p_holm={comp['p_value_holm']:.6f} {sig}")


def main():
    parser = argparse.ArgumentParser(description='Run statistical analysis on experiment results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing per-seed results')
    parser.add_argument('--output-dir', type=str, default='results/statistical_tests',
                        help='Output directory for analysis results')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['agnews', 'sst2', 'imdb', 'wikitext2'],
                        help='Datasets to analyze')

    args = parser.parse_args()

    # Load all records
    records = load_results_from_dir(args.results_dir)

    if not records:
        print("No run records found!")
        return

    # Group by dataset
    datasets = group_by_dataset(records)

    # Analyze each dataset
    all_analyses = {}
    for dataset_name in args.datasets:
        if dataset_name in datasets:
            output_dir = os.path.join(args.output_dir, dataset_name)
            analysis = analyze_dataset(datasets[dataset_name], dataset_name, output_dir)
            all_analyses[dataset_name] = analysis
        else:
            print(f"No data found for {dataset_name}")

    # Generate summary table
    summary_path = os.path.join(args.output_dir, 'summary_table.json')
    generate_summary_table(all_analyses, summary_path)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()