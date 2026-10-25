"""
Analysis module for attention scaling fresh study.

Computes statistics, paired comparisons, and effect sizes from
per-seed run records.
"""

import json
import os
import math
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats


def load_run_records(results_dir: str) -> list:
    """Load all per-seed run records from directory."""
    records = []
    if not os.path.exists(results_dir):
        return records

    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and not filename.endswith('_ERROR.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                record = json.load(f)
                records.append(record)
    return records


def load_all_run_records(results_dir: str) -> list:
    """Load all run records including failed ones."""
    records = []
    if not os.path.exists(results_dir):
        return records

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                record = json.load(f)
                records.append(record)
    return records


def compute_per_variant_stats(records: list) -> dict:
    """
    Compute per-variant statistics from run records.

    Args:
        records: List of run record dictionaries

    Returns:
        Dictionary mapping variant -> stats dict
    """
    # Group results by variant
    variant_results = {}
    for record in records:
        if record.get('status') != 'completed':
            continue
        variant = record.get('variant', 'unknown')
        if variant not in variant_results:
            variant_results[variant] = []
        variant_results[variant].append(record)

    results = {}
    for variant, records_list in variant_results.items():
        # Classification metrics
        accuracies = [r.get('test_accuracy', 0) for r in records_list if 'test_accuracy' in r]
        macro_f1s = [r.get('test_macro_f1', 0) for r in records_list if 'test_macro_f1' in r]
        test_losses = [r.get('test_loss', 0) for r in records_list if 'test_loss' in r]

        # Language modelling metrics
        perplexities = [r.get('test_perplexity', 0) for r in records_list if 'test_perplexity' in r]
        lm_losses = [r.get('test_loss', 0) for r in records_list if 'test_loss' in r and 'test_perplexity' in r]

        if accuracies:
            results[variant] = {
                'n': len(accuracies),
                'mean_accuracy': float(np.mean(accuracies)),
                'sd_accuracy': float(np.std(accuracies, ddof=1)),
                'se_accuracy': float(np.std(accuracies, ddof=1) / math.sqrt(len(accuracies))),
                'mean_macro_f1': float(np.mean(macro_f1s)) if macro_f1s else None,
                'sd_macro_f1': float(np.std(macro_f1s, ddof=1)) if macro_f1s else None,
                'mean_loss': float(np.mean(test_losses)) if test_losses else None,
                'sd_loss': float(np.std(test_losses, ddof=1)) if test_losses else None,
                'per_seed_accuracy': accuracies,
                'per_seed_macro_f1': macro_f1s,
                'per_seed_loss': test_losses,
            }
        elif perplexities:
            results[variant] = {
                'n': len(perplexities),
                'mean_perplexity': float(np.mean(perplexities)),
                'sd_perplexity': float(np.std(perplexities, ddof=1)),
                'se_perplexity': float(np.std(perplexities, ddof=1) / math.sqrt(len(perplexities))),
                'mean_loss': float(np.mean(lm_losses)) if lm_losses else None,
                'sd_loss': float(np.std(lm_losses, ddof=1)) if lm_losses else None,
                'per_seed_perplexity': perplexities,
                'per_seed_loss': lm_losses,
            }

    return results


def compute_paired_comparisons(variant_results: dict, baseline_variant: str = 'standard') -> dict:
    """
    Compute paired statistical comparisons between variants and baseline.

    Args:
        variant_results: Dictionary from compute_per_variant_stats
        baseline_variant: Name of baseline variant

    Returns:
        Dictionary of paired comparison results
    """
    results = {}

    if baseline_variant not in variant_results:
        return results

    # Determine the metric to use
    baseline_data = variant_results[baseline_variant]
    if 'per_seed_accuracy' in baseline_data:
        metric_key = 'per_seed_accuracy'
        metric_name = 'accuracy'
    elif 'per_seed_perplexity' in baseline_data:
        metric_key = 'per_seed_perplexity'
        metric_name = 'perplexity'
    else:
        return results

    baseline_values = baseline_data.get(metric_key, [])

    for variant, data in variant_results.items():
        if variant == baseline_variant:
            continue

        variant_values = data.get(metric_key, [])

        if len(variant_values) != len(baseline_values):
            continue

        # Paired differences (baseline - variant for accuracy, variant - baseline for perplexity)
        if metric_name == 'accuracy':
            diffs = np.array(baseline_values) - np.array(variant_values)
        else:
            diffs = np.array(variant_values) - np.array(baseline_values)

        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        n = len(diffs)
        se_diff = sd_diff / math.sqrt(n)
        t_stat = mean_diff / se_diff if se_diff > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

        # Cohen's dz
        dz = mean_diff / sd_diff if sd_diff > 0 else 0.0

        # Confidence interval
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        # Wilcoxon signed-rank test
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_values, variant_values)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        # Sign test
        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        n_zero = np.sum(diffs == 0)
        n_nonzero = n_pos + n_neg
        if n_nonzero > 0:
            sign_test_p = 2 * min(
                stats.binom.cdf(n_pos, n_nonzero, 0.5),
                stats.binom.cdf(n_nonzero - n_pos, n_nonzero, 0.5),
            )
        else:
            sign_test_p = 1.0

        results[f"{variant}_vs_{baseline_variant}"] = {
            'variant': variant,
            'baseline': baseline_variant,
            'metric': metric_name,
            'n_pairs': n,
            'mean_diff': float(mean_diff),
            'sd_diff': float(sd_diff),
            'se_diff': float(se_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'cohen_dz': float(dz),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'wilcoxon_stat': float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
            'wilcoxon_p': float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
            'sign_test_p': float(sign_test_p),
            'n_positive': int(n_pos),
            'n_negative': int(n_neg),
            'n_zero': int(n_zero),
        }

    return results


def apply_holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Apply Holm-Bonferroni correction to p-values.

    Args:
        p_values: Dictionary mapping comparison name -> raw p-value

    Returns:
        Dictionary mapping comparison name -> adjusted p-value
    """
    # Sort by p-value
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adjusted = {}

    for i, (name, p_val) in enumerate(sorted_items):
        adjusted_p = min(p_val * (m - i), 1.0)
        adjusted[name] = adjusted_p

    return adjusted


def run_statistical_analysis(variant_results: dict,
                            baseline_variant: str = 'standard') -> dict:
    """
    Run complete statistical analysis.

    Args:
        variant_results: Dictionary from compute_per_variant_stats
        baseline_variant: Name of baseline variant

    Returns:
        Complete statistical analysis results
    """
    paired = compute_paired_comparisons(variant_results, baseline_variant)

    # Extract p-values for Holm correction
    p_values = {k: v['p_value'] for k, v in paired.items()}
    adjusted_p = apply_holm_correction(p_values)

    # Add adjusted p-values to paired results
    for k, v in paired.items():
        v['p_value_holm'] = adjusted_p.get(k, v['p_value'])

    results = {
        'per_variant_stats': variant_results,
        'paired_comparisons': paired,
        'holm_adjusted_p_values': adjusted_p,
    }
    return results


def compute_bootstrap_ci(values: List[float], n_bootstrap: int = 10000,
                         confidence_level: float = 0.95,
                         random_state: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: List of values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        random_state: Random seed

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    values = np.array(values)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def compute_log_perplexity_ci(perplexities: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval on log-perplexity scale and back-transform.

    Args:
        perplexities: List of perplexity values
        confidence_level: Confidence level

    Returns:
        Tuple of (ci_lower, ci_upper) on perplexity scale
    """
    log_perps = np.log(perplexities)
    mean_log = np.mean(log_perps)
    sd_log = np.std(log_perps, ddof=1)
    n = len(log_perps)
    se_log = sd_log / math.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence_level) / 2, n - 1)

    ci_lower_log = mean_log - t_crit * se_log
    ci_upper_log = mean_log + t_crit * se_log

    return float(np.exp(ci_lower_log)), float(np.exp(ci_upper_log))


def save_analysis_results(results: dict, output_path: str):
    """Save analysis results to JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_analysis_results(output_path: str) -> dict:
    """Load analysis results from JSON file."""
    import json
    with open(output_path, 'r') as f:
        return json.load(f)


def compute_checkpoint_stats(records: list) -> dict:
    """
    Compute statistics at each checkpoint epoch/step.

    Args:
        records: List of run records with epoch_log or training_history

    Returns:
        Dictionary with checkpoint statistics
    """
    # Group by variant and checkpoint
    checkpoint_data = {}

    for record in records:
        if record.get('status') != 'completed':
            continue
        variant = record.get('variant', 'unknown')
        if variant not in checkpoint_data:
            checkpoint_data[variant] = {}

        # Check for epoch-based training history
        if 'training_history' in record:
            for epoch_data in record['training_history']:
                epoch = epoch_data.get('epoch')
                if epoch is None:
                    continue
                if epoch not in checkpoint_data[variant]:
                    checkpoint_data[variant][epoch] = {'val_accuracy': [], 'val_loss': [], 'val_macro_f1': []}
                if 'val_accuracy' in epoch_data:
                    checkpoint_data[variant][epoch]['val_accuracy'].append(epoch_data['val_accuracy'])
                if 'val_loss' in epoch_data:
                    checkpoint_data[variant][epoch]['val_loss'].append(epoch_data['val_loss'])
                if 'val_macro_f1' in epoch_data:
                    checkpoint_data[variant][epoch]['val_macro_f1'].append(epoch_data['val_macro_f1'])

        # Check for step-based training history
        elif 'epoch_log' in record:
            for epoch_data in record['epoch_log']:
                epoch = epoch_data.get('epoch')
                if epoch is None:
                    continue
                if epoch not in checkpoint_data[variant]:
                    checkpoint_data[variant][epoch] = {'val_accuracy': [], 'val_loss': [], 'val_macro_f1': []}
                if 'val_acc' in epoch_data:
                    checkpoint_data[variant][epoch]['val_accuracy'].append(epoch_data['val_acc'])
                if 'val_loss' in epoch_data:
                    checkpoint_data[variant][epoch]['val_loss'].append(epoch_data['val_loss'])
                if 'val_f1' in epoch_data:
                    checkpoint_data[variant][epoch]['val_macro_f1'].append(epoch_data['val_f1'])

    # Compute statistics
    results = {}
    for variant, epochs in checkpoint_data.items():
        results[variant] = {}
        for epoch, metrics in epochs.items():
            results[variant][epoch] = {}
            for metric_name, values in metrics.items():
                if values:
                    results[variant][epoch][metric_name] = {
                        'mean': float(np.mean(values)),
                        'sd': float(np.std(values, ddof=1)),
                        'n': len(values),
                    }

    return results


def compute_paired_checkpoint_comparisons(checkpoint_stats: dict,
                                          baseline_variant: str = 'standard') -> dict:
    """
    Compute paired comparisons at each checkpoint.

    Args:
        checkpoint_stats: Output from compute_checkpoint_stats
        baseline_variant: Baseline variant name

    Returns:
        Dictionary of paired comparisons per checkpoint
    """
    results = {}

    if baseline_variant not in checkpoint_stats:
        return results

    baseline_epochs = checkpoint_stats[baseline_variant]

    for variant, epochs in checkpoint_stats.items():
        if variant == baseline_variant:
            continue

        for epoch in epochs:
            if epoch not in baseline_epochs:
                continue

            for metric_name in ['val_accuracy', 'val_loss', 'val_macro_f1']:
                if metric_name not in epochs[epoch] or metric_name not in baseline_epochs[epoch]:
                    continue

                # We need per-seed values, but checkpoint_stats only has aggregates
                # This is a limitation - we need per-seed checkpoint data
                pass

    return results


if __name__ == "__main__":
    # Quick test
    print("Analysis module loaded successfully")