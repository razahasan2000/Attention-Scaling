"""
Evaluation module for attention scaling experiments.

Computes metrics for classification and language modelling tasks,
including accuracy, macro-F1, perplexity, and paired statistical tests.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats


def compute_classification_metrics(predictions: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        
    Returns:
        Dictionary with accuracy and macro-F1
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return math.exp(loss)


def compute_paired_statistics(values1: List[float],
                              values2: List[float]) -> Dict[str, float]:
    """
    Compute paired statistical comparison between two conditions.
    
    Args:
        values1: Values for condition 1
        values2: Values for condition 2
        
    Returns:
        Dictionary with paired test statistics
    """
    from scipy import stats
    
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    # Paired differences
    diffs = values1 - values2
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(values1, values2)
    
    # Cohen's dz (paired effect size)
    dz = mean_diff / sd_diff if sd_diff > 0 else 0.0
    
    # 95% CI for mean difference
    se_diff = sd_diff / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # Wilcoxon signed-rank test
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values1, values2)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
    
    # Sign test
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_zero = np.sum(diffs == 0)
    n_nonzero = n_pos + n_neg
    sign_test_p = 2 * min(
        stats.binom.cdf(n_pos, n_nonzero, 0.5),
        stats.binom.cdf(n_nonzero - n_pos, n_nonzero, 0.5)
    )
    
    return {
        "mean_diff": float(mean_diff),
        "sd_diff": float(sd_diff),
        "se_diff": float(se_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cohen_dz": float(dz),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "sign_test_p": float(sign_test_p),
        "n_pairs": n,
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "n_zero": int(n_zero),
    }


def compute_all_metrics(predictions: np.ndarray,
                       labels: np.ndarray,
                       task_type: str = "classification") -> Dict[str, float]:
    """
    Compute all relevant metrics for a dataset.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_type: Task type ("classification" or "language_modelling")
        
    Returns:
        Dictionary of metrics
    """
    if task_type == "classification":
        return compute_classification_metrics(predictions, labels)
    elif task_type == "language_modelling":
        # For language modelling, predictions are perplexity values
        return {
            "mean_perplexity": float(np.mean(predictions)),
            "sd_perplexity": float(np.std(predictions, ddof=1)),
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_all_pairwise_comparisons(results: Dict[str, List[float]],
                                     baseline_variant: str = "standard") -> Dict[str, Dict]:
    """
    Compute paired statistical tests for all variant comparisons.
    
    Args:
        results: Dictionary mapping variant names to lists of per-seed results
        baseline_variant: Name of baseline variant
        
    Returns:
        Dictionary of paired comparison results
    """
    results_dict = {}
    
    for variant, values in results.items():
        if variant == baseline_variant:
            continue
        
        if variant not in results or baseline_variant not in results:
            continue
        
        variant_values = results[variant]
        baseline_values = results[baseline_variant]
        
        if len(variant_values) != len(baseline_values):
            continue
        
        result = compute_paired_statistics(baseline_values, variant_values)
        result_key = f"{variant}_vs_{baseline_variant}"
        results_dict[result_key] = result
    
    return results_dict


def compute_aggregate_statistics(per_seed_results: Dict[str, List[float]]) -> Dict[str, Dict]:
    """
    Compute aggregate statistics for each variant.
    
    Args:
        per_seed_results: Dictionary mapping variant names to lists of per-seed results
        
    Returns:
        Dictionary of aggregate statistics per variant
    """
    results = {}
    
    for variant, values in per_seed_results.items():
        values = np.array(values)
        results[variant] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "ci_lower": float(np.mean(values) - 
                             stats.t.ppf(0.975, len(values) - 1) * 
                             np.std(values, ddof=1) / math.sqrt(len(values))),
            "ci_upper": float(np.mean(values) + 
                             stats.t.ppf(0.975, len(values) - 1) * 
                             np.std(values, ddof=1) / math.sqrt(len(values))),
        }
    
    return results


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(output_path: str) -> Dict:
    """Load results from JSON file."""
    import json
    
    with open(output_path, 'r') as f:
        return json.load(f)