"""
Full analysis pipeline for all datasets.
Runs: stats + probes + figures + tables + summaries for each dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis import (
    analyze_runs,
    save_analysis_results,
    save_all_tables,
    generate_all_figures,
    run_early_diagnostic_analysis,
    freeze,
    verify_pack,
    generate_overall_summary,
)
from probes import run_all_probes_for_dataset
from data import get_config, DataCache, prepare_dataset
import torch


def analyze_dataset(dataset_name: str, seeds: List[int]):
    """Run full analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name.upper()}")
    print(f"{'='*60}")

    config = get_config(dataset_name)
    device = torch.device(config["training"].get("device", "cuda"))
    cache = DataCache(config["data"]["cache_dir"])
    prepared = prepare_dataset(dataset_name, config, cache)

    artifacts_dir = Path("artifacts") / dataset_name
    runs_dir = artifacts_dir / "runs"

    # 1. Statistics
    task = config["dataset"]["task"]
    results = analyze_runs(runs_dir, task)
    save_analysis_results(results, artifacts_dir / "stats")

    # 2. Probes
    probes = run_all_probes_for_dataset(
        dataset_name, config, prepared, seeds,
        ["standard", "temp", "dk"], artifacts_dir, device
    )

    # 3. Tables
    save_all_tables(results, probes, artifacts_dir / "stats", dataset_name)

    # 4. Figures
    generate_all_figures(results, probes, artifacts_dir / "figures", task)

    # 5. Early diagnostic
    early_result = run_early_diagnostic_analysis(
        dataset_name,
        probes.get("gradient", {}).get("per_run", []),
        results,
        artifacts_dir / "stats",
    )

    print(f"\n{dataset_name.upper()} analysis complete!")
    return results, probes


if __name__ == "__main__":
    from typing import List

    seeds = [0, 1, 2, 3, 4]

    # Analyze all three datasets
    results_sst2, probes_sst2 = analyze_dataset("sst2", seeds)
    results_imdb, probes_imdb = analyze_dataset("imdb", seeds)
    results_wt2, probes_wt2 = analyze_dataset("wikitext2", seeds)

    # 6. Freeze manuscript assets
    print("\n" + "="*60)
    print("Freezing manuscript assets...")
    print("="*60)
    artifacts_root = Path("artifacts")
    freeze_result = freeze(artifacts_root, datasets=["sst2", "imdb", "wikitext2"])
    print(f"Manuscript ready at: {freeze_result['manuscript_ready_dir']}")

    # 7. Verify pack
    print("\n" + "="*60)
    print("Verifying manuscript pack...")
    print("="*60)
    pack_ok = verify_pack(Path("artifacts/manuscript_ready"), verbose=True)

    # 8. Overall summary
    print("\n" + "="*60)
    print("Generating overall summary...")
    print("="*60)
    overall = generate_overall_summary(
        Path("artifacts"),
        ["sst2", "imdb", "wikitext2"],
        Path("artifacts/OVERALL_SUMMARY.md"),
        Path("artifacts/OVERALL_SUMMARY.json"),
    )
    print("Overall summary written.")

    print(f"\nPack verification: {'PASSED' if pack_ok else 'FAILED'}")