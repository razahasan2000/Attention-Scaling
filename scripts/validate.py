"""
Validation script for attention scaling fresh study.

Runs unit tests and validates the experimental setup.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def run_unit_tests(project_dir: str):
    """Run all unit tests."""
    test_files = [
        'tests/test_attention_scale.py',
        'tests/test_seed_match.py',
        'tests/test_gradient_paths.py',
        'tests/test_config.py',
        'tests/test_data_prepare.py',
    ]

    all_passed = True
    for test_file in test_files:
        test_path = os.path.join(project_dir, test_file)
        if os.path.exists(test_path):
            print(f"\n{'='*60}")
            print(f"Running {test_file}")
            print(f"{'='*60}")
            success = run_command(f"python {test_file}", cwd=project_dir)
            if not success:
                print(f"FAILED: {test_file}")
                all_passed = False
            else:
                print(f"PASSED: {test_file}")
        else:
            print(f"SKIP (not found): {test_file}")

    return all_passed


def check_environment():
    """Check that required packages are installed."""
    required = [
        'torch', 'numpy', 'scipy', 'matplotlib', 'yaml',
        'datasets', 'tqdm', 'transformers', 'tokenizers', 'accelerate', 'sklearn',
    ]

    print("Checking environment...")
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {pkg}: MISSING")
            return False

    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA: Available ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  CUDA: Not available (using CPU)")

    return True


def check_directory_structure(project_dir: str):
    """Check that the project directory structure is correct."""
    required_dirs = [
        'configs',
        'src/models',
        'src/attention',
        'src/data',
        'src/training',
        'src/evaluation',
        'src/analysis',
        'scripts',
        'tests',
        'data',
        'checkpoints',
        'logs',
        'results/raw',
        'results/per_seed',
        'results/aggregated',
        'results/statistical_tests',
        'figures',
        'tables',
        'reports',
        'manuscript',
        'supplementary',
        'environment',
    ]

    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(project_dir, dir_path)
        if os.path.exists(full_path):
            print(f"  {dir_path}: OK")
        else:
            print(f"  {dir_path}: MISSING")
            all_exist = False

    return all_exist


def check_config_files(project_dir: str):
    """Check that all config files exist and are valid."""
    config_files = [
        'configs/model_config.yaml',
        'configs/experiment_config.yaml',
        'configs/seed_config.yaml',
        'configs/training_budget_config.yaml',
        'configs/statistical_config.yaml',
        'configs/dataset_configs.yaml',
    ]

    print("Checking config files...")
    all_exist = True
    for config_file in config_files:
        full_path = os.path.join(project_dir, config_file)
        if os.path.exists(full_path):
            print(f"  {config_file}: OK")
        else:
            print(f"  {config_file}: MISSING")
            all_exist = False

    return all_exist


def validate_preflight(project_dir: str):
    """Run preflight validation checks."""
    print(f"\n{'='*60}")
    print("PREFLIGHT VALIDATION")
    print(f"{'='*60}")

    checks = [
        ("Environment", check_environment),
        ("Directory Structure", lambda: check_directory_structure(project_dir)),
        ("Config Files", lambda: check_config_files(project_dir)),
        ("Unit Tests", lambda: run_unit_tests(project_dir)),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        try:
            passed = check_fn()
            if passed:
                print(f"{name}: PASSED")
            else:
                print(f"{name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"{name}: ERROR - {e}")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL PREFLIGHT CHECKS PASSED")
    else:
        print("SOME PREFLIGHT CHECKS FAILED")
    print(f"{'='*60}")

    return all_passed


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(project_dir)  # Go up from scripts/ to project root

    import argparse
    parser = argparse.ArgumentParser(description='Validate attention scaling fresh study setup')
    parser.add_argument('--project-dir', type=str, default=project_dir,
                        help='Project directory path')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip unit tests')
    args = parser.parse_args()

    success = validate_preflight(args.project_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()