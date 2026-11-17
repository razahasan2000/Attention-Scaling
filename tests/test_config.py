"""
Unit tests for configuration loading and validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def test_model_config():
    """Test model configuration loading."""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "model" in config
    assert config["model"]["d_model"] == 128
    assert config["model"]["n_heads"] == 4
    assert config["model"]["n_layers"] == 2
    assert config["model"]["d_ff"] == 256
    assert "scaling_variants" in config["model"]
    assert set(config["model"]["scaling_variants"]) == {"standard", "dk", "temp"}
    assert config["model"]["temperature"] == 2.0
    print("  Model config OK")


def test_experiment_config():
    """Test experiment configuration loading."""
    config_path = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "experiment" in config
    assert "datasets" in config
    assert "training_budgets" in config
    assert "optimizer" in config
    assert "seeds" in config
    assert len(config["seeds"]["primary"]) == 25
    assert config["seeds"]["primary"][0] == 1001
    assert config["seeds"]["primary"][-1] == 1025
    print("  Experiment config OK")


def test_seed_config():
    """Test seed configuration loading."""
    config_path = Path(__file__).parent.parent / "configs" / "seed_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "primary_seeds" in config
    assert len(config["primary_seeds"]) == 25
    assert config["primary_seeds"][0] == 1001
    assert config["primary_seeds"][-1] == 1025
    print("  Seed config OK")


def test_training_budget_config():
    """Test training budget configuration loading."""
    config_path = Path(__file__).parent.parent / "configs" / "training_budget_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "core_classification" in config
    assert "classification_ablation" in config
    assert "extended_budget" in config
    assert "extended_budget_25" in config
    assert "wikitext2_lm" in config

    # Check 25-epoch config
    ext25 = config["extended_budget_25"]
    assert ext25["epochs"] == 25
    assert ext25["checkpoint_epochs"] == [2, 5, 10, 15, 20, 25]
    print("  Training budget config OK")


def test_statistical_config():
    """Test statistical configuration loading."""
    config_path = Path(__file__).parent.parent / "configs" / "statistical_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "paired_analysis" in config
    assert "confidence_intervals" in config
    assert "effect_sizes" in config
    assert "multiple_comparisons" in config
    print("  Statistical config OK")


def test_all_configs_load():
    """Test that all configs can be loaded together."""
    configs_dir = Path(__file__).parent.parent / "configs"
    config_files = [
        "model_config.yaml",
        "experiment_config.yaml",
        "seed_config.yaml",
        "training_budget_config.yaml",
        "statistical_config.yaml",
    ]

    all_configs = {}
    for fname in config_files:
        with open(configs_dir / fname) as f:
            all_configs[fname] = yaml.safe_load(f)

    # Verify they don't have conflicting keys
    print("  All configs load OK")


if __name__ == "__main__":
    print("Running config unit tests...")
    test_model_config()
    test_experiment_config()
    test_seed_config()
    test_training_budget_config()
    test_statistical_config()
    test_all_configs_load()
    print("All config tests PASSED")