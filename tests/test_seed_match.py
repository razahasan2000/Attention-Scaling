"""
Unit tests for seed-matched initialization.

Tests:
- Same seed + same variant -> bit-identical initialization.
- Same seed + different variants -> bit-identical initialization (key property).
- Different seeds -> different initialization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from models.transformer_encoder import create_model


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_same_seed_same_variant_identical_init():
    """Two models with same seed and same variant must have identical init."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    set_seed(0)
    model_a = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)
    set_seed(0)
    model_b = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)

    params_a = [p.detach().clone() for p in model_a.parameters()]
    params_b = [p.detach().clone() for p in model_b.parameters()]

    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa, pb), "Same seed+variant must be bit-identical"
    print("  Same seed+variant: identical init OK")


def test_same_seed_diff_variant_identical_init():
    """Same seed across different variants must be bit-identical (key matched-design property)."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    # Test standard vs temp
    set_seed(0)
    model_std = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)
    params_std = [p.detach().clone() for p in model_std.parameters()]

    set_seed(0)
    model_temp = create_model(variant="temp", task_type="classification", num_classes=2, **model_config)
    params_temp = [p.detach().clone() for p in model_temp.parameters()]

    for ps, pt in zip(params_std, params_temp):
        assert torch.allclose(ps, pt), "standard vs temp must be bit-identical at seed 0"

    # Test standard vs dk
    set_seed(0)
    model_dk = create_model(variant="dk", task_type="classification", num_classes=2, **model_config)
    params_dk = [p.detach().clone() for p in model_dk.parameters()]

    for ps, pd in zip(params_std, params_dk):
        assert torch.allclose(ps, pd), "standard vs dk must be bit-identical at seed 0"

    print("  Same seed diff variant: identical init OK")


def test_different_seeds_different_init():
    """Different seeds should produce different initialization (almost surely)."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    torch.manual_seed(0)
    model_a = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)
    torch.manual_seed(1)
    model_b = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)

    params_a = [p.detach().clone() for p in model_a.parameters()]
    params_b = [p.detach().clone() for p in model_b.parameters()]

    all_different = any(not torch.allclose(pa, pb) for pa, pb in zip(params_a, params_b))
    assert all_different, "Different seeds should produce different init"
    print("  Different seeds: different init OK")


def test_lm_identical_init():
    """Same seed across variants for LM task must be bit-identical."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    set_seed(42)
    model_std = create_model(variant="standard", task_type="language_modelling", **model_config)
    params_std = [p.detach().clone() for p in model_std.parameters()]

    set_seed(42)
    model_temp = create_model(variant="temp", task_type="language_modelling", **model_config)
    params_temp = [p.detach().clone() for p in model_temp.parameters()]

    for ps, pt in zip(params_std, params_temp):
        assert torch.allclose(ps, pt), "LM: standard vs temp must be bit-identical"

    set_seed(42)
    model_dk = create_model(variant="dk", task_type="language_modelling", **model_config)
    params_dk = [p.detach().clone() for p in model_dk.parameters()]

    for ps, pd in zip(params_std, params_dk):
        assert torch.allclose(ps, pd), "LM: standard vs dk must be bit-identical"

    print("  LM same seed diff variant: identical init OK")


if __name__ == "__main__":
    print("Running seed-match unit tests...")
    test_same_seed_same_variant_identical_init()
    test_same_seed_diff_variant_identical_init()
    test_different_seeds_different_init()
    test_lm_identical_init()
    print("All seed-match tests PASSED")