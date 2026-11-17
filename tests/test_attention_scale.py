"""
Unit tests for attention scaling module.

Tests:
- Different variants have different scale factors (s).
- All variants have identical parameter counts.
- Forward pass produces finite outputs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from models.transformer_encoder import create_model, TransformerEncoder


def test_scale_factors_differ():
    """Each variant should have a distinct scale factor."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    variants = ["standard", "dk", "temp"]
    models = {}
    scales = {}

    for variant in variants:
        model = create_model(variant=variant, task_type="classification", num_classes=2, **model_config)
        models[variant] = model
        # Get scaling info from first layer
        info = model.get_scaling_info()
        scales[variant] = info['per_layer'][0]['scaling_factor']

    assert scales["standard"] != scales["temp"], "standard and temp scales must differ"
    assert scales["standard"] != scales["dk"], "standard and dk scales must differ"
    assert scales["temp"] != scales["dk"], "temp and dk scales must differ"
    print(f"  Scales: {scales}")


def test_identical_param_count():
    """All three variants must have identical parameter counts."""
    vocab_size = 1000
    num_classes = 2
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': vocab_size,
    }

    counts = {}
    for variant in ["standard", "temp", "dk"]:
        model = create_model(variant=variant, task_type="classification", num_classes=num_classes, **model_config)
        counts[variant] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert counts["standard"] == counts["temp"], f"param count mismatch: standard vs temp ({counts['standard']} vs {counts['temp']})"
    assert counts["standard"] == counts["dk"], f"param count mismatch: standard vs dk ({counts['standard']} vs {counts['dk']})"
    print(f"  Param counts: {counts}")


def test_forward_finite():
    """Forward pass should produce finite outputs for all variants."""
    vocab_size = 1000
    num_classes = 2
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': vocab_size,
    }
    B, T = 4, 32
    x = torch.randint(0, vocab_size, (B, T))

    for variant in ["standard", "temp", "dk"]:
        model = create_model(variant=variant, task_type="classification", num_classes=num_classes, **model_config)
        logits = model(input_ids=x, labels=torch.randint(0, num_classes, (B,)))
        assert torch.isfinite(logits["logits"]).all(), f"Non-finite logits for {variant}"
        assert logits["logits"].shape == (B, num_classes), f"Wrong shape for {variant}: {logits['logits'].shape}"
        if logits["loss"] is not None:
            assert torch.isfinite(logits["loss"]).all(), f"Non-finite loss for {variant}"
    print("  Forward pass finite for all variants")


def test_create_model_factory():
    """create_model should return correct model types."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    for variant in ["standard", "temp", "dk"]:
        model = create_model(variant=variant, task_type="classification", num_classes=2, **model_config)
        assert isinstance(model, TransformerEncoder)
        assert model.scaling == variant
        if variant == "temp":
            assert model.tau == 2.0
    print("  create_model factory works for all variants")


def test_lm_head():
    """Language modelling head should be created correctly."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    for variant in ["standard", "temp", "dk"]:
        model = create_model(variant=variant, task_type="language_modelling", **model_config)
        assert isinstance(model, TransformerEncoder)
        assert model.task_type == "language_modelling"
        assert hasattr(model, 'lm_head')
        # Check weight tying
        assert model.lm_head.weight is model.token_embedding.weight
    print("  LM head creation works for all variants")


if __name__ == "__main__":
    print("Running attention module unit tests...")
    test_scale_factors_differ()
    test_identical_param_count()
    test_forward_finite()
    test_create_model_factory()
    test_lm_head()
    print("All attention tests PASSED")