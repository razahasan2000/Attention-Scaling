"""
Unit tests for gradient paths.

Tests:
- Backward pass produces non-zero gradients for Q, K, V, logits.
- Gradients are finite.
- Shapes are correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from models.transformer_encoder import create_model


def test_gradient_paths_finite_and_nonzero():
    """Backward through attention should produce finite, non-zero gradients."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    model = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)
    model.train()

    B, T = 2, 16
    x = torch.randint(0, 1000, (B, T))
    labels = torch.randint(0, 2, (B,))

    outputs = model(input_ids=x, labels=labels)
    loss = outputs["loss"]
    assert loss is not None, "Loss should not be None"
    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} gradient should not be None"
        assert torch.isfinite(param.grad).all(), f"{name} gradient should be finite"
        # At least some parameters should have non-zero gradients
        if 'weight' in name and 'embedding' not in name:
            assert param.grad.abs().mean() > 0, f"{name} gradient should be non-zero"

    print("  Gradient paths: finite and non-zero OK")


def test_gradient_shapes():
    """Gradient shapes should match the corresponding parameter shapes."""
    model_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_seq_len': 128,
        'vocab_size': 1000,
    }

    model = create_model(variant="standard", task_type="classification", num_classes=2, **model_config)
    model.train()

    B, T = 2, 16
    x = torch.randint(0, 1000, (B, T))
    labels = torch.randint(0, 2, (B,))

    outputs = model(input_ids=x, labels=labels)
    loss = outputs["loss"]
    loss.backward()

    # Check gradient shapes match parameter shapes
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} gradient should not be None"
        assert param.grad.shape == param.shape, f"{name} gradient shape mismatch: {param.grad.shape} vs {param.shape}"

    print("  Gradient shapes correct OK")


def test_all_variants_gradient_flow():
    """All variants should have proper gradient flow."""
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
        model.train()

        B, T = 2, 16
        x = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 2, (B,))

        outputs = model(input_ids=x, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{variant}: {name} gradient should not be None"
            assert torch.isfinite(param.grad).all(), f"{variant}: {name} gradient should be finite"

        # Zero grads for next iteration
        model.zero_grad()

    print("  All variants gradient flow OK")


def test_lm_gradient_flow():
    """Language modelling task should have proper gradient flow."""
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
        model.train()

        B, T = 2, 16
        x = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 1000, (B, T))
        # Set some labels to -100 (ignore index) to simulate padding
        labels[:, -2:] = -100

        outputs = model(input_ids=x, labels=labels)
        loss = outputs["loss"]
        assert loss is not None, f"{variant}: LM loss should not be None"
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{variant}: {name} gradient should not be None"
            assert torch.isfinite(param.grad).all(), f"{variant}: {name} gradient should be finite"

        model.zero_grad()

    print("  LM gradient flow OK")


if __name__ == "__main__":
    print("Running gradient path unit tests...")
    test_gradient_paths_finite_and_nonzero()
    test_gradient_shapes()
    test_all_variants_gradient_flow()
    test_lm_gradient_flow()
    print("All gradient path tests PASSED")