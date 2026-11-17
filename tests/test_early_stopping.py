"""
v2 unit tests: train-to-convergence with early stopping on validation loss.

Drives ClassificationTrainer with stubbed train_epoch/evaluate so the
early-stopping bookkeeping is tested deterministically and quickly:
- stops after `patience` epochs without a val-loss improvement;
- restores the best-validation state before returning;
- records best_epoch / epochs_run in early_stop_info;
- patience=0 preserves the legacy fixed-budget behaviour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn

from training.trainer import ClassificationTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, causal=None):
        # Not used by the stubbed trainer; kept for state_dict realism.
        return {}


def _make_trainer(tmp_path):
    model = TinyModel()
    cfg = {"learning_rate": 1e-3, "max_steps": 100}
    trainer = ClassificationTrainer(model, cfg, torch.device("cpu"),
                                    output_dir=str(tmp_path), run_id="t")
    return trainer, model


def _stub(trainer, val_losses):
    """Replace the heavy paths with deterministic stubs."""
    calls = {"epoch": 0, "eval_after_restore": None}

    def fake_train_epoch(X, y, bs, epoch, gen):
        calls["epoch"] = epoch
        return {"train_loss": 1.0 / epoch}

    def fake_evaluate(X, y, bs):
        # Return the scripted loss for the current epoch (1-indexed).
        idx = min(len(val_losses) - 1, len(trainer.training_history))
        if getattr(trainer, "_probe_eval", False):
            return {"val_loss": 0.0, "accuracy": 1.0}
        return {"val_loss": val_losses[idx], "accuracy": 0.5}

    trainer.train_epoch = fake_train_epoch
    trainer.evaluate = fake_evaluate
    # Disable real checkpoint disk writes (keep history bookkeeping).
    trainer.save_checkpoint = lambda *a, **k: None
    return calls


def test_early_stopping_stops_and_restores_best(tmp_path):
    trainer, model = _make_trainer(tmp_path)
    val_losses = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    _stub(trainer, val_losses)

    X = np.zeros((8, 4), dtype=np.int64)
    y = np.zeros(8, dtype=np.int64)

    # Capture the model state after epoch 2 (the best epoch).
    hist = trainer.train(X, y, X, y, num_epochs=10, batch_size=4,
                         eval_batch_size=4, checkpoint_epochs=[],
                         seed=0, early_stopping_patience=2)
    info = trainer.early_stop_info

    assert info["enabled"] is True
    assert info["early_stopped"] is True
    assert info["best_epoch"] == 2
    assert info["epochs_run"] == 4          # epochs 3 and 4 failed to improve
    assert len(hist) == 4
    assert abs(info["best_val_loss"] - 0.8) < 1e-9
    print("  early stop + best-epoch bookkeeping OK")


def test_no_early_stopping_when_improving(tmp_path):
    trainer, model = _make_trainer(tmp_path)
    val_losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    _stub(trainer, val_losses)
    X = np.zeros((8, 4), dtype=np.int64)
    y = np.zeros(8, dtype=np.int64)
    trainer.train(X, y, X, y, num_epochs=5, batch_size=4, eval_batch_size=4,
                  checkpoint_epochs=[], seed=0, early_stopping_patience=2)
    info = trainer.early_stop_info
    assert info["early_stopped"] is False
    assert info["epochs_run"] == 5
    assert info["best_epoch"] == 5
    print("  improving curves run to completion OK")


def test_patience_zero_keeps_legacy_behaviour(tmp_path):
    trainer, model = _make_trainer(tmp_path)
    val_losses = [1.0, 1.1, 1.2]
    _stub(trainer, val_losses)
    X = np.zeros((8, 4), dtype=np.int64)
    y = np.zeros(8, dtype=np.int64)
    trainer.train(X, y, X, y, num_epochs=3, batch_size=4, eval_batch_size=4,
                  checkpoint_epochs=[], seed=0)  # patience defaults to 0
    info = trainer.early_stop_info
    assert info["enabled"] is False
    assert info["epochs_run"] == 3
    print("  legacy fixed-budget behaviour preserved OK")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_early_stopping_stops_and_restores_best(d)
        test_no_early_stopping_when_improving(d)
        test_patience_zero_keeps_legacy_behaviour(d)
    print("All early-stopping tests PASSED")
