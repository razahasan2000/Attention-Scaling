"""
Training module for attention scaling experiments.

Implements training loops for classification and language modelling tasks
with checkpointing, gradient accumulation, and detailed metric tracking.
Supports both epoch-based (classification) and step-based (LM) training.
"""

import os
import json
import time
import math
import platform
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from models.transformer_encoder import TransformerEncoder, create_model
from data.prepare import PreparedData, save_prepared_data, load_prepared_data


class ClassificationTrainer:
    """
    Trainer class for classification experiments.

    Handles training, validation, checkpointing, and logging for
    classification tasks (AG News, SST-2, IMDB).
    """

    def __init__(self, model: nn.Module, config: dict, device: torch.device,
                 output_dir: str = "checkpoints", run_id: str = "unknown"):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.run_id = run_id

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
        )

        # Initialize scheduler
        total_steps = config.get("max_steps", 1000)
        warmup_steps = int(total_steps * config.get("warmup_frac", 0.05))
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)

        # Mixed precision scaler
        self.use_amp = False  # config.get("mixed_precision", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_checkpoint = None

        # Training history
        self.training_history = []
        self.validation_history = []
        self.checkpoint_history = []

    def _create_scheduler(self, total_steps: int, warmup_steps: int):
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                    batch_size: int, epoch: int, generator: torch.Generator) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Create permutation for this epoch (reproducible)
        perm = torch.randperm(len(X_train), generator=generator).numpy()

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(self.device)
            yb = torch.from_numpy(y_train[idx]).to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=xb,
                        labels=yb,
                    )
                    loss = outputs["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=xb,
                    labels=yb,
                )
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size]).to(self.device)
                yb = torch.from_numpy(y[i:i + batch_size]).to(self.device)

                outputs = self.model(
                    input_ids=xb,
                    labels=yb,
                )

                loss = outputs["loss"]
                total_loss += loss.item()

                if "logits" in outputs:
                    predictions = outputs["logits"].argmax(dim=-1)
                    all_preds.append(predictions.cpu())
                    correct = (predictions == yb).sum().item()
                    total_correct += correct
                    total_samples += yb.size(0)

                num_batches += 1

        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        if total_samples > 0:
            metrics["accuracy"] = total_correct / total_samples
            if len(all_preds) > 0:
                from sklearn.metrics import f1_score
                all_preds_cat = torch.cat(all_preds).numpy()
                metrics["macro_f1"] = f1_score(y[:total_samples], all_preds_cat, average='macro')

        return metrics

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              num_epochs: int, batch_size: int, eval_batch_size: int,
              checkpoint_epochs: List[int], seed: int) -> Dict:
        """Full training loop with checkpointing."""
        # Generator for reproducible batch order
        gen = torch.Generator(device="cpu").manual_seed(seed)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(X_train, y_train, batch_size, epoch, gen)
            epoch_time = time.perf_counter() - t0

            # Validate
            val_metrics = self.evaluate(X_val, y_val, eval_batch_size)

            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics.get('train_loss', 0),
                "val_loss": val_metrics.get('val_loss', 0),
                "val_accuracy": val_metrics.get('accuracy', 0),
                "val_macro_f1": val_metrics.get('macro_f1', 0),
                "epoch_time_s": epoch_time,
                "train_examples_per_s": (len(X_train) / epoch_time) if epoch_time > 0 else 0,
            }
            self.training_history.append(epoch_metrics)

            print(f"Epoch {epoch}/{num_epochs}: "
                  f"train_loss={train_metrics.get('train_loss', 0):.4f}, "
                  f"val_loss={val_metrics.get('val_loss', 0):.4f}, "
                  f"val_acc={val_metrics.get('accuracy', 0):.4f}, "
                  f"val_f1={val_metrics.get('macro_f1', 0):.4f} ({epoch_time:.1f}s)")

            # Save checkpoint if this epoch is in checkpoint_epochs
            if epoch in checkpoint_epochs or epoch == num_epochs:
                checkpoint_metrics = {
                    "stage": f"epoch{epoch}",
                    "train_loss": train_metrics.get('train_loss'),
                    "val_loss": val_metrics.get('val_loss'),
                    "val_accuracy": val_metrics.get('accuracy'),
                    "val_macro_f1": val_metrics.get('macro_f1'),
                }
                self.save_checkpoint(epoch, val_metrics.get("val_loss", float('inf')), checkpoint_metrics)

        return self.training_history

    def save_checkpoint(self, epoch: int, metric: float, checkpoint_metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metric": metric,
            "run_id": self.run_id,
            "checkpoint_metrics": checkpoint_metrics,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.output_dir / f"{self.run_id}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(str(checkpoint_path))

        # Save best model
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = checkpoint_path
            best_path = self.output_dir / f"{self.run_id}_best.pt"
            torch.save(checkpoint, best_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("metric", float('inf'))

        return checkpoint


class LMTrainer:
    """
    Trainer class for language modelling experiments (WikiText-2).

    Handles step-based training with evaluation at specified intervals.
    """

    def __init__(self, model: nn.Module, config: dict, device: torch.device,
                 output_dir: str = "checkpoints", run_id: str = "unknown"):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.run_id = run_id

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
        )

        # Initialize scheduler
        total_steps = config.get("max_steps", 25000)
        warmup_steps = int(total_steps * config.get("warmup_frac", 0.05))
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)

        # Mixed precision scaler
        self.use_amp = config.get("mixed_precision", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_step = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_checkpoint = None

        # Training history
        self.training_history = []
        self.validation_history = []
        self.checkpoint_history = []

    def _create_scheduler(self, total_steps: int, warmup_steps: int):
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_ids=xb,
                    labels=yb,
                )
                loss = outputs["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                input_ids=xb,
                labels=yb,
            )
            loss = outputs["loss"]
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size]).to(self.device)
                yb = torch.from_numpy(y[i:i + batch_size]).to(self.device)

                outputs = self.model(
                    input_ids=xb,
                    labels=yb,
                )

                loss = outputs["loss"]
                # Count non-padding tokens
                num_tokens = (yb != -100).sum().item() if (yb != -100).any() else yb.numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

        metrics = {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }
        return metrics

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              max_steps: int, batch_size: int, eval_batch_size: int,
              eval_every: int, checkpoint_steps: List[int], seed: int) -> Dict:
        """Full step-based training loop with periodic evaluation and checkpointing."""
        # Generator for reproducible batch order
        gen = torch.Generator(device="cpu").manual_seed(seed)

        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
        epoch = 0

        t_train_total0 = time.perf_counter()

        while self.global_step < max_steps:
            epoch += 1
            # Create permutation for this epoch
            perm = torch.randperm(len(X_train), generator=gen).numpy()

            for i in range(0, len(perm), batch_size):
                if self.global_step >= max_steps:
                    break

                idx = perm[i:i + batch_size]
                xb = torch.from_numpy(X_train[idx]).to(self.device)
                yb = torch.from_numpy(y_train[idx]).to(self.device)

                t0 = time.perf_counter()
                loss = self.train_step(xb, yb)
                step_time = time.perf_counter() - t0

                self.current_step = self.global_step

                # Evaluate at specified intervals
                if self.global_step % eval_every == 0 or self.global_step in checkpoint_steps:
                    val_metrics = self.evaluate(X_val, y_val, eval_batch_size)

                    step_metrics = {
                        "step": self.global_step,
                        "train_loss": loss,
                        "val_loss": val_metrics.get('val_loss', 0),
                        "val_perplexity": val_metrics.get('val_perplexity', 0),
                        "step_time_s": step_time,
                        "train_examples_per_s": batch_size / step_time if step_time > 0 else 0,
                    }
                    self.training_history.append(step_metrics)
                    self.validation_history.append(val_metrics)

                    print(f"Step {self.global_step}/{max_steps}: "
                          f"train_loss={loss:.4f}, "
                          f"val_loss={val_metrics.get('val_loss', 0):.4f}, "
                          f"val_ppl={val_metrics.get('val_perplexity', 0):.2f}")

                    # Save checkpoint
                    if self.global_step in checkpoint_steps or self.global_step == max_steps:
                        checkpoint_metrics = {
                            "stage": f"step{self.global_step}",
                            "train_loss": loss,
                            "val_loss": val_metrics.get('val_loss'),
                            "val_perplexity": val_metrics.get('val_perplexity'),
                        }
                        self.save_checkpoint(self.global_step, val_metrics.get("val_loss", float('inf')), checkpoint_metrics)

        total_train_time = time.perf_counter() - t_train_total0
        return self.training_history

    def save_checkpoint(self, step: int, metric: float, checkpoint_metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metric": metric,
            "run_id": self.run_id,
            "checkpoint_metrics": checkpoint_metrics,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.output_dir / f"{self.run_id}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(str(checkpoint_path))

        # Save best model
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = checkpoint_path
            best_path = self.output_dir / f"{self.run_id}_best.pt"
            torch.save(checkpoint, best_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_step = checkpoint.get("step", 0)
        self.best_metric = checkpoint.get("metric", float('inf'))

        return checkpoint


def train_classification_model(model: nn.Module, prepared_data: PreparedData,
                               config: dict, device: torch.device,
                               output_dir: str, run_id: str) -> Dict:
    """
    Convenience function to train a classification model.

    Args:
        model: Model to train
        prepared_data: PreparedData object with train/val/test splits
        config: Training configuration
        device: Device to train on
        output_dir: Output directory for checkpoints
        run_id: Unique run identifier

    Returns:
        Training history dictionary
    """
    trainer_config = {
        'learning_rate': config.get('optimizer', {}).get('lr', 3e-4),
        'weight_decay': config.get('optimizer', {}).get('weight_decay', 0.01),
        'warmup_frac': config.get('optimizer', {}).get('warmup_fraction', 0.05),
        'max_steps': config.get('training_budgets', {}).get('classification_core', {}).get('epochs', 2) * 100,
        'mixed_precision': config.get('hardware', {}).get('mixed_precision', True),
        'betas': config.get('optimizer', {}).get('betas', (0.9, 0.999)),
        'eps': config.get('optimizer', {}).get('eps', 1e-8),
    }

    trainer = ClassificationTrainer(model, trainer_config, device, output_dir, run_id)

    budget_config = config.get('training_budgets', {}).get('classification_core', {})
    epochs = budget_config.get('epochs', 2)
    checkpoint_epochs = budget_config.get('checkpoint_epochs', [2])

    # Get dataset config from dataset_configs
    dataset_name = prepared_data.metadata.get('dataset', 'agnews')
    dataset_config = config.get('dataset_configs', {}).get(dataset_name, {})
    batch_size = dataset_config.get('batch_size', 64)
    eval_batch_size = dataset_config.get('eval_batch_size', 64)

    # Extract seed from run_id
    seed = int(run_id.split('_seed_')[-1].split('_')[0]) if '_seed_' in run_id else 42

    history = trainer.train(
        prepared_data.X_train, prepared_data.y_train,
        prepared_data.X_val, prepared_data.y_val,
        epochs, batch_size, eval_batch_size,
        checkpoint_epochs, seed
    )

    # Final test evaluation
    test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, eval_batch_size)

    return {
        'training_history': trainer.training_history,
        'validation_history': trainer.validation_history,
        'test_metrics': test_metrics,
        'checkpoint_history': trainer.checkpoint_history,
    }


def train_lm_model(model: nn.Module, prepared_data: PreparedData,
                   config: dict, device: torch.device,
                   output_dir: str, run_id: str) -> Dict:
    """
    Convenience function to train a language model.

    Args:
        model: Model to train
        prepared_data: PreparedData object with train/val/test splits
        config: Training configuration
        device: Device to train on
        output_dir: Output directory for checkpoints
        run_id: Unique run identifier

    Returns:
        Training history dictionary
    """
    trainer_config = {
        'learning_rate': config.get('optimizer', {}).get('lr', 3e-4),
        'weight_decay': config.get('optimizer', {}).get('weight_decay', 0.01),
        'warmup_frac': config.get('optimizer', {}).get('warmup_fraction', 0.05),
        'max_steps': config.get('training_budgets', {}).get('wikitext2_lm', {}).get('max_steps', 25000),
        'mixed_precision': config.get('hardware', {}).get('mixed_precision', True),
        'betas': config.get('optimizer', {}).get('betas', (0.9, 0.999)),
        'eps': config.get('optimizer', {}).get('eps', 1e-8),
    }

    trainer = LMTrainer(model, trainer_config, device, output_dir, run_id)

    budget_config = config.get('training_budgets', {}).get('wikitext2_lm', {})
    max_steps = budget_config.get('max_steps', 25000)
    eval_every = budget_config.get('eval_every', 5000)
    checkpoint_steps = budget_config.get('checkpoint_steps', [5000, 10000, 15000, 20000, 25000])

    dataset_config = config.get('dataset_configs', {}).get('wikitext2', {})
    batch_size = dataset_config.get('batch_size', 16)
    eval_batch_size = dataset_config.get('eval_batch_size', 32)

    # Extract seed from run_id
    seed = int(run_id.split('_seed_')[-1].split('_')[0]) if '_seed_' in run_id else 42

    history = trainer.train(
        prepared_data.X_train, prepared_data.y_train,
        prepared_data.X_val, prepared_data.y_val,
        max_steps, batch_size, eval_batch_size,
        eval_every, checkpoint_steps, seed
    )

    # Final test evaluation
    test_metrics = trainer.evaluate(prepared_data.X_test, prepared_data.y_test, eval_batch_size)

    return {
        'training_history': trainer.training_history,
        'validation_history': trainer.validation_history,
        'test_metrics': test_metrics,
        'checkpoint_history': trainer.checkpoint_history,
    }