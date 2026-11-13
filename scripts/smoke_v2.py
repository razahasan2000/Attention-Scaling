"""
End-to-end smoke test for the v2 pipeline on SYNTHETIC data (no downloads).

Validates the full job path (create_model -> trainer -> record schema) for:
  1. a core-protocol classification job,
  2. a convergence-protocol job (early stopping fields present),
  3. a d_k=128 head-dimension job (n_heads=1),
  4. a causal LM job: held-out perplexity must be SANE (near vocab size,
     NOT ~1.09 as under the v1 leakage bug), and the bidirectional control
     must collapse to far lower perplexity (proving the leak is real and
     that causal masking is what removes it).

Run:  python scripts/smoke_v2.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from data.prepare import PreparedData
from training.trainer import train_classification_model, train_lm_model
from models.transformer_encoder import create_model
from run_all_v2 import run_job, DataPool

DEVICE = torch.device("cpu")
VOCAB = 500
SEQ = 64
N_TRAIN, N_VAL, N_TEST = 1200, 200, 200


def synthetic_classification(seed: int) -> PreparedData:
    rng = np.random.RandomState(seed)
    def arr(n):
        # padding id 0 with probability 0.15 at tail positions
        x = rng.randint(1, VOCAB, size=(n, SEQ))
        lens = rng.randint(SEQ // 2, SEQ + 1, size=n)
        for i in range(n):
            x[i, lens[i]:] = 0
        return x
    def labels(n):
        return rng.randint(0, 4, size=n)
    return PreparedData(
        X_train=arr(N_TRAIN), y_train=labels(N_TRAIN),
        X_val=arr(N_VAL), y_val=labels(N_VAL),
        X_test=arr(N_TEST), y_test=labels(N_TEST),
        vocab={"<pad>": 0, "UNK": 1}, vocab_size=VOCAB, max_len=SEQ,
        metadata={"dataset": "synthetic", "task": "classification"},
    )


def synthetic_lm(seed: int) -> PreparedData:
    rng = np.random.RandomState(seed)
    def arr(n):
        return rng.randint(1, VOCAB, size=(n, SEQ))  # no padding, random text
    # Real pipeline: labels mirror inputs (model shifts internally).
    Xtr, Xva, Xte = arr(600), arr(60), arr(60)
    return PreparedData(
        X_train=Xtr, y_train=Xtr.copy(), X_val=Xva, y_val=Xva.copy(),
        X_test=Xte, y_test=Xte.copy(),
        vocab={"<pad>": 0, "UNK": 1, "<eos>": 2}, vocab_size=VOCAB,
        max_len=SEQ, metadata={"dataset": "synthetic",
                               "task": "language_modelling"},
    )


def base_config():
    return {
        "model": {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 256,
                  "dropout": 0.1, "temperature": 2.0,
                  "multiscale_v1": {"per_head_scales": [5.656854249492381,
                                                        5.656854249492381,
                                                        32.0, 32.0]}},
        "dataset_configs": {
            "synthetic": {"num_classes": 4, "batch_size": 64,
                          "eval_batch_size": 64},
            "wikitext2": {"batch_size": 16, "eval_batch_size": 32},
        },
        "training_budgets": {
            "classification_core": {"epochs": 2, "checkpoint_epochs": [2]},
            "convergence": {"epochs": 3, "checkpoint_epochs": [3],
                            "early_stopping_patience": 1,
                            "early_stopping_min_delta": 0.0},
            "wikitext2_lm": {"max_steps": 30, "eval_every": 15,
                             "checkpoint_steps": [30]},
        },
        "optimizer": {"lr": 3e-4, "weight_decay": 0.01,
                      "warmup_fraction": 0.05, "betas": [0.9, 0.999],
                      "eps": 1e-8},
        "hardware": {"mixed_precision": False},
        "data": {},
    }


def main():
    tmp = tempfile.mkdtemp(prefix="smoke_v2_")
    config = base_config()
    pool = DataPool.__new__(DataPool)  # bypass: inject synthetic data
    pool.config = config
    pool.output_dir = tmp
    pool._pool = {}
    pool.get = lambda dataset, max_len=None: (
        synthetic_lm(0) if dataset in ("wikitext2", "lm")
        else synthetic_classification(0))

    # ---- 1. core classification job --------------------------------------
    job = {"run_id": "synth_standard_seed_1001_25epoch", "kind": "classification",
           "grid": "core", "dataset": "synthetic", "variant": "standard",
           "seed": 1001, "protocol": "core", "overrides": {}}
    rec = run_job(job, config, pool, DEVICE, tmp)
    assert rec["test_accuracy"] is not None and 0.0 <= rec["test_accuracy"] <= 1.0
    assert rec["d_k"] == 32 and rec["n_heads"] == 4
    print(f"1. core classification OK (test_acc={rec['test_accuracy']:.3f}, "
          f"2 epochs recorded={len(rec['training_history'])})")

    # ---- 2. convergence job ------------------------------------------------
    job = {"run_id": "synth_temp_seed_1001_conv", "kind": "classification",
           "grid": "converged", "dataset": "synthetic", "variant": "temp",
           "seed": 1001, "protocol": "conv", "overrides": {}}
    rec = run_job(job, config, pool, DEVICE, tmp)
    info = rec.get("early_stop_info", {})
    assert info.get("enabled") is True, "convergence protocol did not activate"
    assert info.get("epochs_run", 99) <= 3
    print(f"2. convergence job OK (early_stopped={info.get('early_stopped')}, "
          f"epochs_run={info.get('epochs_run')}, "
          f"best_epoch={info.get('best_epoch')})")

    # ---- 3. head-dimension job (d_k = 128 -> 1 head) -----------------------
    job = {"run_id": "synth_dk_seed_1001_25epoch", "kind": "classification",
           "grid": "dk128", "dataset": "synthetic", "variant": "dk",
           "seed": 1001, "protocol": "core", "overrides": {"n_heads": 1}}
    rec = run_job(job, config, pool, DEVICE, tmp)
    assert rec["d_k"] == 128 and rec["n_heads"] == 1
    print(f"3. d_k=128 job OK (test_acc={rec['test_accuracy']:.3f})")

    # ---- 4. causal LM job + bidirectional control --------------------------
    job = {"run_id": "wikitext2_standard_seed_1001_lm25k_causal",
           "kind": "lm", "grid": "lm_causal", "dataset": "wikitext2",
           "variant": "standard", "seed": 1001, "protocol": "lm25k",
           "overrides": {}}
    rec = run_job(job, config, pool, DEVICE, tmp)
    ppl_causal = rec["test_perplexity"]
    assert ppl_causal is not None and ppl_causal > 50, (
        f"causal LM perplexity {ppl_causal:.2f} suspiciously low - leakage?")
    print(f"4. causal LM OK (test_ppl={ppl_causal:.1f}; "
          f"v1-leakage reference was ~1.09)")

    # Control: same path WITHOUT causal masking (the v1 bug) must collapse.
    # Both control arms get an identical elevated budget so the copy rule
    # has time to emerge; only the mask differs.
    ctl_cfg = json.loads(json.dumps(config))
    ctl_cfg["training_budgets"]["wikitext2_lm"] = {
        "max_steps": 500, "eval_every": 500, "checkpoint_steps": [500]}
    ctl_cfg["optimizer"] = dict(config["optimizer"], lr=2e-3)
    ppls = {}
    for tag, causal in [("bidi (v1 behaviour)", False), ("causal (v2)", True)]:
        torch.manual_seed(1001)
        model = create_model(variant="standard", d_model=128, n_heads=4,
                             n_layers=2, d_ff=256, dropout=0.1,
                             max_seq_len=SEQ, vocab_size=VOCAB,
                             task_type="language_modelling")
        model.set_causal(causal)
        res = train_lm_model(model=model, prepared_data=synthetic_lm(0),
                             config=ctl_cfg, device=DEVICE,
                             output_dir=os.path.join(tmp, "ctl_" + tag[:4]),
                             run_id="ctl")
        ppls[tag] = res["test_metrics"]["val_perplexity"]
    ppl_bidi = ppls["bidi (v1 behaviour)"]
    ppl_causal_ctl = ppls["causal (v2)"]
    assert ppl_bidi < ppl_causal_ctl / 3, (
        f"bidirectional control ppl {ppl_bidi:.2f} did not collapse "
        f"relative to causal {ppl_causal_ctl:.2f}")
    print(f"   control: bidi(v1) ppl={ppl_bidi:.2f} vs causal(v2) "
          f"ppl={ppl_causal_ctl:.1f}  <-- leak confirmed and fixed")

    print("\nSMOKE TEST PASSED: all four v2 paths work end to end.")


if __name__ == "__main__":
    main()
