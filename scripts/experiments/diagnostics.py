"""Attention diagnostics at replica scale (for the Discussion section).

For seed 1001, AG News replica config, 2-epoch training:
  - attention entropy, max attention weight (first layer, val batch)
  - Q/K/V gradient norms on a fresh batch
  - fraction of clamped logits
  - ||W_q||, ||W_k|| growth from init (compensation evidence)

For LM seed 1001 (corrected causal mask): same diagnostics + logit scale.
"""

import json
import math
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def first_layer_weights(model, ids):
    blk = model.layers[0]
    mask = model.build_mask(ids, causal=False)
    h = blk.norm1(model.drop(model.tok(ids) + model.pos[:, : ids.size(1), :]))
    q, k, v = blk._split(blk.Wq(h)), blk._split(blk.Wk(h)), blk._split(blk.Wv(h))
    logits = torch.matmul(q, k.transpose(-2, -1))
    scale = C.scale_factor(model.variant, q.size(-1))
    if model.variant == "multiscale_v1":
        for hh in range(q.size(1)):
            logits[:, hh] = logits[:, hh] / C.scale_factor(model.variant, q.size(-1), 2.0, hh)
        scale = None
    elif scale:
        logits = logits / scale
    n_clamp = int((torch.clamp(logits, -50, 50) != logits).sum())
    logits = torch.clamp(logits, -50, 50).masked_fill(mask == 0, float("-inf"))
    w = F.softmax(logits, -1)
    key = (ids != 0).float()
    denom = key.sum(1, keepdim=True)
    ent = -(torch.nan_to_num(w, 0.0) * torch.nan_to_num(w, 0.0).clamp(min=1e-12).log()).sum(-1)
    ent = (ent * key.unsqueeze(1)).sum(2) / denom.clamp(min=1).unsqueeze(1)
    mx = (w * key[:, None, None, :]).flatten(2).max(-1).values
    return float(ent.mean()), float(mx.mean()), n_clamp / max(1, logits.numel()), w


def diag_cls(seed=1001):
    d = np.load(f"{C.PROC_DIR}/agnews.npz")
    out = {}
    torch.manual_seed(seed)
    ref = C.Transformer(vocab=12000, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                        dropout=0.1, max_len=64, variant="standard", task="classification")
    init_norms = {k: float(v.norm()) for k, v in ref.state_dict().items()
                  if k.startswith(("layers.0.Wq", "layers.0.Wk"))}
    models = C.build_matched(seed, ["standard", "dk", "temp"],
                             vocab=12000, d_model=128, n_heads=4, n_layers=2,
                             d_ff=256, dropout=0.1, max_len=64,
                             task="classification", n_classes=4)
    ids = torch.from_numpy(d["Xva"][:128]).long()
    yva = torch.from_numpy(d["yva"][:128]).long()
    for v, m in models.items():
        C.train_cls(m, d["Xtr"], d["ytr"], d["Xva"], d["yva"], seed,
                    epochs=2, bs=32, lr=3e-4)
        ent, mx, clamp_frac, w = first_layer_weights(m, ids)
        # gradient norms on a fresh batch
        m.train()
        m.zero_grad()
        loss = F.cross_entropy(m(ids), yva)
        loss.backward()
        gq = float(m.layers[0].Wq.weight.grad.norm())
        gk = float(m.layers[0].Wk.weight.grad.norm())
        gv = float(m.layers[0].Wv.weight.grad.norm())
        wq = float(m.layers[0].Wq.weight.norm())
        wk = float(m.layers[0].Wk.weight.norm())
        out[v] = {"entropy": round(ent, 3), "max_weight": round(mx, 3),
                  "clamp_frac": clamp_frac, "grad_QK": round(gq + gk, 4),
                  "grad_V": round(gv, 4),
                  "Wq_norm_ratio": round(wq / init_norms["layers.0.Wq.weight"], 2),
                  "Wk_norm_ratio": round(wk / init_norms["layers.0.Wk.weight"], 2)}
    return out


def diag_lm(seed=1001):
    tr = np.load(f"{C.PROC_DIR}/wikitext2_train.npy")
    va = np.load(f"{C.PROC_DIR}/wikitext2_valid.npy")
    out = {}
    models = C.build_matched(seed, ["standard", "dk", "temp"],
                             vocab=8000, d_model=96, n_heads=3, n_layers=2,
                             d_ff=192, dropout=0.1, max_len=64, task="lm")
    ids = torch.from_numpy(va[:32 * 64].reshape(32, 64)).long()
    for v, m in models.items():
        C.train_lm(m, tr, va, seed, epochs=3, bs=32, seq=64, lr=3e-4)
        ent, mx, clamp_frac, _ = first_layer_weights(m, ids)
        ppl, _ = C.eval_lm(m, va, seq=64)
        # effective logit scale on real data
        blk = m.layers[0]
        h = blk.norm1(m.tok(ids) + m.pos[:, :64, :])
        q, k = blk._split(blk.Wq(h)), blk._split(blk.Wk(h))
        raw = torch.matmul(q, k.transpose(-2, -1))
        out[v] = {"entropy": round(ent, 3), "max_weight": round(mx, 3),
                  "clamp_frac": clamp_frac,
                  "raw_logit_std": round(float(raw.std()), 3),
                  "ppl": round(ppl, 1)}
    return out


if __name__ == "__main__":
    C.set_threads()
    res = {"cls_agnews_seed1001": diag_cls(), "lm_seed1001": diag_lm()}
    with open(f"{C.RESULTS_DIR}/diagnostics.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
