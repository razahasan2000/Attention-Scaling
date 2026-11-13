"""
Common infrastructure for the Attention-Scaling replica experiments (Stage 2).

Mirrors the architecture of razahasan2000/Attention-Scaling (src/attention/scaling.py,
src/models/transformer_encoder.py):
  - pre-LN blocks, learned positional embeddings, masked mean pooling,
  - tied LM head, logit clamp (-50, 50) BEFORE masking, exact-zero masked weights,
  - causal (lower-triangular) + key-padding mask for language modelling,
  - bidirectional attention with key-padding mask for classification,
  - variants: standard (1/sqrt(d_k)), dk (1/d_k), temp (1/(tau*sqrt(d_k)), tau=2),
    plus multiscale_v1 (alternating per-head scales) and dk_lrcomp (1/d_k with
    LR compensation) for the E1-E4 ablation arms.

Replica-scale protocol (documented in the manuscript):
  - CPU-only execution, 2 cores; torch threads pinned to 1 per worker.
  - Matched seeds: for each seed, all variants are initialised from a
    bit-identical state dict and consume identical data batches.
"""

import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1
RESULTS_DIR = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results"))
PROC_DIR = os.environ.get("AS_PROC", os.path.join(os.getcwd(), "data", "proc"))
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")

VARIANTS = ("standard", "dk", "temp")
TAU = 2.0


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9']+|[.,!?;:]")


def tokenize_words(text):
    return _WORD_RE.findall(text.lower())


def tokenize_wikitext_line(line):
    # WikiText-2 is pre-tokenised; whitespace split is exact.
    return line.lower().split()


def build_vocab(token_lists, max_vocab, min_freq):
    from collections import Counter
    cnt = Counter()
    for toks in token_lists:
        cnt.update(toks)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in cnt.most_common():
        if c < min_freq or len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode(toks, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in toks[:max_len]]
    ids += [PAD] * (max_len - len(ids))          # right-pad to fixed length
    return ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def scale_factor(variant, d_k, tau=TAU, head_idx=None):
    if variant == "standard":
        return math.sqrt(d_k)
    if variant == "dk" or variant == "dk_lrcomp":
        # dk_lrcomp = same 1/d_k divisor, LR compensated in the runner (x sqrt(d_k))
        return float(d_k)
    if variant == "temp":
        return tau * math.sqrt(d_k)
    if variant == "multiscale_v1":
        # Alternating heads: even -> standard, odd -> 1/d_k (repo E1/E3 arm)
        return math.sqrt(d_k) if (head_idx % 2 == 0) else float(d_k)
    raise ValueError(variant)


class Attention(nn.Module):
    """Scaled dot-product attention with configurable divisor s (clamp-before-mask).

    Stage-3 operating-point knobs:
      (logit gain is applied to q/k in Block, BEFORE QK-Norm, so that the
       normaliser can absorb it — the faithful frontier-stabiliser semantics);
      softcap c    : logits = c * tanh(logits / c) after the divisor
                     (Gemma-2-style attention logit soft-capping).
    """

    def __init__(self, variant, d_k, tau=TAU, softcap=0.0):
        super().__init__()
        self.variant, self.d_k, self.tau = variant, d_k, tau
        self.softcap = float(softcap)

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: (B, H, L, d_k); mask: (B, 1, L, L) or (B, 1, 1, L), 1 = attend
        d_k = Q.size(-1)
        logits = torch.matmul(Q, K.transpose(-2, -1))
        if self.variant == "multiscale_v1":
            for h in range(Q.size(1)):
                logits[:, h] = logits[:, h] / scale_factor(self.variant, d_k, self.tau, h)
        else:
            logits = logits / scale_factor(self.variant, d_k, self.tau)
        if self.softcap > 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        logits = torch.clamp(logits, min=-50.0, max=50.0)   # clamp BEFORE mask (repo v2)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        return torch.matmul(w, V)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, variant,
                 qk_norm=False, logit_gain=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(variant, d_model // n_heads)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.n_heads, self.d_model = n_heads, d_model
        # Logit gain: q and k are each scaled by sqrt(g) AFTER Wq/Wk and BEFORE
        # QK-Norm. The dot product then scales by g (matching a large-logit
        # operating point), while QK-Norm — being scale-invariant — absorbs
        # the injection entirely (the frontier-stabiliser semantics).
        self.logit_gain = float(logit_gain)
        self._gsqrt = math.sqrt(self.logit_gain)
        # QK-Norm (Henry et al., 2020; ViT-22B-style per-head LayerNorm): normalise
        # q and k before the dot product. Default affine init (1, 0) is
        # deterministic, so matched initialisation with non-QK-Norm variants is
        # preserved bit-for-bit.
        self.qk_norm = qk_norm
        if qk_norm:
            self.qnorm = nn.LayerNorm(d_model // n_heads)
            self.knorm = nn.LayerNorm(d_model // n_heads)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, -1).transpose(1, 2)

    def qk_from_h(self, h):
        """q, k from pre-normalised input h (with gain and optional QK-Norm)."""
        q, k = self._split(self.Wq(h)), self._split(self.Wk(h))
        if self._gsqrt != 1.0:
            q, k = q * self._gsqrt, k * self._gsqrt
        if self.qk_norm:
            q, k = self.qnorm(q), self.knorm(k)
        return q, k

    def qk(self, x):
        """Compute (optionally QK-Normalised) q, k from the block input."""
        return self.qk_from_h(self.norm1(x))

    def forward(self, x, mask):
        B, L, _ = x.shape
        h = self.norm1(x)
        q, k = self.qk_from_h(h)
        v = self._split(self.Wv(h))
        a = self.attn(q, k, v, mask)
        a = a.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = x + self.drop(self.Wo(a))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 dropout=0.1, max_len=64, variant="standard",
                 task="classification", n_classes=4,
                 logit_gain=1.0, softcap=0.0, qk_norm=False):
        super().__init__()
        self.task = task
        self.variant = variant
        self.causal_override = None   # None -> task default; False -> leaky (v1 bug)
        self.tok = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, variant,
                   qk_norm=qk_norm, logit_gain=logit_gain) for _ in range(n_layers)])
        self.final = nn.LayerNorm(d_model)
        if task == "classification":
            self.head = nn.Linear(d_model, n_classes)
        else:
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # tied
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def build_mask(self, ids, causal):
        key = (ids != PAD)                      # (B, L)
        L = ids.size(1)
        if causal:
            tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=ids.device))
            m = key[:, None, :] & tril[None, :, :]     # (B, L, L)
            return m[:, None, :, :]                    # (B, 1, L, L)
        return key[:, None, None, :]                   # (B, 1, 1, L)

    def forward(self, ids):
        B, L = ids.shape
        causal = (self.task == "lm") if self.causal_override is None else self.causal_override
        mask = self.build_mask(ids, causal)
        x = self.drop(self.tok(ids) + self.pos[:, :L, :])
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final(x)
        if self.task == "classification":
            key = (ids != PAD).float()
            pooled = (x * key.unsqueeze(-1)).sum(1) / key.sum(1, keepdim=True).clamp(min=1.0)
            return self.head(pooled)
        return self.head(x)                              # (B, L, vocab)


# ---------------------------------------------------------------------------
# Matched-seed construction
# ---------------------------------------------------------------------------
def build_matched(seed, variants, **kw):
    """All variants start from a bit-identical state dict (torch.manual_seed(seed)).

    Extra Stage-3 modules (QK-Norm affine params) are absent from the reference
    state dict and keep their deterministic default init; all shared parameters
    are loaded bit-identically, so the matched-seed guarantee is preserved.
    """
    torch.manual_seed(seed)
    ref = Transformer(variant="standard", **kw)
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    out = {}
    for v in variants:
        m = Transformer(variant=v, **kw)
        m.load_state_dict(state, strict=False)
        out[v] = m
    del ref
    return out


# ---------------------------------------------------------------------------
# Batch iteration (identical across variants for a given seed)
# ---------------------------------------------------------------------------
def epoch_order(n, seed, epoch):
    return np.random.RandomState(seed * 7919 + epoch).permutation(n)


def cls_batches(X, y, bs, seed, epoch):
    order = epoch_order(len(y), seed, epoch)
    for i in range(0, len(y), bs):
        idx = order[i:i + bs]
        yield torch.from_numpy(X[idx]).long(), torch.from_numpy(y[idx]).long()


def lm_batches(ids, bs, seq, seed, epoch):
    n_chunks = len(ids) // seq
    data = ids[: n_chunks * seq].reshape(n_chunks, seq)
    order = epoch_order(n_chunks, seed, epoch)
    for i in range(0, n_chunks - (n_chunks % bs), bs):
        idx = order[i:i + bs]
        yield torch.from_numpy(data[idx]).long()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def set_threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised (multiprocessing workers)


def lr_at(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_cls(model, X, y, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        ids = torch.from_numpy(X[i:i + bs]).long()
        pred = model(ids).argmax(-1)
        correct += (pred == torch.from_numpy(y[i:i + bs]).long()).sum().item()
    model.train()
    return correct / len(y)


def train_cls(model, Xtr, ytr, Xva, yva, seed, epochs=2, bs=32, lr=3e-4,
              Xte=None, yte=None, early_stop=None, record_curve=False,
              probe_each_epoch=None):
    """early_stop: patience (int) or None. probe_each_epoch: fn(model)->dict.
    Returns dict."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.999), eps=1e-8)
    steps_total = epochs * math.ceil(len(ytr) / bs)
    gstep, best, best_state, best_ep, bad = 0, float("inf"), None, 0, 0
    curve = []
    for ep in range(epochs):
        for ids, lab in cls_batches(Xtr, ytr, bs, seed, ep):
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep, steps_total, lr)
            opt.zero_grad()
            loss = F.cross_entropy(model(ids), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for i in range(0, len(yva), 256):
                ids = torch.from_numpy(Xva[i:i + 256]).long()
                lab = torch.from_numpy(yva[i:i + 256]).long()
                vloss += F.cross_entropy(model(ids), lab, reduction="sum").item()
            vloss /= len(yva)
        ep_probe = probe_each_epoch(model) if probe_each_epoch is not None else None
        model.train()
        vacc = eval_cls(model, Xva, yva)
        tacc = eval_cls(model, Xte, yte) if Xte is not None else None
        if record_curve:
            entry = {"epoch": ep + 1, "val_loss": vloss, "val_acc": vacc,
                     "test_acc": tacc}
            if ep_probe is not None:
                entry["probe"] = ep_probe
            curve.append(entry)
        if early_stop is None:
            best_ep, best_vacc = ep + 1, vacc
            continue
        if vloss < best - 1e-4:
            best, best_ep, bad = vloss, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop:
                break
    if early_stop is not None and best_state is not None:
        model.load_state_dict(best_state)
    res = {"epochs_run": (best_ep if early_stop is not None else epochs),
           "val_acc": eval_cls(model, Xva, yva)}
    if Xte is not None:
        res["test_acc"] = eval_cls(model, Xte, yte)
    if record_curve:
        res["curve"] = curve
    return res


@torch.no_grad()
def eval_lm(model, ids, bs=64, seq=64):
    model.eval()
    n = len(ids) // seq
    data = ids[: n * seq].reshape(n, seq)
    total_nll, count = 0.0, 0
    for i in range(0, n, bs):
        chunk = torch.from_numpy(data[i:i + bs]).long()
        logits = model(chunk)[:, :-1, :]
        tgt = chunk[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                              ignore_index=PAD, reduction="sum")
        total_nll += nll.item()
        count += int((tgt != PAD).sum())
    model.train()
    return math.exp(total_nll / count), total_nll / count


def train_lm(model, ids_tr, ids_va, seed, epochs=3, bs=32, seq=64, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.999), eps=1e-8)
    n_steps = epochs * (len(ids_tr) // (seq * bs))
    gstep = 0
    for ep in range(epochs):
        for chunk in lm_batches(ids_tr, bs, seq, seed, ep):
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep, n_steps, lr)
            opt.zero_grad()
            logits = model(chunk)[:, :-1, :]
            tgt = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
    return eval_lm(model, ids_va, seq=seq)


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------
def append_result(rec):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_results():
    done = set()
    if not os.path.exists(RESULTS_FILE):
        return done
    with open(RESULTS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                # key on (phase, dataset, variant, dk, seed, spec); the arm field
                # is redundant (for abl jobs the arm IS the variant).
                done.add((r["phase"], r.get("dataset"), r.get("variant"),
                          r.get("dk"), r.get("seed"), r.get("spec", "")))
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Stage-3 operating-point probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe(model, ids):
    """Operating-point diagnostics on the model's own q/k logits.

    Per layer: sigma_raw (std of q.k before any gain/divisor), the realized
    attention entropy / max weight under the model's own configuration, and
    the KL divergence between attention maps under the standard divisor and
    each alternative divisor applied to the SAME logits (the pure divisor
    contrast that the power law predicts).
    """
    was_training = model.training
    model.eval()
    L = ids.size(1)
    causal = (model.task == "lm") if model.causal_override is None \
        else bool(model.causal_override)
    mask = model.build_mask(ids, causal)
    d_k = model.layers[0].attn.d_k
    div = {"standard": math.sqrt(d_k), "dk": float(d_k),
           "temp": TAU * math.sqrt(d_k)}
    g = model.layers[0].logit_gain
    cap = model.layers[0].attn.softcap
    own = model.variant if model.variant in div else None
    out = {"gain": g, "softcap": cap, "divisor": own, "d_k": d_k, "layers": []}
    key_q = (ids != PAD)                                   # (B, L) real queries
    attn_mask_sq = mask[:, 0]                              # (B, L, L) or (B, 1, L)
    x = model.drop(model.tok(ids) + model.pos[:, :L, :])

    def row_stats(w):
        # w: (B, H, Lq, Lk); returns entropy and max weight per (B, H, Lq)
        ent = -(w.clamp(min=1e-12).log() * w).sum(-1)
        return ent, w.max(-1).values

    def agg(per_row):
        # mean over heads, then weighted mean over query rows (real rows only)
        v = per_row.mean(1)                                # (B, L)
        if attn_mask_sq.size(1) == 1:                      # cls mask (B,1,L)
            roww = key_q.float()
        else:                                              # LM causal (B,L,L)
            roww = torch.ones_like(v)
        return float((v * roww).sum() / roww.sum().clamp(min=1.0))

    def sm(c):
        z = raw * (1.0 / c)
        z = z.masked_fill(mask == 0, float("-inf"))
        return torch.nan_to_num(F.softmax(z, -1))

    for li, blk in enumerate(model.layers):
        q, k = blk.qk(x)
        raw = torch.matmul(q, k.transpose(-2, -1))         # (B, H, L, L)
        # raw already includes the operating point (gain, QK-Norm effects);
        # sigma_raw is the pre-divisor logit std at this operating point
        entry = {"layer": li + 1, "sigma_raw": round(float(raw.std()), 4)}
        # realized attention under the model's own divisor/cap
        if own is not None:
            z = raw * (1.0 / div[own])
            if cap > 0.0:
                z = cap * torch.tanh(z / cap)
            z = torch.clamp(z, -50.0, 50.0).masked_fill(mask == 0, float("-inf"))
            w_own = torch.nan_to_num(F.softmax(z, -1))
            ent, mx = row_stats(w_own)
            entry["entropy_own"] = round(agg(ent), 4)
            entry["maxw_own"] = round(agg(mx), 4)
            # pure divisor contrast on the same logits (power-law quantity)
            w_sd, w_dk, w_tp = sm(div["standard"]), sm(div["dk"]), sm(div["temp"])
            eps = 1e-12
            kl_sd_dk = (w_sd * ((w_sd + eps).log() - (w_dk + eps).log())).sum(-1)
            kl_sd_tp = (w_sd * ((w_sd + eps).log() - (w_tp + eps).log())).sum(-1)
            entry["kl_std_dk"] = round(agg(kl_sd_dk), 5)
            entry["kl_std_temp"] = round(agg(kl_sd_tp), 5)
        out["layers"].append(entry)
        x = blk(x, mask)
    if was_training:
        model.train()
    return out


def probe_summary(pd_):
    """Flatten a probe() output into a compact record (means over layers)."""
    lay = pd_["layers"]
    n = len(lay)
    s = {"gain": pd_["gain"], "softcap": pd_["softcap"],
         "divisor": pd_["divisor"],
         "sigma_raw": round(sum(l["sigma_raw"] for l in lay) / n, 4)}
    for k in ("entropy_own", "maxw_own", "kl_std_dk", "kl_std_temp"):
        if k in lay[0]:
            s[k] = round(sum(l[k] for l in lay) / n, 5)
    # realized post-divisor logit std under the model's own divisor
    if pd_["divisor"]:
        divmap = {"standard": math.sqrt(pd_["d_k"]), "dk": float(pd_["d_k"]),
                  "temp": TAU * math.sqrt(pd_["d_k"])}
        s["sigma_post_own"] = round(s["sigma_raw"] / divmap[pd_["divisor"]], 4)
    return s
