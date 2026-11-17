"""Frontier-regime probe: inference-time scaling-divisor swap on pretrained GPT-2.

Answers the "frontier scale" critique with the largest decoder runnable on the
2-core CPU sandbox. GPT-2 was TRAINED with the 2017 default divisor
1/sqrt(d_k); at inference we swap only the divisor (no fine-tuning) and measure
WikiText-2 perplexity + per-layer operating-point diagnostics.

  variants:
    standard : scaling = 1/sqrt(d_k)   (GPT-2's training-time divisor)
    dk       : scaling = 1/d_k         (the alternative divisor)
    temp     : scaling = 1/(2 sqrt(d_k)) (temperature tau = 2)

Because q,k are frozen, raw logits q.k are identical across variants; only the
divisor changes. Per layer we record sigma_raw (pre-divisor logit std), the
realised attention entropy under each divisor, and the KL divergences
KL(P_std || P_dk) / KL(P_std || P_temp) -- the exact quantities the Stage-3
power law predicts, now measured in a pretrained network whose logit variance
is orders of magnitude larger than the replica models.

Output: data/results/frontier_gpt2.json
Usage:  python3 frontier_gpt2.py [--models gpt2,gpt2-medium] [--nseq 160] [--seqlen 128]
"""
import argparse
import gc
import json
import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

WT2_VALID = os.environ.get("AS_RAW", os.path.join(os.getcwd(), "data", "raw")) + "/wt2_valid.parquet"
RESULTS = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results"))
TAU = 2.0


def scalings(d_head):
    return {"standard": d_head ** -0.5,
            "dk": float(d_head) ** -1.0,
            "temp": (TAU * math.sqrt(d_head)) ** -1.0}


def build_tokens(nseq, seqlen):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    df = pd.read_parquet(WT2_VALID)
    stream = tok("\n\n".join(df["text"].tolist()), return_tensors="np")["input_ids"][0]
    n = min(nseq, len(stream) // seqlen - 1)
    seqs = np.array([stream[i * seqlen:(i + 1) * seqlen] for i in range(n)])
    return torch.from_numpy(seqs).long(), tok


@torch.no_grad()
def ppl_pass(model, seqs, bs=8):
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, len(seqs), bs):
        ids = seqs[i:i + bs]
        logits = model(ids).logits[:, :-1, :]
        tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              tgt.reshape(-1), reduction="sum")
        tot_nll += float(nll)
        tot_tok += tgt.numel()
    return math.exp(tot_nll / tot_tok), tot_nll / tot_tok


@torch.no_grad()
def stats_pass(model, seqs, d_head, bs=4, max_batches=2):
    """Per-layer raw-logit operating point on a small sample."""
    sc = scalings(d_head)
    n_layers = model.config.n_layer
    acc = [{"raw_sq": 0.0, "raw_n": 0, "kl_sd_dk": 0.0, "kl_sd_tp": 0.0,
            "H": {k: 0.0 for k in sc}, "nb": 0} for _ in range(n_layers)]

    hooks, handles = [], []

    def make_hook(li):
        def hook(mod, args, out):
            hooks.append((li, out))
        return hook

    # attach hooks to every c_attn (captures q,k for this layer)
    for li, block in enumerate(model.transformer.h):
        attn = next(m for n, m in block.named_children()
                    if hasattr(m, "c_attn"))
        handles.append(attn.c_attn.register_forward_hook(make_hook(li)))

    nb = 0
    for i in range(0, min(len(seqs), bs * max_batches), bs):
        ids = seqs[i:i + bs]
        hooks.clear()
        model(ids)
        L = ids.size(1)
        E = model.config.n_embd
        H = model.config.n_head
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        keep = causal.view(1, 1, L, L)
        for li, qkv in hooks:
            q, k = qkv[..., :E], qkv[..., E:2 * E]
            q = q.view(*q.shape[:-1], H, -1).transpose(1, 2)   # (B,H,L,d)
            k = k.view(*k.shape[:-1], H, -1).transpose(1, 2)
            raw = torch.matmul(q, k.transpose(-2, -1))          # (B,H,L,L)
            a = acc[li]
            # sigma over the VALID (causally allowed) entries only
            valid = raw.masked_select(keep)
            a["raw_sq"] += float((valid ** 2).sum())
            a["raw_n"] += valid.numel()
            # per-row masked softmax (each row attends only j <= i)
            for name, s in sc.items():
                z = raw * s
                z = z.masked_fill(~keep, float("-inf"))
                logw = z - torch.logsumexp(z, -1, keepdim=True)
                logw = logw.masked_fill(~keep, 0.0)      # avoid 0 * -inf
                ent = -(logw.exp() * logw).sum(-1)                     # (B,H,L)
                a["H"][name] += float(ent.sum())
            b_std, b_dk, b_tp = sc["standard"], sc["dk"], sc["temp"]
            z_std = raw * b_std
            z_std = z_std.masked_fill(~keep, float("-inf"))
            lq = z_std - torch.logsumexp(z_std, -1, keepdim=True)
            for name, b2 in (("kl_sd_dk", b_dk), ("kl_sd_tp", b_tp)):
                z2 = raw * b2
                z2 = z2.masked_fill(~keep, float("-inf"))
                lr = z2 - torch.logsumexp(z2, -1, keepdim=True)
                d = (lq - lr).masked_fill(~keep, 0.0)    # avoid -inf - -inf
                kl = (lq.exp().masked_fill(~keep, 0.0) * d).sum(-1)    # (B,H,L)
                a[name] += float(kl.sum())
            a["nb"] += raw.size(0) * raw.size(1) * L          # query rows counted
    for h in handles:
        h.remove()

    layers = []
    for li, a in enumerate(acc):
        sigma_raw = math.sqrt(a["raw_sq"] / a["raw_n"])
        row = {"layer": li + 1, "sigma_raw": round(sigma_raw, 4)}
        for name, s in sc.items():
            row[f"sigma_post_{name}"] = round(sigma_raw * s, 4)
            row[f"H_{name}"] = round(a["H"][name] / a["nb"], 4)
        row["kl_std_dk"] = round(a["kl_sd_dk"] / a["nb"], 5)
        row["kl_std_temp"] = round(a["kl_sd_tp"] / a["nb"], 5)
        # power-law prediction for KL(std||dk): 0.5*(b1-b2)^2*sigma^2*(1-1/nbar)
        nbar = (L + 1) / 2.0
        row["law_kl_std_dk"] = round(
            0.5 * (b_std - b_dk) ** 2 * sigma_raw ** 2 * (1 - 1 / nbar), 5)
        layers.append(row)
    return layers


def run_model(name, seqs):
    from transformers import GPT2LMHeadModel
    print(f"--- {name} ---", flush=True)
    model = GPT2LMHeadModel.from_pretrained(f"openai-community/{name}",
                                            attn_implementation="eager")
    model.eval()
    d_head = model.config.n_embd // model.config.n_head
    out = {"n_params": int(sum(p.numel() for p in model.parameters())),
           "n_layer": model.config.n_layer, "d_head": d_head,
           "n_head": model.config.n_head,
           "training_divisor": "standard",
           "variants": {}}

    layers = stats_pass(model, seqs, d_head)
    out["layers"] = layers
    out["sigma_raw_mean"] = round(float(np.mean([l["sigma_raw"] for l in layers])), 4)
    out["sigma_post_standard_mean"] = round(
        float(np.mean([l["sigma_post_standard"] for l in layers])), 4)

    for variant, s in scalings(d_head).items():
        for block in model.transformer.h:
            attn = next(m for n, m in block.named_children()
                        if hasattr(m, "c_attn"))
            attn.scaling = s
        ppl, nll = ppl_pass(model, seqs)  # scaling already patched into attn modules
        out["variants"][variant] = {"scaling": round(s, 6), "ppl": round(ppl, 2),
                                    "val_nll": round(nll, 4)}
        print(f"  {variant:>9}: ppl={ppl:8.2f}  (scaling={s:.5f})", flush=True)

    st_ = out["variants"]["standard"]["ppl"]
    for variant in ("dk", "temp"):
        p = out["variants"][variant]["ppl"]
        out["variants"][variant]["ppl_delta_pct"] = round(100 * (p - st_) / st_, 2)
    kl = float(np.mean([l["kl_std_dk"] for l in layers]))
    out["kl_std_dk_mean"] = round(kl, 5)
    out["law_kl_std_dk_mean"] = round(
        float(np.mean([l["law_kl_std_dk"] for l in layers])), 5)
    del model
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt2,gpt2-medium")
    ap.add_argument("--nseq", type=int, default=160)
    ap.add_argument("--seqlen", type=int, default=128)
    args = ap.parse_args()

    torch.set_num_threads(1)
    seqs, _ = build_tokens(args.nseq, args.seqlen)
    print(f"{len(seqs)} sequences x {args.seqlen} tokens", flush=True)

    out = {"protocol": {"data": "wikitext-2-raw-v1 validation",
                        "n_seq": len(seqs), "seq_len": args.seqlen,
                        "intervention": "inference-time divisor swap, no fine-tuning",
                        "variants": {"standard": "1/sqrt(d_k) (training divisor)",
                                     "dk": "1/d_k", "temp": "tau=2"}}}
    for name in args.models.split(","):
        name = name.strip()
        try:
            res = run_model(name, seqs)
            # merge into the existing JSON so separate invocations accumulate
            try:
                prev = json.load(open(f"{RESULTS}/frontier_gpt2.json"))
            except Exception:
                prev = {}
            prev["protocol"] = out["protocol"]
            prev[name] = res
            with open(f"{RESULTS}/frontier_gpt2.json", "w") as f:
                json.dump(prev, f, indent=1)
        except Exception as e:                      # e.g. OOM on 3 GB sandbox
            print(f"  {name}: FAILED {e!r}", flush=True)
    print(json.dumps({k: v.get("ppl") for k, v in
                      json.load(open(f"{RESULTS}/frontier_gpt2.json")).items()
                      if isinstance(v, dict) and "ppl" in v}, indent=1))


if __name__ == "__main__":
    main()
