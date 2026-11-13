"""Frontier-scale divisor-swap evaluation for pretrained decoders (1B-7B, GPU).

The CPU sandbox caps what can be *executed* (GPT-2 / GPT-2-medium, see
experiments/frontier_gpt2.py). This script is the 1B-7B counterpart: load any
HuggingFace decoder-LM checkpoint, swap ONLY the attention scaling divisor at
inference time (the checkpoint keeps its training-time weights), and measure
WikiText-2 perplexity plus per-layer operating-point diagnostics.

  variants
    standard : s = 1/sqrt(d_head)   (the model's training-time divisor)
    dk       : s = 1/d_head
    temp     : s = 1/(2 sqrt(d_head))   (temperature tau = 2)

  diagnostics per layer
    sigma_raw        std of raw q.k logits (pre-division)
    sigma_post_*     std after each divisor
    KL(std||dk), KL(std||temp)  on the SAME logits (the power-law contrast)
    law_kl_*         the 0.5*(b1-b2)^2*sigma^2*(1-1/n) prediction

Supported architectures (eager attention, scaling attribute per family):
  Llama / Qwen2 / Mistral          -> self_attn.scaling
  GPT-2                            -> attn.scaling
  GPT-NeoX (Pythia)                -> attention.norm_factor  (divisor = value)
  GPT-J                            -> attention.scale_attn

Examples
  # GPT-2-XL (1.5B), single GPU
  python frontier_gpu_1b7b.py --model gpt2-xl --nseq 500 --seqlen 256

  # Llama-3.2-8B on 80 GB (bf16)
  python frontier_gpu_1b7b.py --model meta-llama/Meta-Llama-3.2-8B \
      --nseq 500 --seqlen 256 --dtype bfloat16

  # Pythia scaling ladder for the size-vs-sigma figure
  for M in EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b; do
    python frontier_gpu_1b7b.py --model $M --nseq 500 --seqlen 256
  done

Output: JSON with per-variant perplexity (delta % vs standard), per-layer
sigma_raw / sigma_post / entropy / KL, and the protocol block.
"""
import argparse
import collections
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scaling patching, per architecture family
# ---------------------------------------------------------------------------
def scalings(d_head, tau=2.0):
    return {"standard": d_head ** -0.5,
            "dk": float(d_head) ** -1.0,
            "temp": (tau * math.sqrt(d_head)) ** -1.0}


def iter_attention_modules(model):
    """Yield (name, module, attribute) for every patchable attention module."""
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "GPTNeoXAttention":
            yield name, mod, "norm_factor"          # neox divides logits by it
        elif cls == "GPTJAttention":
            yield name, mod, "scale_attn"
        elif cls in ("LlamaAttention", "Qwen2Attention", "MistralAttention",
                     "Gemma2Attention", "FalconAttention", "GPT2Attention"):
            yield name, mod, "scaling"


def set_divisor(model, scaling):
    """Patch every attention module to the requested scaling factor."""
    n = 0
    for name, mod, attr in iter_attention_modules(model):
        cur = getattr(mod, attr, None)
        if attr == "scaling":
            mod.scaling = scaling
        elif attr == "norm_factor":
            mod.norm_factor = 1.0 / scaling
        elif attr == "scale_attn":
            mod.scale_attn = scaling
        n += 1
    return n


def head_dim_of(model):
    cfg = model.config
    if getattr(cfg, "head_dim", None):
        return int(cfg.head_dim)
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    if hidden and heads:
        return hidden // heads
    raise ValueError("cannot infer head dim; pass --d-head")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_tokens(tokenizer, nseq, seqlen, data_file=None):
    import pandas as pd
    if data_file:
        df = pd.read_parquet(data_file)
    else:
        from datasets import load_dataset
        df = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="validation").to_pandas()[["text"]]
    stream = tokenizer("\n\n".join(df["text"].tolist()),
                       return_tensors="np")["input_ids"][0]
    n = min(nseq, len(stream) // seqlen - 1)
    seqs = np.array([stream[i * seqlen:(i + 1) * seqlen] for i in range(n)])
    return torch.from_numpy(seqs).long()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def ppl_pass(model, seqs, bs, device):
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, len(seqs), bs):
        ids = seqs[i:i + bs].to(device)
        logits = model(ids).logits[:, :-1, :]
        tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              tgt.reshape(-1), reduction="sum")
        tot_nll += float(nll)
        tot_tok += int(tgt.numel())
    return math.exp(tot_nll / tot_tok), tot_nll / tot_tok


@torch.no_grad()
def stats_pass(model, seqs, d_head, device, bs=2, max_batches=2, tau=2.0):
    """Per-layer raw-logit operating point via hooks on q/k projections."""
    sc = scalings(d_head, tau)
    nl = getattr(model.config, "num_hidden_layers", None) or model.config.n_layer
    caps = {li: {} for li in range(nl)}
    handles = []

    def layer_idx(name):
        for tag in (".h.", ".layers.", ".gpt_neox.layers."):
            if tag in name:
                tail = name.split(tag)[1].split(".")[0]
                if tail.isdigit():
                    return int(tail)
        return 0

    for name, mod in model.named_modules():
        if name.endswith("q_proj") or name.endswith("q_attn"):
            handles.append(mod.register_forward_hook(
                lambda m, a, o, i=layer_idx(name): caps[i].__setitem__("q", o)))
        elif name.endswith("k_proj") or name.endswith("k_attn"):
            handles.append(mod.register_forward_hook(
                lambda m, a, o, i=layer_idx(name): caps[i].__setitem__("k", o)))
        elif name.endswith("query_key_value"):     # GPT-NeoX fused qkv
            handles.append(mod.register_forward_hook(
                lambda m, a, o, i=layer_idx(name): caps[i].__setitem__("qkv", o)))
        elif name.endswith("c_attn"):              # GPT-2 fused qkv
            handles.append(mod.register_forward_hook(
                lambda m, a, o, i=layer_idx(name): caps[i].__setitem__("qkv", o)))

    n_heads = getattr(model.config, "num_attention_heads", None) or model.config.n_head
    per_layer = collections.defaultdict(list)
    L = seqs.size(1)
    for i in range(0, min(len(seqs), bs * max_batches), bs):
        ids = seqs[i:i + bs].to(device)
        for c in caps.values():
            c.clear()
        model(ids)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        keep = causal.view(1, 1, L, L)
        for li, c in caps.items():
            if "qkv" in c:                          # fused qkv output
                o = c["qkv"]
                E = o.shape[-1] // 3
                q, k = o[..., :E], o[..., E:2 * E]
            elif "q" in c and "k" in c:
                q, k = c["q"], c["k"]
            else:
                continue
            q = q.view(*q.shape[:-1], n_heads, -1).transpose(1, 2)
            k = k.view(*k.shape[:-1], n_heads, -1).transpose(1, 2)
            raw = torch.matmul(q, k.transpose(-2, -1))          # (B,H,L,L)
            # valid-entry sigma + per-row masked H/KL, aggregated per layer
            valid = raw.masked_select(keep).float()
            sc_l = {nm: s for nm, s in sc.items()}
            ents, kls = {}, {}
            for nm, s in sc_l.items():
                z = (raw * s).masked_fill(~keep, float("-inf"))
                logw = (z - torch.logsumexp(z, -1, keepdim=True)).masked_fill(~keep, 0.0)
                ents[nm] = -(logw.exp() * logw).sum(-1)
            z_std = (raw * sc["standard"]).masked_fill(~keep, float("-inf"))
            lq = z_std - torch.logsumexp(z_std, -1, keepdim=True)
            for nm, b2 in (("kl_std_dk", sc["dk"]), ("kl_std_temp", sc["temp"])):
                z2 = (raw * b2).masked_fill(~keep, float("-inf"))
                lr = z2 - torch.logsumexp(z2, -1, keepdim=True)
                d = (lq - lr).masked_fill(~keep, 0.0)
                kls[nm] = (lq.exp().masked_fill(~keep, 0.0) * d).sum(-1)
            per_layer[li].append((valid, ents, kls, int(keep.sum()), L))
    for h in handles:
        h.remove()

    nbar = (L + 1) / 2.0
    b_std, b_dk, b_tp = sc["standard"], sc["dk"], sc["temp"]
    layers = []
    for li in sorted(per_layer):
        parts = per_layer[li]
        sq = sum(float((p[0] ** 2).sum()) for p in parts)
        n = sum(p[0].numel() for p in parts)
        sigma = math.sqrt(sq / n)
        row = {"layer": li + 1, "sigma_raw": round(sigma, 4)}
        for nm, s in sc.items():
            row[f"sigma_post_{nm}"] = round(sigma * s, 4)
            row[f"H_{nm}"] = round(sum(float(p[1][nm].sum()) for p in parts) /
                                   sum(p[1][nm].numel() for p in parts), 4)
        for nm in ("kl_std_dk", "kl_std_temp"):
            row[nm] = round(sum(float(p[2][nm].sum()) for p in parts) /
                            sum(p[2][nm].numel() for p in parts), 5)
        row["law_kl_std_dk"] = round(
            0.5 * (b_std - b_dk) ** 2 * sigma ** 2 * (1 - 1 / nbar), 5)
        layers.append(row)
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nseq", type=int, default=500)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--d-head", type=int, default=0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--data", default="", help="path to wt2 parquet (optional)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    d_head = args.d_head or head_dim_of(model)

    seqs = build_tokens(tok, args.nseq, args.seqlen, args.data or None)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
          f"d_head={d_head}, {len(seqs)} x {args.seqlen} tokens on {device}")

    layers = stats_pass(model, seqs, d_head, device, tau=args.tau)
    out = {"model": args.model,
           "n_params": int(sum(p.numel() for p in model.parameters())),
           "d_head": d_head, "tau": args.tau,
           "protocol": {"data": "wikitext-2 validation", "n_seq": len(seqs),
                        "seq_len": args.seqlen,
                        "intervention": "inference-time divisor swap"},
           "layers": layers,
           "sigma_raw_mean": round(sum(l["sigma_raw"] for l in layers) / len(layers), 4),
           "variants": {}}
    base = None
    for variant, s in scalings(d_head, args.tau).items():
        set_divisor(model, s)
        ppl, _ = ppl_pass(model, seqs, args.bs, device)
        out["variants"][variant] = {"scaling": round(s, 8), "ppl": round(ppl, 2)}
        base = base if base is not None else ppl
        if variant != "standard":
            out["variants"][variant]["ppl_delta_pct"] = round(
                100 * (ppl - base) / base, 2)
        print(f"  {variant:>9}: ppl={ppl:.2f}")

    out_path = args.out or f"frontier_{args.model.split('/')[-1]}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
