# Attention-Scaling

Parameter-free comparison of three scaling functions in Transformer self-attention under a strict seed-matched design.

| Variant         | Formula                                            | Effective divisor \(s\)      |
|-----------------|----------------------------------------------------|------------------------------|
| **Standard**    | \(\mathrm{softmax}(QK^\top / \sqrt{d_k})\,V\)      | \(\sqrt{d_k}\)               |
| **1/\(d_k\)**   | \(\mathrm{softmax}(QK^\top / d_k)\,V\)             | \(d_k\)                      |
| **Temperature** | \(\mathrm{softmax}(QK^\top / (\tau\sqrt{d_k}))\,V\) | \(\tau\sqrt{d_k}\) (\(\tau=2\)) |

All variants share **bit-identical initialization**, the same data order, and the same optimizer trajectory. The only difference is a single divisor applied to the attention logits.

## Key Idea

The practical impact of the scaling function is governed by the **raw (pre-division) logit standard deviation** \(\sigma\):

\[
\mathrm{KL}(P_{b_1}\,\|\,P_{b_2}) \approx \frac12 (b_1 - b_2)^2\,\sigma^2\Bigl(1 - \tfrac1n\Bigr)
\]

- When \(\sigma \lesssim 1\) the choice of divisor is essentially inert.
- When \(\sigma \gtrsim 3\) the choice becomes decisive.

\(\sigma\) can be measured in a single forward pass (see `scripts/experiments/common.py`).

## Repository Layout

```
Attention-Scaling/
├── configs/                 # YAML experiment configs
├── src/
│   ├── attention/           # scaling variants (the one-line change)
│   ├── models/              # pre-LN Transformer encoder
│   ├── training/
│   └── evaluation/
├── scripts/
│   ├── experiments/         # Stage-2/3 runner (resume-safe, CPU-friendly)
│   │   ├── runner.py
│   │   ├── prepare_data.py
│   │   ├── verify_mask.py
│   │   ├── aggregate.py
│   │   ├── agg_stage3.py
│   │   ├── regime_law.py
│   │   └── frontier_gpt2.py
│   └── frontier/
│       └── frontier_gpu_1b7b.py   # 1B–7B inference-time divisor swap
├── tests/                   # unit tests (seed matching, mask, gradients)
├── results/                 # raw + aggregated run records
├── figures/                 # generated plots
├── requirements.txt
└── LICENSE
```

## Quick Start (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Prepare replica-scale data
python scripts/experiments/prepare_data.py

# Verify causal mask implementation
python scripts/experiments/verify_mask.py

# Run core grids (resume-safe)
python scripts/experiments/runner.py --phase core   --procs 2
python scripts/experiments/runner.py --phase lm     --procs 2
python scripts/experiments/runner.py --phase gain   --procs 2

# Aggregate statistics
python scripts/experiments/aggregate.py
python scripts/experiments/agg_stage3.py
```

## Frontier Probe (GPT-2 and larger models)

**CPU (GPT-2 / GPT-2-medium):**
```bash
python scripts/experiments/frontier_gpt2.py --models gpt2,gpt2-medium --nseq 160 --seqlen 128
```

**GPU (1B–7B models):**
```bash
python scripts/frontier/frontier_gpu_1b7b.py --model gpt2-xl --nseq 500 --seqlen 256
python scripts/frontier/frontier_gpu_1b7b.py --model EleutherAI/pythia-1.4b
python scripts/frontier/frontier_gpu_1b7b.py --model meta-llama/Meta-Llama-3.2-8B --dtype bfloat16
```

The script swaps only the attention scaling factor at inference time and reports perplexity + per-layer diagnostics.

## Unit Tests

```bash
pytest tests/
```

Covers bit-identical initialization, gradient flow, output shapes, and exact causal/padding mask behavior.

## License

MIT
