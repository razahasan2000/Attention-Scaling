"""
v2 unit tests: head-dimension grid d_k in {32, 64, 128}.

Design invariant: d_model stays 128 and n_heads = d_model / d_k, so the
Q/K/V/O projections are always Linear(128, 128) and the PARAMETER COUNT IS
IDENTICAL across the grid. Only the per-head split changes. The scaling
variants divide by d_k-dependent constants, so s differs by construction:

  d_k=32: s = sqrt(32) ~ 5.657 | d_k=64: s = 8 | d_k=128: s = sqrt(128) ~ 11.314
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from models.transformer_encoder import create_model

D_MODEL = 128
VOCAB = 200
SEQ = 32
HEAD_DIMS = [32, 64, 128]


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def test_head_dim_model_construction():
    for dk in HEAD_DIMS:
        n_heads = D_MODEL // dk
        model = create_model(variant="standard", d_model=D_MODEL,
                             n_heads=n_heads, n_layers=2, d_ff=256,
                             max_seq_len=SEQ, vocab_size=VOCAB,
                             task_type="classification", num_classes=4)
        assert model.n_heads == n_heads
        attn = model.layers[0].attention
        assert attn.d_k == dk, f"d_k mismatch: expected {dk}, got {attn.d_k}"
        assert attn.get_scaling_factor() == dk ** 0.5
    print("  construction & scaling factor OK")


def test_parameter_count_invariance():
    counts = {}
    for dk in HEAD_DIMS:
        torch.manual_seed(0)  # same seed -> same initialised values
        model = create_model(variant="standard", d_model=D_MODEL,
                             n_heads=D_MODEL // dk, n_layers=2, d_ff=256,
                             max_seq_len=SEQ, vocab_size=VOCAB,
                             task_type="classification", num_classes=4)
        counts[dk] = _count_params(model)
    assert len(set(counts.values())) == 1, f"param counts differ: {counts}"
    print(f"  parameter-count invariance OK ({counts[32]:,} params for all d_k)")


def test_seed_match_across_head_dims():
    """Same seed + same variant -> bit-identical init for every d_k arm."""
    refs = {}
    for dk in HEAD_DIMS:
        torch.manual_seed(1234)
        m1 = create_model(variant="standard", d_model=D_MODEL,
                          n_heads=D_MODEL // dk, n_layers=2, d_ff=256,
                          max_seq_len=SEQ, vocab_size=VOCAB,
                          task_type="classification", num_classes=4)
        torch.manual_seed(1234)
        m2 = create_model(variant="dk", d_model=D_MODEL,
                          n_heads=D_MODEL // dk, n_layers=2, d_ff=256,
                          max_seq_len=SEQ, vocab_size=VOCAB,
                          task_type="classification", num_classes=4)
        for (n1, p1), (n2, p2) in zip(m1.state_dict().items(),
                                      m2.state_dict().items()):
            assert torch.equal(p1, p2), f"init mismatch at {n1} (d_k={dk})"
        refs[dk] = m1
    print("  seed-matched init across variants and d_k OK")


def test_forward_shapes_all_dims():
    for dk in HEAD_DIMS:
        model = create_model(variant="standard", d_model=D_MODEL,
                             n_heads=D_MODEL // dk, n_layers=2, d_ff=256,
                             max_seq_len=SEQ, vocab_size=VOCAB,
                             task_type="classification", num_classes=4)
        x = torch.randint(1, VOCAB, (2, SEQ))
        out = model(input_ids=x, labels=torch.zeros(2, dtype=torch.long))
        assert out["logits"].shape == (2, 4)
        assert torch.isfinite(out["loss"])
    print("  forward shapes OK")


if __name__ == "__main__":
    test_head_dim_model_construction()
    test_parameter_count_invariance()
    test_seed_match_across_head_dims()
    test_forward_shapes_all_dims()
    print("All head-dimension grid tests PASSED")
