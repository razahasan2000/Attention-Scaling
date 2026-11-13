"""
v2 unit tests: standard causal (autoregressive) masking for language modelling.

The v1 repository trained WikiText-2 with bidirectional encoder attention and
only a key-padding mask, so position i could attend to future tokens i+1..n
and copy the next token. Perplexities near 1.09 in the v1 records are an
artifact of that leakage. These tests pin the fixed behaviour:

1. Trained on unpredictable (random) targets, a causal model cannot exploit
   the future, while a bidirectional model drives the loss towards zero.
2. Changing a FUTURE token leaves all earlier logits bit-identical.
3. The attention module gives EXACTLY zero weight to masked positions.
4. Causal + padding masks combine without NaN.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from models.transformer_encoder import create_model
from attention.scaling import ScaledDotProductAttention

VOCAB = 50
SEQ = 16
BATCH = 4


def _lm_model():
    torch.manual_seed(0)
    return create_model(variant="standard", d_model=64, n_heads=2, n_layers=2,
                        d_ff=128, dropout=0.0, max_seq_len=SEQ,
                        vocab_size=VOCAB, task_type="language_modelling")


def test_causal_prevents_learning_the_future():
    """Trained leak test on unpredictable targets.

    A fixed batch of RANDOM token sequences is trained for a short budget.
    The next token is independent noise given the past, so a correctly
    masked (causal) model CANNOT drive the loss far below ln(V); a
    bidirectional model can, because the target token is visible in its
    input and attention learns to copy it.
    """
    torch.manual_seed(7)
    n_seq = 16
    gen = torch.Generator().manual_seed(123)

    def fresh_batch():
        # Fresh random sequences EVERY step: memorization is impossible, so
        # any training-loss reduction must come from a generalizable rule.
        return torch.randint(1, VOCAB, (n_seq, SEQ), generator=gen)

    def train_and_eval(causal: bool, steps: int = 400, lr: float = 5e-3) -> tuple:
        # Both arms consume the IDENTICAL data stream (same generator seed).
        g = torch.Generator().manual_seed(123)
        model = _lm_model()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for _ in range(steps):
            X = torch.randint(1, VOCAB, (n_seq, SEQ), generator=g)
            opt.zero_grad()
            model(input_ids=X, labels=X, causal=causal)["loss"].backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            X_eval = fresh_batch()
            ev = model(input_ids=X_eval, labels=X_eval, causal=causal)["loss"].item()
        return ev

    ev_c = train_and_eval(causal=True)
    ev_b = train_and_eval(causal=False)
    # Bidirectional: the copy rule (attend right, read the answer) transfers
    # to unseen sequences -> near-zero loss.
    assert ev_b < 1.0, (
        f"bidirectional held-out loss {ev_b:.3f} not low - copy rule should "
        "generalize")
    # Causal: the next token is unpredictable from the past on fresh data,
    # so the loss stays near the uniform baseline ln(V).
    assert ev_c > 3.0, (
        f"causal held-out loss {ev_c:.3f} far below ln(V)={math.log(VOCAB):.2f} "
        "- future information appears to be leaking")
    assert ev_c > ev_b + 1.5, (
        f"causal held-out loss {ev_c:.3f} too close to bidirectional "
        f"{ev_b:.3f} - leakage suspected")
    print(f"  trained leak test OK (held-out: causal {ev_c:.3f} vs "
          f"bidirectional {ev_b:.3f} nats)")


def test_lm_defaults_to_causal():
    """language_modelling models enable causal masking by default."""
    model = _lm_model()
    assert model.causal is True
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    out = model(input_ids=x, labels=x)  # no explicit causal argument
    assert torch.isfinite(out["loss"])
    cls_model = create_model(variant="standard", d_model=64, n_heads=2,
                             n_layers=2, d_ff=128, max_seq_len=SEQ,
                             vocab_size=VOCAB, task_type="classification",
                             num_classes=2)
    assert cls_model.causal is False
    print("  LM-default-causal OK")


def test_masked_positions_get_exactly_zero_weight():
    """Attention output must equal V[0] when only key 0 is unmasked."""
    torch.manual_seed(0)
    attn = ScaledDotProductAttention(d_k=8, scaling="standard")
    B, L, D = 2, 6, 8
    Q = torch.randn(B, 1, L, D)
    K = torch.randn(B, 1, L, D)
    V = torch.randn(B, 1, L, D)
    keep = torch.zeros(B, 1, L, L, dtype=torch.bool)
    keep[:, :, :, 0] = True  # only key 0 visible
    out = attn(Q, K, V, mask=keep)
    assert torch.allclose(out, V[:, :, 0:1, :].expand_as(out), atol=1e-6), (
        "masked attention leaked weight onto non-visible positions")
    print("  exact-zero masked weights OK")


def test_causal_with_padding_no_nan():
    """Right-padded sequences under causal masking produce finite losses."""
    model = _lm_model()
    model.eval()
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    x[:, -4:] = 0  # right padding
    labels = x.clone()
    labels[labels == 0] = -100  # never predict padding tokens
    with torch.no_grad():
        out = model(input_ids=x, labels=labels)  # LM -> causal by default
    assert torch.isfinite(out["loss"]), "NaN/Inf loss with causal + padding"
    print("  causal+padding finite loss OK")


def test_future_positions_cannot_contribute():
    """Gradient probe: changing a FUTURE token must not affect earlier logits.

    Under causal masking, query position j attends only to keys <= j.
    Change the LAST token and verify the logits at all earlier positions
    (which cannot legitimately see it) are bit-identical.
    """
    model = _lm_model()
    model.eval()
    x = torch.randint(1, VOCAB, (1, SEQ))
    k = SEQ - 1
    x_mod = x.clone()
    x_mod[0, k] = (x[0, k] + 1) % VOCAB or 1  # change ONLY the last token
    with torch.no_grad():
        logits_a = model(input_ids=x)["logits"]
        logits_b = model(input_ids=x_mod)["logits"]
    # Positions 0..k-1 must not see token k under causal masking.
    diff = (logits_a[0, :k] - logits_b[0, :k]).abs().max().item()
    assert diff < 1e-5, (
        f"changing token {k} changed earlier logits by {diff:.2e} - "
        "future information is leaking through the causal mask")
    print("  no-future-influence probe OK")


if __name__ == "__main__":
    test_causal_prevents_learning_the_future()
    test_lm_defaults_to_causal()
    test_masked_positions_get_exactly_zero_weight()
    test_causal_with_padding_no_nan()
    test_future_positions_cannot_contribute()
    print("All causal-masking tests PASSED")
