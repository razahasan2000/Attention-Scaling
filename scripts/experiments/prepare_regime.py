"""Prepare the Stage-3 'regime-restored' AG News dataset.

Same tokenisation/vocabulary protocol as prepare_data.py, but with a larger
training pool so that training can actually accumulate optimisation steps:
  - pool 6,500 -> 6,000 train / 500 val (fixed RNG 42), test 4,000
  - seq_len 64 (unchanged: the operating point is manipulated via training
    length and logit gain, not via sequence length)
Output: PROC/agnews_big.npz + agnews_big.meta.json
"""

import json
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_agnews, SEQ
from common import PROC_DIR, build_vocab, encode, tokenize_words


def main():
    import csv
    RAW = os.environ.get("AS_RAW", os.path.join(os.getcwd(), "data", "raw"))

    def read(path):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 3:
                    rows.append((int(r[0]) - 1, r[1] + " " + r[2]))
        return rows

    tr = read(f"{RAW}/agnews_train.csv")
    te = read(f"{RAW}/agnews_test.csv")
    rng = np.random.RandomState(42)
    tr_idx = rng.permutation(len(tr))[:6500]
    te_idx = rng.permutation(len(te))[:4000]
    pool = [tr[i] for i in tr_idx]
    test = [te[i] for i in te_idx]
    ncls = 4

    toks_pool = [tokenize_words(t) for _, t in pool]
    toks_test = [tokenize_words(t) for _, t in test]
    vocab = build_vocab(toks_pool, max_vocab=12000, min_freq=2)
    Xp = np.array([encode(t, vocab, SEQ) for t in toks_pool], dtype=np.int32)
    yp = np.array([l for l, _ in pool], dtype=np.int64)
    Xt = np.array([encode(t, vocab, SEQ) for t in toks_test], dtype=np.int32)
    yt = np.array([l for l, _ in test], dtype=np.int64)
    trX, vaX = Xp[:6000], Xp[6000:]
    ytr, yva = yp[:6000], yp[6000:]
    np.savez_compressed(f"{PROC_DIR}/agnews_big.npz", Xtr=trX, ytr=ytr, Xva=vaX,
                        yva=yva, Xt=Xt, yt=yt)
    meta = {"name": "agnews_big", "n_classes": ncls, "vocab": len(vocab),
            "train": int(len(trX)), "val": int(len(vaX)), "test": int(len(Xt)),
            "seq": SEQ, "unk_rate_test": float((Xt == 1).mean())}
    with open(f"{PROC_DIR}/agnews_big.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(meta)


if __name__ == "__main__":
    main()
