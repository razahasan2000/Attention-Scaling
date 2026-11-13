"""Prepare tokenised replica datasets: AG News, SST-2, IMDB, WikiText-2.

Replica protocol (manuscript S2-replica):
  - classification: train pool 2,500 -> 2,000 train / 500 val (fixed RNG 42),
    test subsets: AG News 4,000 / SST-2 872 (official dev) / IMDB 2,500.
  - seq_len 64, per-dataset word-level vocabulary built on the train subset.
  - LM: WikiText-2 train stream -> first 96,000 tokens; official validation
    file -> first 40,000 tokens; vocab 8,000 built on the train block.
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (PROC_DIR, build_vocab, encode, tokenize_words,
                    tokenize_wikitext_line)

RAW = os.environ.get("AS_RAW", os.path.join(os.getcwd(), "data", "raw"))
SEQ = 64


def load_agnews():
    def read(path):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 3:
                    rows.append((int(r[0]) - 1, r[1] + " " + r[2]))
        return rows
    tr, te = read(f"{RAW}/agnews_train.csv"), read(f"{RAW}/agnews_test.csv")
    rng = np.random.RandomState(42)
    tr_idx = rng.permutation(len(tr))[:2500]
    te_idx = rng.permutation(len(te))[:4000]
    pool = [tr[i] for i in tr_idx]
    test = [te[i] for i in te_idx]
    return pool, test, 4


def load_sst2():
    def read(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    rows.append((int(parts[1]), parts[0]))
        return rows
    tr, te = read(f"{RAW}/SST-2/train.tsv"), read(f"{RAW}/SST-2/dev.tsv")
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(tr))[:2500]
    pool = [tr[i] for i in idx]
    return pool, te, 2


def load_imdb():
    tr = pd.read_parquet(f"{RAW}/imdb_train.parquet")
    te = pd.read_parquet(f"{RAW}/imdb_test.parquet")
    rng = np.random.RandomState(42)
    tr_idx = rng.permutation(len(tr))[:2500]
    te_idx = rng.permutation(len(te))[:2500]
    pool = [(int(tr.iloc[i]["label"]), tr.iloc[i]["text"]) for i in tr_idx]
    test = [(int(te.iloc[i]["label"]), te.iloc[i]["text"]) for i in te_idx]
    return pool, test, 2


def prep_classification(name, loader):
    pool, test, ncls = loader()
    toks_pool = [tokenize_words(t) for _, t in pool]
    toks_test = [tokenize_words(t) for _, t in test]
    vocab = build_vocab(toks_pool, max_vocab=12000, min_freq=2)
    Xp = np.array([encode(t, vocab, SEQ) for t in toks_pool], dtype=np.int32)
    yp = np.array([l for l, _ in pool], dtype=np.int64)
    Xt = np.array([encode(t, vocab, SEQ) for t in toks_test], dtype=np.int32)
    yt = np.array([l for l, _ in test], dtype=np.int64)
    tr, va = Xp[:2000], Xp[2000:]
    ytr, yva = yp[:2000], yp[2000:]
    np.savez_compressed(f"{PROC_DIR}/{name}.npz", Xtr=tr, ytr=ytr, Xva=va,
                        yva=yva, Xt=Xt, yt=yt)
    meta = {"name": name, "n_classes": ncls, "vocab": len(vocab),
            "train": int(len(tr)), "val": int(len(va)), "test": int(len(Xt)),
            "seq": SEQ, "unk_rate_test": float((Xt == 1).mean())}
    with open(f"{PROC_DIR}/{name}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(meta)


def prep_lm():
    df = pd.read_parquet(f"{RAW}/wt2_train.parquet")
    toks = []
    for t in df["text"].tolist():
        toks.extend(tokenize_wikitext_line(t))
    train_ids_raw = toks[:96000]
    vocab = build_vocab([train_ids_raw], max_vocab=8000, min_freq=2)
    tr = np.array([vocab.get(t, 1) for t in train_ids_raw], dtype=np.int32)
    dfv = pd.read_parquet(f"{RAW}/wt2_valid.parquet")
    vtoks = []
    for t in dfv["text"].tolist():
        vtoks.extend(tokenize_wikitext_line(t))
    vtoks = vtoks[:40000]
    va = np.array([vocab.get(t, 1) for t in vtoks], dtype=np.int32)
    np.save(f"{PROC_DIR}/wikitext2_train.npy", tr)
    np.save(f"{PROC_DIR}/wikitext2_valid.npy", va)
    meta = {"name": "wikitext2", "vocab": len(vocab),
            "train_tokens": int(len(tr)), "val_tokens": int(len(va)),
            "unk_rate_val": float((va == 1).mean()),
            "unk_rate_train": float((tr == 1).mean())}
    with open(f"{PROC_DIR}/wikitext2.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(meta)


if __name__ == "__main__":
    os.makedirs(PROC_DIR, exist_ok=True)
    prep_classification("agnews", load_agnews)
    prep_classification("sst2", load_sst2)
    prep_classification("imdb", load_imdb)
    prep_lm()
    print("done")
