"""
Data preparation for attention scaling experiments.

Reconstructed v2 module (was missing from the repository). Implements the
preprocessing specification of the manuscript (Supplementary S2):

- Classification (AG News, SST-2, IMDB):
  * lowercase + regex tokenisation ``[a-z0-9']+``
  * training sets subsampled to ``train_subset_size`` (10,000) with a seeded
    held-out validation split (SST-2 uses the official GLUE validation split)
  * frozen official test partitions
  * vocabularies: top ``max_vocab`` (18,855) tokens at minimum frequency 2,
    built from training data only; specials ``<pad>``=0, ``UNK``=1
  * label maps preserve original class identities (AG News 1-4 -> 0-3)
  * right-padding to max_len (128)

- Language modelling (WikiText-2):
  * whitespace tokenisation (case preserved) with an ``<eos>`` token
  * vocabulary: top 18,855 tokens at minimum frequency 2 plus specials
    ``<pad>``=0, ``UNK``=1, ``<eos>``=2
  * token stream chopped into contiguous windows of ``max_len`` (128)
  * labels mirror inputs (next-token shift happens inside the model);
    padding positions (id 0) are marked -100 in the labels

IMPORTANT (v2): the language-modelling task is trained with standard causal
(autoregressive) masking enforced in ``models.transformer_encoder``; this
module only prepares the token stream.
"""

import os
import re
import json
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Tokenizers and vocabularies
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def build_vocab_classification(texts: List[str], max_vocab: int = 18855,
                               min_freq: int = 2) -> Dict[str, int]:
    """Build a lowercased classification vocabulary.

    Specials: ``<pad>``=0, ``UNK``=1. Tokens are extracted with the regex
    ``[a-z0-9']+`` over lowercased text.
    """
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(_TOKEN_RE.findall(text.lower()))
    vocab = {"<pad>": 0, "UNK": 1}
    for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def build_vocab_lm(texts: List[str], max_vocab: int = 18855,
                   min_freq: int = 2) -> Dict[str, int]:
    """Build a case-preserving LM vocabulary (whitespace splitting).

    Specials: ``<pad>``=0, ``UNK``=1, ``<eos>``=2.
    """
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<pad>": 0, "UNK": 1, "<eos>": 2}
    for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


class SimpleTokenizer:
    """Lowercased regex tokenizer with right-padding to ``max_len``."""

    def __init__(self, vocab: Dict[str, int], max_len: int = 128,
                 unk_token: str = "UNK", pad_token: str = "<pad>"):
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = vocab[unk_token]
        self.pad_id = vocab[pad_token]

    def encode(self, text: str) -> List[int]:
        tokens = _TOKEN_RE.findall(text.lower())[: self.max_len]
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)


class LMTokenizer:
    """Whitespace tokenizer for LM (case preserved, variable length)."""

    def __init__(self, vocab: Dict[str, int], max_len: int = 128,
                 unk_token: str = "UNK", pad_token: str = "<pad>",
                 eos_token: str = "<eos>"):
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = vocab[unk_token]
        self.pad_id = vocab[pad_token]
        self.eos_id = vocab[eos_token]

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        rows = [self.encode(t) for t in texts]
        try:
            return np.array(rows, dtype=np.int64)
        except ValueError:  # ragged rows -> object array
            return np.array([np.array(r, dtype=np.int64) for r in rows],
                            dtype=object)


# ---------------------------------------------------------------------------
# PreparedData container
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Container for prepared train/val/test arrays and the vocabulary."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    vocab: Dict[str, int]
    vocab_size: int
    max_len: int
    metadata: Dict = field(default_factory=dict)


def save_prepared_data(data: PreparedData, path: Path) -> None:
    """Save PreparedData to a directory (npz arrays + json vocab/meta)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "arrays.npz",
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
    )
    with open(path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(data.vocab, f)
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": data.vocab_size,
            "max_len": data.max_len,
            "metadata": data.metadata,
        }, f, indent=2)


def load_prepared_data(path: Path) -> PreparedData:
    """Load PreparedData from a directory written by save_prepared_data."""
    path = Path(path)
    arrays = np.load(path / "arrays.npz")
    with open(path / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return PreparedData(
        X_train=arrays["X_train"], y_train=arrays["y_train"],
        X_val=arrays["X_val"], y_val=arrays["y_val"],
        X_test=arrays["X_test"], y_test=arrays["y_test"],
        vocab=vocab,
        vocab_size=meta["vocab_size"],
        max_len=meta["max_len"],
        metadata=meta.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Dataset download / cache
# ---------------------------------------------------------------------------

def download_and_cache_dataset(hf_dataset: str, hf_subset: Optional[str],
                               cache_dir: str = "./artifacts/cache"):
    """Download a HuggingFace dataset into ``cache_dir`` (idempotent)."""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset(hf_dataset, hf_subset, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# DataCache with SHA-256 verification
# ---------------------------------------------------------------------------

class DataCache:
    """Disk cache with SHA-256 integrity verification.

    Files: ``{key}.pt`` (pickled object) and ``{key}_manifest.json``
    (SHA-256 + metadata).
    """

    def __init__(self, cache_dir: str = "./artifacts/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def _manifest_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}_manifest.json"

    def save(self, key: str, obj, metadata: Optional[Dict] = None) -> None:
        path = self.get_cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with open(self._manifest_path(key), "w", encoding="utf-8") as f:
            json.dump({"sha256": sha256, "metadata": metadata or {}}, f)

    def load(self, key: str):
        path = self.get_cache_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Cache miss for key: {key}")
        with open(self._manifest_path(key), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for cached object '{key}' "
                f"(expected {manifest['sha256']}, got {actual})")
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Classification preparation
# ---------------------------------------------------------------------------

def _prepare_classification(cfg: dict, cache: DataCache) -> PreparedData:
    """Prepare a classification dataset according to its config block."""
    from datasets import load_dataset

    name = cfg["name"]
    max_len = int(cfg.get("max_len", 128))
    subset_size = int(cfg.get("train_subset_size", 10000))
    val_frac = cfg.get("val_split_frac", 0.2)
    split_seed = int(cfg.get("train_split_seed", 42))
    text_col = cfg.get("text_column", "text")
    label_col = cfg.get("label_column", "label")
    num_classes = int(cfg.get("num_classes", 2))

    ds = load_dataset(cfg["hf_dataset"], cfg.get("hf_subset"),
                      cache_dir=str(cache.cache_dir))

    metadata = {"dataset": name, "task": "classification",
                "num_classes": num_classes, "max_len": max_len}

    # ---- training pool + validation split --------------------------------
    if name == "sst2":
        # Official GLUE splits: train 67,349 / validation 872 / test 1,821.
        train_texts = list(ds["train"][text_col])
        train_labels = list(ds["train"][label_col])
        val_texts = list(ds["validation"][text_col])
        val_labels = list(ds["validation"][label_col])
        metadata["val_source"] = "glue_official_validation"
    else:
        train_full = ds["train"]
        n = len(train_full[text_col])
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(n)
        n_val = int(round(n * float(val_frac)))
        val_idx = set(perm[:n_val].tolist())
        train_texts, train_labels, val_texts, val_labels = [], [], [], []
        t_col, l_col = train_full[text_col], train_full[label_col]
        for i in range(n):
            if i in val_idx:
                val_texts.append(t_col[i]); val_labels.append(int(l_col[i]))
            else:
                train_texts.append(t_col[i]); train_labels.append(int(l_col[i]))
        metadata["val_source"] = f"seeded_{val_frac}_split"

    # ---- training subsample (seeded shuffle) ------------------------------
    rng = np.random.RandomState(split_seed + 1)
    order = rng.permutation(len(train_texts))
    keep = order[:subset_size]
    train_texts = [train_texts[i] for i in keep]
    train_labels = [train_labels[i] for i in keep]

    # ---- label mapping -----------------------------------------------------
    if name == "agnews":
        # Defensive identity-preserving remap of the original 1..4 ids.
        label_map = {1: 0, 2: 1, 3: 2, 4: 3}
    else:
        label_map = {}

    def map_label(y):
        y = int(y)
        return label_map.get(y, y)

    train_labels = [map_label(y) for y in train_labels]
    val_labels = [map_label(y) for y in val_labels]

    # ---- vocabulary from training data only -------------------------------
    vocab = build_vocab_classification(train_texts, max_vocab=18855, min_freq=2)
    tokenizer = SimpleTokenizer(vocab, max_len=max_len)

    X_train = tokenizer.encode_batch(train_texts)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val = tokenizer.encode_batch(val_texts)
    y_val = np.array(val_labels, dtype=np.int64)

    # ---- official test partition -----------------------------------------
    test_texts = list(ds["test"][text_col])
    test_labels_raw = list(ds["test"][label_col])
    if all(int(y) < 0 for y in test_labels_raw):
        # GLUE test labels are hidden; fall back to the labeled validation
        # partition and record it explicitly in the metadata.
        X_test, y_test = X_val, y_val
        metadata["test_source"] = "validation_fallback (official test labels hidden)"
    else:
        X_test = tokenizer.encode_batch(test_texts)
        y_test = np.array([map_label(y) for y in test_labels_raw], dtype=np.int64)
        metadata["test_source"] = "official_test"

    metadata.update({
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "vocab_size": len(vocab),
    })
    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        vocab=vocab, vocab_size=len(vocab), max_len=max_len,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Language-modelling preparation (WikiText-2)
# ---------------------------------------------------------------------------

def _prepare_lm(cfg: dict, cache: DataCache) -> PreparedData:
    """Prepare WikiText-2 as a contiguous token stream chopped into windows."""
    from datasets import load_dataset

    max_len = int(cfg.get("max_len", 128))
    subset_rows = int(cfg.get("train_subset_size", 5000))

    ds = load_dataset(cfg["hf_dataset"], cfg.get("hf_subset"),
                      cache_dir=str(cache.cache_dir))
    text_col = cfg.get("text_column", "text")

    def clean_rows(split, limit=None):
        rows = []
        for line in ds[split][text_col]:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            rows.append(line)
            if limit is not None and len(rows) >= limit:
                break
        return rows

    train_rows = clean_rows("train", limit=subset_rows)
    val_rows = clean_rows("validation")
    test_rows = clean_rows("test")

    vocab = build_vocab_lm(train_rows, max_vocab=18855, min_freq=2)
    tokenizer = LMTokenizer(vocab, max_len=max_len)

    def rows_to_windows(rows):
        # Concatenate tokens with <eos> separators, then chop fixed windows.
        stream = []
        for row in rows:
            stream.extend(tokenizer.encode(row))
            stream.append(tokenizer.eos_id)
        n_windows = len(stream) // max_len
        arr = np.array(stream[: n_windows * max_len], dtype=np.int64)
        X = arr.reshape(n_windows, max_len)
        y = X.copy()
        y[y == tokenizer.pad_id] = -100  # never predict padding
        return X, y

    X_train, y_train = rows_to_windows(train_rows)
    X_val, y_val = rows_to_windows(val_rows)
    X_test, y_test = rows_to_windows(test_rows)

    metadata = {
        "dataset": "wikitext2", "task": "language_modelling",
        "max_len": max_len, "vocab_size": len(vocab),
        "train_rows": len(train_rows), "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)), "test_windows": int(len(X_test)),
        "causal_masking": True,
    }
    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        vocab=vocab, vocab_size=len(vocab), max_len=max_len,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prepare_data(config: dict, cache: Optional[DataCache] = None) -> PreparedData:
    """Prepare a dataset from a merged experiment config.

    ``config`` must contain a ``dataset`` block (one section of
    ``dataset_configs.yaml``). Classification and LM tasks are dispatched
    automatically.
    """
    if cache is None:
        cache = DataCache(config.get("data", {}).get("cache_dir", "./artifacts/cache"))
    dataset_cfg = config.get("dataset", {})
    name = dataset_cfg.get("name", "")
    max_len = int(dataset_cfg.get("max_len", 128))

    # v2: include max_len in the cache key so sequence-length sensitivity
    # arms (64/128/256) never collide with the baseline cache.
    cache_key = f"prepared_{name}_v2_causal_len{max_len}"
    try:
        return cache.load(cache_key)
    except (FileNotFoundError, ValueError):
        pass

    task = dataset_cfg.get("task", "classification")
    if task == "classification":
        data = _prepare_classification(dataset_cfg, cache)
    elif task in ("language_modeling", "language_modelling"):
        data = _prepare_lm(dataset_cfg, cache)
    else:
        raise ValueError(f"Unknown task for dataset '{name}': {task}")

    cache.save(cache_key, data, {"dataset": name, "task": task})
    return data
