"""
Data preparation for attention scaling experiments using HuggingFace datasets.

Supports:
- SST-2: Binary sentiment classification (GLUE)
- IMDB: Binary sentiment classification
- AG News: 4-class topic classification
- WikiText-2: Word-level language modelling

All datasets are loaded with consistent preprocessing and matched seed design
for reproducibility across attention scaling variants.
"""

import json
import random
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class PreparedData:
    """Container for prepared dataset tensors."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    vocab: Dict[str, int]
    vocab_size: int
    max_len: int
    metadata: Dict[str, Any]


class DataCache:
    """Manages cached datasets with SHA-256 verification."""

    def __init__(self, cache_dir: str = "./artifacts/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, str]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _sanitize_key(self, key: str) -> str:
        """Replace unsafe characters in cache key for filesystem compatibility."""
        return key.replace("/", "_").replace("\\", "_").replace(":", "_").replace("None", "null")

    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._sanitize_key(key)}.pt"

    def get_manifest_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._sanitize_key(key)}_manifest.json"

    def is_cached(self, key: str) -> bool:
        cache_path = self.get_cache_path(key)
        manifest_path = self.get_manifest_path(key)
        return cache_path.exists() and manifest_path.exists()

    def save(self, key: str, data: Any, metadata: Optional[Dict] = None):
        cache_path = self.get_cache_path(key)
        manifest_path = self.get_manifest_path(key)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache_path)

        # Compute SHA-256
        with open(cache_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        manifest = {
            "key": key,
            "sha256": sha256,
            "metadata": metadata or {},
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.manifest[key] = sha256
        self._save_manifest()

    def load(self, key: str) -> Any:
        cache_path = self.get_cache_path(key)
        manifest_path = self.get_manifest_path(key)

        if not cache_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Cache not found for key: {key}")

        # Verify SHA-256
        with open(cache_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        with open(manifest_path) as f:
            manifest = json.load(f)

        if sha256 != manifest["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {key}: expected {manifest['sha256']}, got {sha256}")

        return torch.load(cache_path)


def download_and_cache_dataset(
    cache: DataCache,
    hf_dataset: str,
    hf_subset: Optional[str],
    split: str,
    cache_key: str,
    force_download: bool = False,
) -> Any:
    """Download dataset from HuggingFace and cache it."""
    if cache.is_cached(cache_key) and not force_download:
        print(f"Loading {cache_key} from cache...")
        return cache.load(cache_key)

    print(f"Downloading {hf_dataset}/{hf_subset or 'default'} [{split}]...")
    dataset = load_dataset(hf_dataset, hf_subset, split=split)
    data = list(dataset)

    metadata = {
        "hf_dataset": hf_dataset,
        "hf_subset": hf_subset,
        "split": split,
        "num_examples": len(data),
    }
    cache.save(cache_key, data, metadata)
    return data


class SimpleTokenizer:
    """Simple whitespace + punctuation tokenizer for classification."""

    def __init__(self, vocab: Dict[str, int], max_len: int, unk_token: str = "UNK", pad_token: str = "<pad>"):
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = vocab.get(unk_token, 1)
        self.pad_id = vocab.get(pad_token, 0)
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> List[int]:
        # Simple lowercase regex-like tokenization
        import re
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)


class LMTokenizer:
    """Word-level tokenizer for language modeling."""

    def __init__(self, vocab: Dict[str, int], max_len: int, unk_token: str = "UNK", pad_token: str = "<pad>", eos_token: str = "<eos>"):
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = vocab.get(unk_token, 1)
        self.pad_id = vocab.get(pad_token, 0)
        self.eos_id = vocab.get(eos_token, 2)
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)


def build_vocab_classification(texts: List[str], max_vocab: int = 20000, min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from classification texts."""
    counter = Counter()
    import re
    for text in texts:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        counter.update(tokens)

    # Special tokens
    vocab = {"<pad>": 0, "UNK": 1}
    for token, freq in counter.most_common(max_vocab - 2):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def build_vocab_lm(texts: List[str], max_vocab: int = 30000, min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from LM texts (word-level)."""
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    vocab = {"<pad>": 0, "UNK": 1, "<eos>": 2}
    for token, freq in counter.most_common(max_vocab - 3):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def prepare_classification_data(
    config: Dict[str, Any],
    cache: DataCache,
    seed: int = 42,
) -> PreparedData:
    """Prepare classification dataset (SST-2, IMDB, AG News)."""
    ds_config = config["dataset"]
    train_config = config["training"]
    data_config = config["data"]

    hf_dataset = ds_config["hf_dataset"]
    hf_subset = ds_config.get("hf_subset")
    text_col = ds_config["text_column"]
    label_col = ds_config["label_column"]
    max_len = ds_config["max_len"]
    num_classes = ds_config["num_classes"]

    # Download splits
    train_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "train", f"{hf_dataset}_{hf_subset}_train")
    test_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "test", f"{hf_dataset}_{hf_subset}_test")

    # Try to download validation; if dataset has no validation split, split train instead
    try:
        val_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "validation", f"{hf_dataset}_{hf_subset}_val")
    except (ValueError, FileNotFoundError, Exception):
        print(f"  No validation split found for {hf_dataset}. Splitting train into train/val.")
        val_split_frac = data_config.get("val_split_frac", 0.2)
        random.seed(seed)
        random.shuffle(train_data)
        val_size = int(len(train_data) * val_split_frac)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]

    # Handle GLUE-style datasets where test set has no labels (labels = -1)
    # Check if test labels are valid
    test_labels = [ex[label_col] for ex in test_data]
    if len(test_labels) > 0 and (min(test_labels) < 0 or max(test_labels) >= num_classes):
        print(f"  Test set has invalid labels (min={min(test_labels)}, max={max(test_labels)}). Using validation as test.")
        # Use validation as test, split train into train/val
        test_data = val_data
        val_split_frac = data_config.get("val_split_frac", 0.2)
        random.seed(seed)
        random.shuffle(train_data)
        val_size = int(len(train_data) * val_split_frac)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]

    # Subsample if specified
    train_subset = data_config.get("train_subset_size")
    val_subset = data_config.get("val_subset_size")
    test_subset = data_config.get("test_subset_size")

    if train_subset and len(train_data) > train_subset:
        random.seed(data_config.get("train_split_seed", 42))
        random.shuffle(train_data)
        train_data = train_data[:train_subset]

    if val_subset and len(val_data) > val_subset:
        random.seed(data_config.get("train_split_seed", 42) + 1)
        random.shuffle(val_data)
        val_data = val_data[:val_subset]

    if test_subset and len(test_data) > test_subset:
        random.seed(data_config.get("train_split_seed", 42) + 2)
        random.shuffle(test_data)
        test_data = test_data[:test_subset]

    # Build vocab from train texts only
    train_texts = [ex[text_col] for ex in train_data]
    vocab = build_vocab_classification(train_texts, max_vocab=config["model"]["vocab_size"])
    vocab_size = len(vocab)

    # Tokenize
    tokenizer = SimpleTokenizer(vocab, max_len)

    X_train = tokenizer.encode_batch([ex[text_col] for ex in train_data])
    y_train = np.array([ex[label_col] for ex in train_data], dtype=np.int64)

    X_val = tokenizer.encode_batch([ex[text_col] for ex in val_data])
    y_val = np.array([ex[label_col] for ex in val_data], dtype=np.int64)

    X_test = tokenizer.encode_batch([ex[text_col] for ex in test_data])
    y_test = np.array([ex[label_col] for ex in test_data], dtype=np.int64)

    metadata = {
        "dataset": ds_config["name"],
        "task": "classification",
        "num_classes": num_classes,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "vocab_size": vocab_size,
        "max_len": max_len,
        "train_split_seed": data_config.get("train_split_seed", 42),
    }

    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        vocab=vocab, vocab_size=vocab_size, max_len=max_len,
        metadata=metadata,
    )


def prepare_lm_data(
    config: Dict[str, Any],
    cache: DataCache,
    seed: int = 42,
) -> PreparedData:
    """Prepare language modeling dataset (WikiText-2)."""
    ds_config = config["dataset"]
    model_config = config["model"]
    data_config = config["data"]

    hf_dataset = ds_config["hf_dataset"]
    hf_subset = ds_config.get("hf_subset")
    text_col = ds_config["text_column"]
    max_len = ds_config["max_len"]

    # Download splits
    train_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "train", f"{hf_dataset}_{hf_subset}_train")
    val_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "validation", f"{hf_dataset}_{hf_subset}_val")
    test_data = download_and_cache_dataset(cache, hf_dataset, hf_subset, "test", f"{hf_dataset}_{hf_subset}_test")

    # Concatenate all text for vocab building (train only)
    train_texts = [ex[text_col] for ex in train_data if ex[text_col].strip()]
    vocab = build_vocab_lm(train_texts, max_vocab=model_config["vocab_size"])
    vocab_size = len(vocab)

    # For LM, we create sequences of length max_len
    tokenizer = LMTokenizer(vocab, max_len)

    def create_sequences(texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create (input, target) pairs for LM."""
        all_input_ids = []
        all_target_ids = []

        for text in texts:
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                continue
            # Create overlapping windows
            for i in range(0, len(ids) - max_len, max_len // 2):
                window = ids[i:i + max_len + 1]
                if len(window) == max_len + 1:
                    all_input_ids.append(window[:-1])
                    all_target_ids.append(window[1:])

        if not all_input_ids:
            return np.array([]), np.array([])

        return np.array(all_input_ids, dtype=np.int64), np.array(all_target_ids, dtype=np.int64)

    X_train, y_train = create_sequences(train_texts)
    X_val, y_val = create_sequences([ex[text_col] for ex in val_data if ex[text_col].strip()])
    X_test, y_test = create_sequences([ex[text_col] for ex in test_data if ex[text_col].strip()])

    # Subsample if specified
    train_subset = data_config.get("train_subset_size")
    if train_subset and len(X_train) > train_subset:
        np.random.seed(seed)
        idx = np.random.choice(len(X_train), train_subset, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    metadata = {
        "dataset": ds_config["name"],
        "task": "language_modeling",
        "train_sequences": len(X_train),
        "val_sequences": len(X_val),
        "test_sequences": len(X_test),
        "vocab_size": vocab_size,
        "max_len": max_len,
    }

    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        vocab=vocab, vocab_size=vocab_size, max_len=max_len,
        metadata=metadata,
    )


def prepare_data(config: Dict[str, Any], cache: DataCache, seed: int = 42) -> PreparedData:
    """Main entry point - dispatch based on task type."""
    task = config["dataset"]["task"]
    if task == "classification":
        return prepare_classification_data(config, cache, seed)
    elif task == "language_modeling":
        return prepare_lm_data(config, cache, seed)
    else:
        raise ValueError(f"Unknown task: {task}")


def save_prepared_data(data: PreparedData, output_dir: Path):
    """Save prepared data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "data.npz",
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
    )
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(data.vocab, f)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(data.metadata, f, indent=2)


def load_prepared_data(data_dir: Path) -> PreparedData:
    """Load prepared data from disk."""
    npz = np.load(data_dir / "data.npz")
    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    return PreparedData(
        X_train=npz["X_train"], y_train=npz["y_train"],
        X_val=npz["X_val"], y_val=npz["y_val"],
        X_test=npz["X_test"], y_test=npz["y_test"],
        vocab=vocab, vocab_size=len(vocab), max_len=metadata["max_len"],
        metadata=metadata,
    )