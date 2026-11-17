"""
Unit tests for data preparation module.

Tests:
- Vocabulary building works correctly.
- Tokenizers encode/decode correctly.
- Data preparation produces correct shapes.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from data.prepare import (
    build_vocab_classification,
    build_vocab_lm,
    SimpleTokenizer,
    LMTokenizer,
    PreparedData,
    DataCache,
    download_and_cache_dataset,
)


def test_build_vocab_classification():
    """Test vocabulary building for classification."""
    texts = [
        "This is a test sentence.",
        "Another test sentence here.",
        "Short text.",
    ]
    vocab = build_vocab_classification(texts, max_vocab=100, min_freq=1)

    # Check special tokens
    assert "<pad>" in vocab
    assert "UNK" in vocab
    assert vocab["<pad>"] == 0
    assert vocab["UNK"] == 1

    # Check that tokens from texts are in vocab
    assert "test" in vocab
    assert "sentence" in vocab
    assert "this" in vocab
    assert "is" in vocab
    assert "a" in vocab

    print("  build_vocab_classification OK")


def test_build_vocab_lm():
    """Test vocabulary building for language modeling."""
    texts = [
        "This is a test sentence .",
        "Another test sentence here .",
        "Short text .",
    ]
    vocab = build_vocab_lm(texts, max_vocab=100, min_freq=1)

    # Check special tokens
    assert "<pad>" in vocab
    assert "UNK" in vocab
    assert "<eos>" in vocab
    assert vocab["<pad>"] == 0
    assert vocab["UNK"] == 1
    assert vocab["<eos>"] == 2

    # Check that tokens from texts are in vocab (LM tokenizer uses whitespace split, preserves case)
    assert "test" in vocab
    assert "sentence" in vocab
    assert "This" in vocab  # LM tokenizer preserves case
    assert "is" in vocab

    print("  build_vocab_lm OK")


def test_simple_tokenizer():
    """Test SimpleTokenizer encode/decode."""
    vocab = {"<pad>": 0, "UNK": 1, "hello": 2, "world": 3, "test": 4}
    tokenizer = SimpleTokenizer(vocab, max_len=10)

    # Test encoding
    text = "Hello world test"
    ids = tokenizer.encode(text)
    assert len(ids) == 10
    assert ids[0] == 2  # hello
    assert ids[1] == 3  # world
    assert ids[2] == 4  # test
    assert ids[3] == 0  # pad
    assert ids[-1] == 0  # pad

    # Test truncation
    long_text = " ".join(["hello"] * 20)
    ids = tokenizer.encode(long_text)
    assert len(ids) == 10
    assert all(id == 2 for id in ids)

    # Test batch encoding
    texts = ["Hello world", "test hello"]
    batch = tokenizer.encode_batch(texts)
    assert batch.shape == (2, 10)
    assert batch.dtype == np.int64

    print("  SimpleTokenizer OK")


def test_lm_tokenizer():
    """Test LMTokenizer encode/decode."""
    vocab = {"<pad>": 0, "UNK": 1, "<eos>": 2, "hello": 3, "world": 4, "test": 5}
    tokenizer = LMTokenizer(vocab, max_len=10)

    # Test encoding
    text = "hello world test"
    ids = tokenizer.encode(text)
    assert len(ids) == 3
    assert ids == [3, 4, 5]

    # Test batch encoding - returns list of arrays (no padding)
    texts = ["hello world", "test hello"]
    batch = tokenizer.encode_batch(texts)
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.int64
    assert batch.shape[0] == 2  # 2 texts
    # Each row can have different length (no padding in LM tokenizer)
    assert batch[0].shape == (2,)  # "hello world" -> 2 tokens
    assert batch[1].shape == (2,)  # "test hello" -> 2 tokens

    print("  LMTokenizer OK")


def test_prepared_data_container():
    """Test PreparedData container."""
    data = PreparedData(
        X_train=np.zeros((10, 128), dtype=np.int64),
        y_train=np.zeros(10, dtype=np.int64),
        X_val=np.zeros((5, 128), dtype=np.int64),
        y_val=np.zeros(5, dtype=np.int64),
        X_test=np.zeros((5, 128), dtype=np.int64),
        y_test=np.zeros(5, dtype=np.int64),
        vocab={"<pad>": 0, "UNK": 1},
        vocab_size=2,
        max_len=128,
        metadata={"dataset": "test", "task": "classification"},
    )

    assert data.X_train.shape == (10, 128)
    assert data.y_train.shape == (10,)
    assert data.vocab_size == 2
    assert data.max_len == 128

    print("  PreparedData container OK")


def test_datacache():
    """Test DataCache save/load with SHA-256 verification."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DataCache(cache_dir=tmpdir)

        # Test data
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        cache.save("test_key", test_data, {"meta": "data"})

        # Check files created
        cache_files = list(Path(tmpdir).glob("test_key*"))
        assert len(cache_files) >= 2  # .pt and _manifest.json

        # Load and verify
        loaded = cache.load("test_key")
        assert loaded == test_data

        # Test cache miss
        try:
            cache.load("nonexistent_key")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

        print("  DataCache OK")


def test_datacache_corruption_detection():
    """Test DataCache detects corrupted files."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DataCache(cache_dir=tmpdir)

        test_data = {"key": "value"}
        cache.save("test_key", test_data)

        # Corrupt the cache file
        cache_path = cache.get_cache_path("test_key")
        with open(cache_path, "wb") as f:
            f.write(b"corrupted data")

        # Should raise ValueError on load
        try:
            cache.load("test_key")
            assert False, "Should have raised ValueError for SHA-256 mismatch"
        except ValueError as e:
            assert "SHA-256 mismatch" in str(e)

        print("  DataCache corruption detection OK")


if __name__ == "__main__":
    print("Running data preparation unit tests...")
    test_build_vocab_classification()
    test_build_vocab_lm()
    test_simple_tokenizer()
    test_lm_tokenizer()
    test_prepared_data_container()
    test_datacache()
    test_datacache_corruption_detection()
    print("All data preparation tests PASSED")