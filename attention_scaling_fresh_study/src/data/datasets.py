"""
Dataset loading and preprocessing for attention scaling experiments.

Supports:
- SST-2: Binary sentiment classification (GLUE)
- IMDB: Binary sentiment classification
- AG News: 4-class topic classification
- WikiText-2: Word-level language modelling

All datasets are loaded with consistent preprocessing and matched seed design
for reproducibility across attention scaling variants.
"""

import os
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union


class TextClassificationDataset(Dataset):
    """
    Base class for text classification datasets.
    
    Handles tokenization, padding, and mask creation for classification tasks.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 max_seq_len: int = 128, pad_token_id: int = 0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Pad if needed
        padding_length = self.max_seq_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * padding_length
        tokens = tokens + [self.pad_token_id] * padding_length
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class LanguageModellingDataset(Dataset):
    """
    Dataset for language modelling with WikiText-2.
    
    Handles tokenization, truncation, and padding for language modelling.
    """
    
    def __init__(self, texts: List[str], tokenizer, max_seq_len: int = 128,
                 pad_token_id: int = 0, stride: int = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.stride = stride if stride is not None else max_seq_len
        
        # Tokenize all text and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        # Create overlapping chunks
        self.examples = []
        for i in range(0, len(all_tokens) - self.max_seq_len + 1, self.stride):
            chunk = all_tokens[i:i + self.max_seq_len]
            self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad if needed
        padding_length = self.max_seq_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * padding_length
        tokens = tokens + [self.pad_token_id] * padding_length
        
        # For language modelling: input = tokens[:-1], labels = tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_sst2_data(data_dir: str, max_seq_len: int = 128,
                   train_subset: int = 10000) -> Dict[str, List]:
    """
    Load SST-2 dataset.
    
    Args:
        data_dir: Path to SST-2 data directory
        max_seq_len: Maximum sequence length
        train_subset: Number of training examples to use
        
    Returns:
        Dictionary with train, validation, and test data
    """
    # Try to load from standard locations
    train_path = os.path.join(data_dir, "SST-2", "train.tsv")
    dev_path = os.path.join(data_dir, "SST-2", "dev.tsv")
    test_path = os.path.join(data_dir, "SST-2", "test.tsv")
    
    data = {}
    
    for split_name, file_path in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
        texts = []
        labels = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                header = lines[0].strip().split('\t')
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        texts.append(parts[1])
                        labels.append(int(parts[0]))
        
        data[split_name] = {
            "texts": texts,
            "labels": labels,
        }
    
    # Subsample training data if requested
    if train_subset and len(data["train"]["texts"]) > train_subset:
        indices = list(range(len(data["train"]["texts"])))
        np.random.seed(42)
        np.random.shuffle(indices)
        selected = indices[:train_subset]
        data["train"]["texts"] = [data["train"]["texts"][i] for i in selected]
        data["train"]["labels"] = [data["train"]["labels"][i] for i in selected]
    
    return data


def load_imdb_data(data_dir: str, max_seq_len: int = 128,
                   train_subset: int = 10000) -> Dict[str, List]:
    """
    Load IMDB dataset.
    
    Args:
        data_dir: Path to IMDB data directory
        max_seq_len: Maximum sequence length
        train_subset: Number of training examples to use
        
    Returns:
        Dictionary with train, validation, and test data
    """
    train_pos_path = os.path.join(data_dir, "IMDB", "train", "pos")
    train_neg_path = os.path.join(data_dir, "IMDB", "train", "neg")
    test_pos_path = os.path.join(data_dir, "IMDB", "test", "pos")
    test_neg_path = os.path.join(data_dir, "IMDB", "test", "neg")
    
    def load_imdb_files(pos_path, neg_path):
        texts = []
        labels = []
        
        if os.path.exists(pos_path):
            for fname in os.listdir(pos_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(pos_path, fname), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(1)
        
        if os.path.exists(neg_path):
            for fname in os.listdir(neg_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(neg_path, fname), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(0)
        
        return texts, labels
    
    data = {}
    
    # Load train data
    train_texts, train_labels = load_imdb_files(train_pos_path, train_neg_path)
    data["train"] = {"texts": train_texts, "labels": train_labels}
    
    # Load test data
    test_texts, test_labels = load_imdb_files(test_pos_path, test_neg_path)
    data["test"] = {"texts": test_texts, "labels": test_labels}
    
    # Create validation split from training data
    if len(train_texts) > 0:
        np.random.seed(42)
        indices = list(range(len(train_texts)))
        np.random.shuffle(indices)
        
        val_size = int(len(train_texts) * 0.2)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        data["validation"] = {
            "texts": [train_texts[i] for i in val_indices],
            "labels": [train_labels[i] for i in val_indices],
        }
        data["train"] = {
            "texts": [train_texts[i] for i in train_indices],
            "labels": [train_labels[i] for i in train_indices],
        }
    
    # Subsample training data if requested
    if train_subset and len(data["train"]["texts"]) > train_subset:
        np.random.seed(42)
        indices = list(range(len(data["train"]["texts"])))
        np.random.shuffle(indices)
        selected = indices[:train_subset]
        data["train"]["texts"] = [data["train"]["texts"][i] for i in selected]
        data["train"]["labels"] = [data["train"]["labels"][i] for i in selected]
    
    return data


def load_agnews_data(data_dir: str, max_seq_len: int = 128,
                     train_subset: int = 10000) -> Dict[str, List]:
    """
    Load AG News dataset.
    
    Args:
        data_dir: Path to AG News data directory
        max_seq_len: Maximum sequence length
        train_subset: Number of training examples to use
        
    Returns:
        Dictionary with train, validation, and test data
    """
    train_path = os.path.join(data_dir, "AG News", "train.csv")
    test_path = os.path.join(data_dir, "AG News", "test.csv")
    
    def load_agnews_csv(file_path):
        texts = []
        labels = []
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header if present
                start_idx = 0
                if lines and lines[0].startswith('"class'):
                    start_idx = 1
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: "class","title","description"
                    parts = line.split('",', 2)
                    if len(parts) >= 3:
                        label = int(parts[0].replace('"', ''))
                        text = parts[1].strip('"') + ' ' + parts[2].strip('"').rstrip('"')
                        texts.append(text)
                        labels.append(label - 1)  # Convert to 0-indexed
        
        return texts, labels
    
    data = {}
    
    # Load training data
    if os.path.exists(train_path):
        train_texts, train_labels = load_agnews_csv(train_path)
        data["train"] = {"texts": train_texts, "labels": train_labels}
    
    # Load test data
    if os.path.exists(test_path):
        test_texts, test_labels = load_agnews_csv(test_path)
        data["test"] = {"texts": test_texts, "labels": test_labels}
    
    # Create validation split from training data
    if len(train_texts) > 0:
        np.random.seed(42)
        indices = list(range(len(train_texts)))
        np.random.shuffle(indices)
        
        val_size = int(len(train_texts) * 0.2)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        data["validation"] = {
            "texts": [train_texts[i] for i in val_indices],
            "labels": [train_labels[i] for i in val_indices],
        }
        data["train"] = {
            "texts": [train_texts[i] for i in train_indices],
            "labels": [train_labels[i] for i in train_indices],
        }
    
    # Subsample training data if requested
    if train_subset and len(data["train"]["texts"]) > train_subset:
        np.random.seed(42)
        indices = list(range(len(data["train"]["texts"])))
        np.random.shuffle(indices)
        selected = indices[:train_subset]
        data["train"]["texts"] = [data["train"]["texts"][i] for i in selected]
        data["train"]["labels"] = [data["train"]["labels"][i] for i in selected]
    
    return data


def load_wikitext2_data(data_dir: str, max_seq_len: int = 128,
                        train_subset: int = 5000) -> Dict[str, List]:
    """
    Load WikiText-2 dataset for language modelling.
    
    Args:
        data_dir: Path to WikiText-2 data directory
        max_seq_len: Maximum sequence length
        train_subset: Number of training sequences to use
        
    Returns:
        Dictionary with train, validation, and test data
    """
    train_path = os.path.join(data_dir, "WikiText-2", "wiki.train.tokens")
    valid_path = os.path.join(data_dir, "WikiText-2", "wiki.valid.tokens")
    test_path = os.path.join(data_dir, "WikiText-2", "wiki.test.tokens")
    
    def load_wikitext2_file(file_path):
        texts = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
        return texts
    
    data = {}
    
    # Load training data
    if os.path.exists(train_path):
        train_texts = load_wikitext2_file(train_path)
        data["train"] = {"texts": train_texts}
    
    # Load validation data
    if os.path.exists(valid_path):
        val_texts = load_wikitext2_file(valid_path)
        data["validation"] = {"texts": val_texts}
    
    # Load test data
    if os.path.exists(test_path):
        test_texts = load_wikitext2_file(test_path)
        data["test"] = {"texts": test_texts}
    
    # Subsample training data if requested
    if train_subset and len(data["train"]["texts"]) > train_subset:
        np.random.seed(42)
        indices = list(range(len(data["train"]["texts"])))
        np.random.shuffle(indices)
        selected = indices[:train_subset]
        data["train"]["texts"] = [data["train"]["texts"][i] for i in selected]
    
    return data


def load_dataset(dataset_name: str, data_dir: str, max_seq_len: int = 128,
                 train_subset: int = None) -> Dict[str, List]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset (sst2, imdb, agnews, wikitext2)
        data_dir: Path to data directory
        max_seq_len: Maximum sequence length
        train_subset: Number of training examples to use
        
    Returns:
        Dictionary with train, validation, and test data
    """
    if dataset_name.lower() == "sst2":
        return load_sst2_data(data_dir, max_seq_len, train_subset)
    elif dataset_name.lower() == "imdb":
        return load_imdb_data(data_dir, max_seq_len, train_subset)
    elif dataset_name.lower() == "agnews":
        return load_agnews_data(data_dir, max_seq_len, train_subset)
    elif dataset_name.lower() == "wikitext2":
        return load_wikitext2_data(data_dir, max_seq_len, train_subset)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloader(dataset: Dataset, batch_size: int = 64,
                      shuffle: bool = True, num_workers: int = 4,
                      pin_memory: bool = True) -> DataLoader:
    """
    Create a DataLoader for a dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_dataset_hash(texts: List[str]) -> str:
    """
    Compute a hash of dataset texts for reproducibility tracking.
    
    Args:
        texts: List of text strings
        
    Returns:
        SHA-256 hash string
    """
    combined = "\n".join(texts)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()