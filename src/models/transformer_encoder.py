"""
Compact pre-LN Transformer encoder architecture.

Implements a compact Transformer encoder with:
- Token embeddings + learned positional embeddings
- Two encoder blocks (LayerNorm -> Multi-Head Self-Attention -> Residual -> 
  LayerNorm -> Feed-Forward (GELU) -> Residual)
- Final LayerNorm
- Masked mean pooling over non-padding positions
- Task-dependent head (binary classification, 4-class classification, or LM head)

All variants are parameter-free modifications of the same architecture,
making the design matched for seed-matched controlled experiments.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.scaling import MultiHeadSelfAttention


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, d_model: int, max_seq_len: int = 128):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Single encoder block: LayerNorm -> Multi-Head Self-Attention -> Residual ->
    LayerNorm -> Feed-Forward (GELU) -> Residual.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 scaling: str = "standard", tau: float = 2.0,
                 per_head_scales: list = None):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            scaling=scaling,
            tau=tau,
            per_head_scales=per_head_scales,
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Attention -> Residual
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN: LayerNorm -> FFN -> Residual
        ffn_out = self.ff(self.norm2(x))
        x = x + self.dropout1(ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Compact pre-LN Transformer encoder.
    
    Architecture:
    - Token embeddings + learned positional embeddings
    - N encoder blocks (LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual)
    - Final LayerNorm
    - Task-dependent head (classification or language modelling)
    
    Scaling variants:
    - standard: softmax(QK^T / sqrt(d_k)) V
    - dk: softmax(QK^T / d_k) V
    - temp: softmax(QK^T / (tau * sqrt(d_k))) V
    
    All variants are parameter-free rescalings, making the design matched
    for seed-matched controlled experiments.
    """
    
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1, max_seq_len: int = 128,
                 vocab_size: int = 18855, scaling: str = "standard",
                 tau: float = 2.0, num_classes: int = 4,
                 per_head_scales: list = None, task_type: str = "classification"):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.scaling = scaling
        self.tau = tau
        self.task_type = task_type
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Encoder blocks
        self.layers = nn.ModuleList([
            MultiHeadSelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                scaling=scaling,
                tau=tau,
                per_head_scales=per_head_scales,
            )
            for _ in range(n_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Task-dependent head
        if task_type == "classification":
            self.classifier = nn.Linear(d_model, num_classes)
            self.task_type = "classification"
        elif task_type == "language_modelling":
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Tie weights with token embeddings
            self.lm_head.weight = self.token_embedding.weight
            self.task_type = "language_modelling"
        else:
            raise ValueError(f"Unknown task_type: {task_type}. "
                           "Must be 'classification' or 'language_modelling'")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> dict:
        """
        Forward pass through the Transformer encoder.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Ground truth labels for classification or LM
            
        Returns:
            Dictionary containing logits, loss (if labels provided), and attention weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings + positional embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Pass through encoder blocks
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        
        # Final LayerNorm
        x = self.final_norm(x)
        
        # Task-dependent head
        if self.task_type == "classification":
            # Masked mean pooling over non-padding positions
            if attention_mask is not None:
                mask_float = attention_mask.squeeze(1).squeeze(1).float()  # (batch, seq_len)
                mask_sum = mask_float.sum(dim=1, keepdim=True)  # (batch, 1)
                masked_sum = (x * mask_float.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
                pooled = masked_sum / mask_sum.clamp(min=1e-8)  # (batch, d_model)
            else:
                pooled = x.mean(dim=1)  # (batch, d_model)
            
            logits = self.classifier(pooled)  # (batch, num_classes)
            
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            
            return {
                "logits": logits,
                "loss": loss,
                "pooled": pooled,
            }
        
        elif self.task_type == "language_modelling":
            # Language modelling head
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
            
            loss = None
            if labels is not None:
                # Shift labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            
            return {
                "logits": logits,
                "loss": loss,
            }
    
    def get_scaling_info(self) -> dict:
        """Return scaling configuration information."""
        scaling_info = {
            "scaling": self.scaling,
            "tau": self.tau,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_k": self.d_model // self.n_heads,
        }
        
        # Get per-layer scaling info
        layer_scaling = []
        for layer in self.layers:
            layer_scaling.append(layer.attention.get_scaling_info())
        scaling_info["per_layer"] = layer_scaling
        
        return scaling_info


def create_model(variant: str = "standard", tau: float = 2.0,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1,
                 max_seq_len: int = 128, vocab_size: int = 18855,
                 task_type: str = "classification",
                 num_classes: int = 4,
                 per_head_scales: list = None) -> TransformerEncoder:
    """
    Factory function to create a Transformer model with specified scaling variant.
    
    Args:
        variant: Scaling variant ("standard", "dk", "temp")
        tau: Temperature for temperature scaling
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        task_type: Task type ("classification" or "language_modelling")
        num_classes: Number of classes (for classification)
        per_head_scales: Per-head scaling factors for multiscale variants
        
    Returns:
        TransformerEncoder instance
    """
    return TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        scaling=variant,
        tau=tau,
        num_classes=num_classes,
        per_head_scales=per_head_scales,
        task_type=task_type,
    )