"""
Scaled dot-product attention variants with different scaling functions.

Implements three attention scaling variants:
1. Standard: softmax(QK^T / sqrt(d_k)) V
2. Alternative ($1/d_k$): softmax(QK^T / d_k) V  
3. Temperature: softmax(QK^T / (tau * sqrt(d_k))) V

All variants are parameter-free rescalings of the same logits,
introducing no additional arithmetic operations or memory traffic.
This makes the design matched: for a fixed seed, every variant is
instantiated from bit-identical initial state and consumes identical
data batches, so paired differences reflect the scaling function alone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention with configurable scaling factor.
    
    Supports three scaling variants:
    - standard: logits / sqrt(d_k)
    - dk: logits / d_k
    - temp: logits / (tau * sqrt(d_k))
    """
    
    def __init__(self, d_k: int, scaling: str = "standard", tau: float = 2.0):
        """
        Args:
            d_k: Per-head dimension (d_model / n_heads)
            scaling: One of "standard", "dk", "temp"
            tau: Temperature factor for temperature scaling (default: 2.0)
        """
        super().__init__()
        self.d_k = d_k
        self.scaling = scaling
        self.tau = tau
        
        if scaling not in ("standard", "dk", "temp", "multiscale_v1", "multiscale_v2"):
            raise ValueError(f"Unknown scaling variant: {scaling}. "
                           f"Must be one of 'standard', 'dk', 'temp', 'multiscale_v1', 'multiscale_v2'")
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix of shape (batch_size, n_heads, seq_len, d_k)
            K: Key matrix of shape (batch_size, n_heads, seq_len, d_k)
            V: Value matrix of shape (batch_size, n_heads, seq_len, d_v)
            mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
                  or broadcastable shape
            
        Returns:
            Output tensor of shape (batch_size, n_heads, seq_len, d_v)
        """
        # Compute attention logits: Q @ K^T / sqrt(d_k) base scaling
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply variant-specific scaling
        if self.scaling == "standard":
            # Standard: already divided by sqrt(d_k)
            pass
        elif self.scaling == "dk":
            # 1/d_k: divide logits by sqrt(d_k) again (equivalent to dividing by d_k)
            logits = logits / math.sqrt(self.d_k)
        elif self.scaling == "temp":
            # Temperature: divide logits by tau * sqrt(d_k)
            logits = logits / self.tau
        elif self.scaling in ("multiscale_v1", "multiscale_v2"):
            # Per-head scaling will be applied in MultiHeadSelfAttention
            pass
        else:
            raise ValueError(f"Unknown scaling variant: {self.scaling}")
        
        # Apply attention mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50, max=50)
        
        # Apply softmax
        attention_weights = F.softmax(logits, dim=-1)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_weights, V)
        
        return output
    
    def get_scaling_factor(self) -> float:
        """Return the effective scaling factor s."""
        if self.scaling == "standard":
            return math.sqrt(self.d_k)
        elif self.scaling == "dk":
            return self.d_k
        elif self.scaling == "temp":
            return self.tau * math.sqrt(self.d_k)
        elif self.scaling in ("multiscale_v1", "multiscale_v2"):
            return math.sqrt(self.d_k)  # Base scale
        else:
            raise ValueError(f"Unknown scaling variant: {self.scaling}")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with configurable scaling variants.
    
    Implements pre-LN Transformer encoder block with multi-head self-attention.
    Supports multiple attention scaling variants for controlled empirical comparison.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 scaling: str = "standard", tau: float = 2.0,
                 per_head_scales: list = None):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            scaling: Scaling variant ("standard", "dk", "temp")
            tau: Temperature for temperature scaling
            per_head_scales: Optional per-head scaling factors for multiscale_v1
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.scaling = scaling
        self.tau = tau
        self.per_head_scales = per_head_scales
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize attention scaling function
        self.attention = ScaledDotProductAttention(
            d_k=self.d_k, scaling=scaling, tau=tau
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute multi-head self-attention.
        
        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor of shape (batch_size, seq_len, d_model)
            V: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = Q.shape
        
        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply per-head scaling for multiscale variants
        if self.scaling in ("multiscale_v1", "multiscale_v2") and self.per_head_scales is not None:
            # Apply per-head scaling directly to logits
            logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            for h in range(self.n_heads):
                logits[:, h, :, :] = logits[:, h, :, :] / self.per_head_scales[h]
            
            # Apply attention mask if provided
            if mask is not None:
                logits = logits.masked_fill(mask == 0, float("-inf"))
            
            # Clamp logits for numerical stability
            logits = torch.clamp(logits, min=-50, max=50)
            
            # Apply softmax
            attention_weights = F.softmax(logits, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Compute weighted sum of values
            attn_output = torch.matmul(attention_weights, V)
        else:
            # Apply attention
            attn_output = self.attention(Q, K, V, mask=mask)
        
        # Reshape back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output
    
    def get_scaling_factor(self) -> float:
        """Return the effective scaling factor s."""
        return self.attention.get_scaling_factor()
    
    def get_scaling_info(self) -> dict:
        """Return detailed scaling information."""
        return {
            "scaling": self.scaling,
            "tau": self.tau,
            "d_k": self.d_k,
            "scaling_factor": self.get_scaling_factor(),
            "per_head_scales": self.per_head_scales,
        }


def create_attention_variant(d_model: int, n_heads: int, variant: str,
                             tau: float = 2.0, per_head_scales: list = None):
    """
    Factory function to create attention variant.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        variant: Scaling variant ("standard", "dk", "temp")
        tau: Temperature for temperature scaling
        per_head_scales: Per-head scaling factors for multiscale variants
        
    Returns:
        MultiHeadSelfAttention instance
    """
    return MultiHeadSelfAttention(
        d_model=d_model,
        n_heads=n_heads,
        scaling=variant,
        tau=tau,
        per_head_scales=per_head_scales,
    )


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute attention entropy for head diversity analysis.
    
    Args:
        attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        
    Returns:
        Entropy tensor of shape (batch_size, n_heads, seq_len)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    attn_probs = attention_weights + eps
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
    return entropy


def compute_inter_head_cosine_similarity(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute inter-head cosine similarity for head diversity analysis.
    
    Args:
        attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        
    Returns:
        Cosine similarity matrix of shape (batch_size, n_heads, n_heads)
    """
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # Reshape to (batch_size, n_heads, seq_len * seq_len)
    attention_flat = attention_weights.view(batch_size, n_heads, -1)
    
    # Normalize each head's attention distribution
    attention_norm = attention_flat / (attention_weights.sum(dim=-1, keepdim=True) + 1e-12)
    
    # Compute cosine similarity between heads
    attention_norm = attention_norm / (attention_norm.norm(dim=-1, keepdim=True) + 1e-12)
    cosine_sim = torch.matmul(attention_norm, attention_norm.transpose(-2, -1))
    
    return cosine_sim