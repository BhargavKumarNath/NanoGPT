import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .positional_encoding import RotaryPositionalEmbeddings

class MultiHeadAttention(nn.Module):
    """A from scratch implementation of Multi-Head Self Attention"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Query, Key, Value, and the final output
        # Can combine Q, K, V into a single linear layer for efficiency
        self.wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = rope

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Attention mask. For causal masking, this should
                                           be a tensor of shape (seq_len, seq_len)
                                           where the upper triangle is `True`.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V and split
        # qkv: (batch_size, seq_len, 3 * embed_dim)
        qkv = self.wqkv(x)
        # q, k, v: each (batch_size, seq_len, embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape for multi-head computation
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary
        if self.rope:
            q, k = self.rope(q, k)

        # 3. Compute attention scores (scaled dot-product)
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # 4. Apply mask
        if mask is not None:
            # mask shape: (seq_len, seq_len)
            # attn_scores shape: (batch_size, num_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # 5. Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Apply weights to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        context = attn_weights @ v

        # 7. Concatenate heads and project back
        # Transpose and reshape to (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        return output
    
class GroupedQueryAttention(nn.Module):
    """A from scratch implementation of Grouped-Query Attention (GQA).
    Multi-Query Attention (MQA) is a special case of GQA where num_kv_heads=1
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be multiple of num_kv_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_kv_heads

        # Linear projections for Query, Key, and Value
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Efficiently repeats the key and value heads n_rep times.
        (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        # Use expand and then reshape to be memory-efficient
        return (
            x.unsqueeze(2)
             .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
             .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Grouped-Query Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Causal mask of shape (seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Reshape for multi-head computation
        # q: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: (batch_size, seq_len, num_kv_heads * head_dim) -> (batch_size, num_kv_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. Repeat K and V heads to match Q heads
        k = self.repeat_kv(k, self.kv_repeat_factor)
        v = self.repeat_kv(v, self.kv_repeat_factor)

        # 4. Compute attention scores (same as MHA from here)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = attn_weights @ v

        # 5. Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        return output

class SlidingWindowAttention(nn.Module):
    """A from scratch implementation of Sliding Window Attention with GQA
    
    Each token can only attend to a fixed size window of tokens before it, making it O(n*w) instead of O(n^2), where n is seq_len and w is window_size
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, window_size: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        # Can reuse the GQA implementation for the core logic
        self.gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout, bias)
        self.window_size = window_size
        self.mask = None # Cache for the attention mask

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device):
        """Creates a causal sliding window mask (True = masked)."""
        if self.mask is not None and self.mask.shape[-1] >= seq_len:
            return self.mask[:, :, :seq_len, :seq_len].to(device)

        # Initialize everything as masked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        # For each token position i, unmask the valid window range
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1  # include itself
            mask[i, start:end] = False

        # Add dimensions for batch and heads for broadcasting
        self.mask = mask.unsqueeze(0).unsqueeze(1)
        return self.mask.to(device)


    def forward(self, x: torch.Tensor):
        """
        Forward pass for Sliding Window Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Create the attention mask
        # The mask will have shape (1, 1, seq_len, seq_len) and will be broadcasted
        sliding_window_mask = self._create_sliding_window_mask(seq_len, x.device)

        # 2. Use the GQA forward pass with our generated mask
        # --- Replicating GQA's forward pass logic here ---
        q = self.gqa.wq(x)
        k = self.gqa.wk(x)
        v = self.gqa.wv(x)
        
        q = q.view(batch_size, seq_len, self.gqa.num_heads, self.gqa.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.gqa.num_kv_heads, self.gqa.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.gqa.num_kv_heads, self.gqa.head_dim).transpose(1, 2)
        
        k = self.gqa.repeat_kv(k, self.gqa.kv_repeat_factor)
        v = self.gqa.repeat_kv(v, self.gqa.kv_repeat_factor)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.gqa.head_dim ** -0.5)
        
        attn_scores = attn_scores.masked_fill(sliding_window_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.gqa.attn_dropout(attn_weights)
        
        context = attn_weights @ v
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.gqa.embed_dim)
        output = self.gqa.out_proj(context)
        output = self.gqa.resid_dropout(output)

        return output