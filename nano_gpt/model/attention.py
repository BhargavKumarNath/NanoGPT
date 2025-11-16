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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_kv: tuple[torch.Tensor, torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            x (torch.Tensor): Input tensor. Shape (batch, seq_len, embed_dim)
            mask (torch.Tensor, optional): Causal mask.
            past_kv (tuple, optional): A tuple containing the cached key and value tensors.
            use_cache (bool): If True, return the updated key and value tensors.
            start_pos (int): Starting position for RoPE.
        """
        batch_size, seq_len, _ = x.shape
        
        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            q, k = self.rope(q, k, start_pos)
        
        # KV Cache Logic - now k and v already have correct positional encoding
        if past_kv is not None:
            # Concatenate the new k, v with the cached ones
            past_key, past_value = past_kv
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        # If use_cache is True, we store the new k,v for the next iteration
        present_kv = (k, v) if use_cache else None
        # End KV Cache Logic
        
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is not None:
            # q has length seq_len (which is 1 during generation after first step)
            # k has full length including cache
            kv_len = k.size(-2)
            attn_scores = attn_scores.masked_fill(mask[:, :, -seq_len:, :kv_len], float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = attn_weights @ v

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        if use_cache:
            return output, present_kv
        else:
            return output
    
class GroupedQueryAttention(nn.Module):
    """A from scratch implementation of Grouped-Query Attention (GQA).
    Multi-Query Attention (MQA) is a special case of GQA where num_kv_heads=1
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
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
        
        # Add RoPE
        self.rope = rope

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_kv: tuple[torch.Tensor, torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0):
        """
        Forward pass for Grouped-Query Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Causal mask.
            past_kv (tuple, optional): A tuple containing the cached key and value tensors.
            use_cache (bool): If True, return the updated key and value tensors.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
            (optional) tuple: New KV cache
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Reshape for multi-head computation
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply RoPE BEFORE handling cache
        if self.rope:
            q, k = self.rope(q, k, start_pos)
        
        # 4. KV Cache Logic
        if past_kv is not None:
            # Concatenate the new k, v with the cached ones
            past_key, past_value = past_kv
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        # If use_cache is True, we store the new k,v for the next iteration
        present_kv = (k, v) if use_cache else None
        # End KV Cache Logic

        # 5. Repeat K and V heads to match Q heads
        k_rep = self.repeat_kv(k, self.kv_repeat_factor)
        v_rep = self.repeat_kv(v, self.kv_repeat_factor)

        # 6. Compute attention scores
        attn_scores = (q @ k_rep.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is not None:
            kv_len = k_rep.size(-2)
            attn_scores = attn_scores.masked_fill(mask[:, :, -seq_len:, :kv_len], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = attn_weights @ v_rep

        # 7. Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        if use_cache:
            return output, present_kv
        else:
            return output

class SlidingWindowAttention(nn.Module):
    """A from scratch implementation of Sliding Window Attention with GQA
    
    Each token can only attend to a fixed size window of tokens before it, making it O(n*w) instead of O(n^2), where n is seq_len and w is window_size
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, window_size: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__()
        # GQA implementation for the core logic
        self.gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout, bias, rope=rope)
        self.window_size = window_size
        self.mask = None

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device):
        """Creates a causal sliding window mask (True = masked)."""
        if self.mask is not None and self.mask.shape[-1] >= seq_len:
            return self.mask[:, :, :seq_len, :seq_len].to(device)

        # Initialise everything as masked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        # For each token position i, unmask the valid window range
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1  
            mask[i, start:end] = False

        # Add dimensions for batch and heads for broadcasting
        self.mask = mask.unsqueeze(0).unsqueeze(1)
        return self.mask.to(device)


    def forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor = None,
    past_kv: tuple[torch.Tensor, torch.Tensor] = None,
    use_cache: bool = False,
    start_pos: int = 0
):
        """
        Forward pass for Sliding Window / Multi-Head Attention with RoPE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Attention mask
            past_kv (tuple[torch.Tensor, torch.Tensor], optional): Cached key and value
            use_cache (bool): Whether to return KV cache
            start_pos (int): Starting position for RoPE (rotary embeddings)

        Returns:
            torch.Tensor or tuple: Output tensor of shape (batch_size, seq_len, embed_dim)
                                Optionally returns (output, present_kv) if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # 1. Create sliding window mask if not provided
        if mask is None:
            mask = self._create_sliding_window_mask(seq_len, x.device)

        # 2. Project q, k, v
        q = self.gqa.wq(x)
        k = self.gqa.wk(x)
        v = self.gqa.wv(x)

        # 3. Reshape for multi-head
        q = q.view(batch_size, seq_len, self.gqa.num_heads, self.gqa.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.gqa.num_kv_heads, self.gqa.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.gqa.num_kv_heads, self.gqa.head_dim).transpose(1, 2)

        # 4. Repeat KV heads if needed
        k = self.gqa.repeat_kv(k, self.gqa.kv_repeat_factor)
        v = self.gqa.repeat_kv(v, self.gqa.kv_repeat_factor)

        # 5. Apply RoPE
        if hasattr(self.gqa, 'rope') and self.gqa.rope is not None:
            q, k = self.gqa.rope(q, k, start_pos)

        # 6. Handle past KV cache for incremental decoding
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # 7. Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (self.gqa.head_dim ** -0.5)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # 8. Compute attention weights and context
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.gqa.attn_dropout(attn_weights)
        context = attn_weights @ v

        # 9. Merge heads and output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.gqa.embed_dim)
        output = self.gqa.out_proj(context)
        output = self.gqa.resid_dropout(output)

        if use_cache:
            return output, present_kv
        else:
            return output

