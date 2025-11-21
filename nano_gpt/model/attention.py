import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import RotaryPositionalEmbeddings

class GroupedQueryAttention(nn.Module):
    """
    A from scratch implementation of Grouped-Query Attention (GQA) 
    using PyTorch's optimized Scaled Dot Product Attention (SDPA).
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_kv_heads

        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)
        
        self.rope = rope

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_kv: tuple = None, use_cache: bool = False, start_pos: int = 0):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.rope:
            q, k = self.rope(q, k, start_pos)
        
        # KV Cache Management
        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present_kv = (k, v) if use_cache else None

        # Repeat KV heads for GQA logic
        if self.kv_repeat_factor > 1:
            k = k.repeat_interleave(self.kv_repeat_factor, dim=1)
            v = v.repeat_interleave(self.kv_repeat_factor, dim=1)

        # FLASH ATTENTION 2 IMPLEMENTATION        
        is_causal = (mask is None and seq_len > 1)
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask if not is_causal else None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if use_cache:
            return output, present_kv
        else:
            return output

class MultiHeadAttention(GroupedQueryAttention):
    """
    Standard Multi-Head Attention (MHA). 
    We implement it as a wrapper around GroupedQueryAttention where num_kv_heads == num_heads.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_kv_heads=num_heads, 
            dropout=dropout, 
            bias=bias, 
            rope=rope
        )

class HybridAttention(nn.Module):
    """
    A more powerful attention mechanism combining local (Sliding Window) and global attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_global_heads: int, window_size: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads > num_global_heads

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_global_heads = num_global_heads
        self.num_local_heads = num_heads - num_global_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = rope

    def _create_masks(self, seq_len: int, device: torch.device):
        # Global Mask (Standard Causal Mask)
        global_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        # Local Mask (Sliding Window Causal Mask)
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1
            local_mask[i, start:end] = False
        
        return global_mask.to(device), local_mask.to(device)

    def forward(self, x: torch.Tensor, past_kv: tuple = None, use_cache: bool = False, start_pos: int = 0, mask=None):        
        batch_size, seq_len, _ = x.shape
        
        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            q, k = self.rope(q, k, start_pos)
            
        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present_kv = (k, v) if use_cache else None
        kv_len = k.shape[-2]

        # The Core Logic: Split Heads
        # Split Q, K, V into global and local groups
        q_global, q_local = q.split([self.num_global_heads, self.num_local_heads], dim=1)
        k_global, k_local = k.split([self.num_global_heads, self.num_local_heads], dim=1)
        v_global, v_local = v.split([self.num_global_heads, self.num_local_heads], dim=1)

        # Create Masks
        global_mask, local_mask = self._create_masks(kv_len, x.device)
        
        global_mask = global_mask[-seq_len:]
        local_mask = local_mask[-seq_len:]
        
        # Compute Attention for Each Group
        # Global Attention
        attn_scores_global = (q_global @ k_global.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_scores_global = attn_scores_global.masked_fill(global_mask, float('-inf'))
        attn_weights_global = F.softmax(attn_scores_global, dim=-1)
        attn_weights_global = self.attn_dropout(attn_weights_global)
        context_global = attn_weights_global @ v_global

        # Local Attention
        attn_scores_local = (q_local @ k_local.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_scores_local = attn_scores_local.masked_fill(local_mask, float('-inf'))
        attn_weights_local = F.softmax(attn_scores_local, dim=-1)
        attn_weights_local = self.attn_dropout(attn_weights_local)
        context_local = attn_weights_local @ v_local

        # Concatenate Results
        context = torch.cat((context_global, context_local), dim=1)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        if use_cache:
            return output, present_kv
        else:
            return output

class SlidingWindowAttention(nn.Module):
    """A from scratch implementation of Sliding Window Attention with GQA"""
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, window_size: int, dropout: float = 0.1, bias: bool = True, rope: RotaryPositionalEmbeddings = None):
        super().__init__()
        self.gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout, bias, rope=rope)
        self.window_size = window_size
        self.mask = None

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device):
        if self.mask is not None and self.mask.shape[-1] >= seq_len:
            return self.mask[:, :, :seq_len, :seq_len].to(device)

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = i + 1  
            mask[i, start:end] = False

        self.mask = mask.unsqueeze(0).unsqueeze(1)
        return self.mask.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_kv: tuple = None, use_cache: bool = False, start_pos: int = 0):
        batch_size, seq_len, _ = x.shape
        if mask is None:
            mask = self._create_sliding_window_mask(seq_len, x.device)

        # Use the GQA instance to calculate attention
        return self.gqa(x, mask=mask, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos)