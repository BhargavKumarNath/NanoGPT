import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    """A from scratch implementation of Rotary Positional Embeddings (RoPE).
    This module pre-computes the sine and cosine frequencies and applies them to the query and key tensors.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre compute the theta values and sin/cos tables
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Register the inverse frequencies
        self.register_buffer("cached_inv_freq", self.inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-computes and caches the sine and cosine values."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.cached_inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.cached_inv_freq)

        # Shape: (seq_len, dim//2) -> (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotary embeddings to the input tensor.

        Args:
            x: (batch, heads, seq_len, head_dim)
            cos: pre-sliced cos table for sequence positions
            sin: pre-sliced sin table for sequence positions
        """
        # Reshape x into pairs: (..., d) -> (..., d/2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

        # Reshape sin/cos for broadcasting
        cos = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
        sin = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]

        # Apply rotation
        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        x_rotated = torch.stack([x_out1, x_out2], dim=-1).flatten(start_dim=-2)
        return x_rotated.type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int):
        """
        Applies RoPE to query and key tensors.

        Args:
            q: (batch, heads, seq_len, head_dim)
            k: (batch, heads, seq_len, head_dim)
            start_pos: starting position for rotary embedding
        """
        seq_len = q.shape[2]

        # Ensure cache is large enough
        if start_pos + seq_len > self.max_seq_len:
            self._build_cache(start_pos + seq_len)

        # Slice precomputed positional encodings
        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]

        # Apply rotary embeddings
        q_rotated = self._apply_rotary_emb(q, cos, sin)
        k_rotated = self._apply_rotary_emb(k, cos, sin)

        return q_rotated, k_rotated
