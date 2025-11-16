import torch
import torch.nn as nn
from typing import Type
from .regularization import DropPath 

class TransformerBlock(nn.Module):
    """
    A single block of transformer model, using pre-normalisation
    """
    def __init__(self, embed_dim: int, attn: nn.Module, ffn: nn.Module, norm_cls: Type[nn.Module], drop_path_rate: float = 0.0):
        """
        Args:
            embed_dim (int): The embedding dimension.
            attn (nn.Module): An instantiated attention module.
            ffn (nn.Module): An instantiated feed-forward network module.
            norm_cls (Type[nn.Module]): The class for the normalization layer (e.g., RMSNorm)
        """
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = norm_cls(embed_dim)
        self.norm2 = norm_cls(embed_dim)

        # DropPath layers
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_kv=None, use_cache: bool = False, start_pos: int = 0):
        """
        Forward pass for the Transformer Block with mask support.
        
        Architecture: Pre-Norm (Norm -> Attention -> Residual -> Norm -> FFN -> Residual)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Causal attention mask
            past_kv (tuple, optional): Cached key and value tensors
            use_cache (bool): Whether to return KV cache
            start_pos (int): Starting position for RoPE
        
        Returns:
            torch.Tensor or tuple: Output tensor, optionally with present_kv if use_cache=True
        """
        # First Sub layer: Attention with mask
        attn_result = self.attn(
            self.norm1(x),
            mask=mask,
            past_kv=past_kv,
            use_cache=use_cache,
            start_pos=start_pos
        )
        
        # Handle KV cache return
        if use_cache:
            attn_output, present_kv = attn_result
        else:
            attn_output = attn_result
            present_kv = None
        
        x = x + self.drop_path1(attn_output)

        # Second Sub layer: Feed forward Network
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.drop_path2(ffn_output)

        if use_cache:
            return x, present_kv
        else:
            return x
