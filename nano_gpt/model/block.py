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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Teansformer Block.
        
        Architecture: Pre-Norm (Norm -> Attention -> Residual -> Norm -> FFN -> Residual)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # First Sub layer: Atrention
        attn_output = self.attn(self.norm1(x))
        x = x + self.drop_path1(attn_output)

        # Second Sub layer: Feed forward Network
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.drop_path2(ffn_output)

        return x