import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    A from scratch implementation of Root Mean Square Normalization
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # The gain parameter is a learnable scale factor
        self.gain = nn.Parameter(torch.ones(embed_dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the root mean square and normalizes the input tensor. Using rsqrt (reciprocal square root) is numerically stable and efficient
        """
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + self.eps)
        return x * rsqrt
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm

        Args:
            x (torch.Tensor): Input tensor. Normalization is applied on the last dimension.

        Returns:
            torch.Tensor: Normalized output tensor of the same shape.
        """
        # Normalized and then apply the learnable gain
        normalized_x = self._norm(x.float()).type_as(x)
        return self.gain * normalized_x

