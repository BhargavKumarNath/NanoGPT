import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """
    A from scratch implementation of the SwiGLU Feed Forward Network. This is a modern replacement for standard FFNs with ReLU or GELU activations.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        # If hidden_dim is not provided use the LLaMA paper's recommendation:
        # a multiple of 256 and roughly 2/3 of the size of a standard FFN's hidden layer.
        # Standard FFN hidden is 4 * embed_dim. So, (4 * embed_dim) * (2/3)
        if hidden_dim is None:
            hidden_dim = int(2/3 * 4 * embed_dim)
            # Make it a multiple of 256 for hardware efficiency
            hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        
        # The SwiGLU formulation uses three liner layers
        # W1 and W3 are the parallel layers for the gate and value
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        # w2 is the output projection
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLY FFN.

        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
        torch.Tensor: Output tensor of the same shape as input.
        """

        # Gated liner unit calculation
        gate = F.silu(self.w1(x))
        value = self.w3(x)

        # Element wise multiplication
        gated_value = gate * value

        # Final output projection
        output = self.w2(gated_value)
        output = self.dropout(output)

        return output
    