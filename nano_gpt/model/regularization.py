import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Stochastically drop a tensor. This is the core function for DropPath.
    When dropping, the entire tensor is zeroed out. The remaining tensors are scaled.
    """
    if drop_prob == 0. or not training:
        return x
    
    # Calculate the survival probability
    keep_prob = 1 - drop_prob
    
    # Create a random mask for each item in the batch
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    
    # Scale the output by the survival probability
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath layer.
    
    Randomly zeros out entire samples (in a batch) during training.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)