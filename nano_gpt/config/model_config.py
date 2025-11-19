from dataclasses import dataclass

@dataclass
class NanoGptConfig:
    """Configuration class for the NanoGPT model"""
    vocab_size: int = 300
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 64
    dropout: float = 0.1
    bias: bool = True

    # Attention
    attention_type: str = 'mha'
    num_kv_heads: int = 2
    window_size: int = 32
    num_global_heads: int = None
    
    # FFN Specific 
    ffn_hidden_dim: int = None
    