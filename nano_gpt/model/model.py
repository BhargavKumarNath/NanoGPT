import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from .block import TransformerBlock
from .attention import MultiHeadAttention, GroupedQueryAttention, SlidingWindowAttention
from .feed_forward import SwiGLUFeedForward
from .normalization import RMSNorm
from .positional_encoding import RotaryPositionalEmbeddings
from ..config.model_config import NanoGptConfig

class NanoGptModel(nn.Module):
    """Full NanoGPT language model"""

    def __init__(self, config: NanoGptConfig):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None, "vocab_size must be set"
        assert config.seq_len is not None, "seq_len must be set"

        # Embedding layers
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional Encodin g(RoPE)
        # RoPE is applied inside attention module
        head_dim = config.embed_dim // config.num_heads
        rope = RotaryPositionalEmbeddings(head_dim, max_seq_len=config.seq_len)

        # Transformer Blocks
        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            # Choose attention mechanism based on config
            if config.attention_type == 'mha':
                attn = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout, config.bias, rope=rope)
            elif config.attention_type == 'gqa':
                attn = GroupedQueryAttention(config.embed_dim, config.num_heads, config.num_kv_heads, config.dropout, config.bias)
                # RoPE would need to be integrated into GQA as well for a full implementation
            elif config.attention_type == 'swa':
                attn = SlidingWindowAttention(config.embed_dim, config.num_heads, config.num_kv_heads, config.window_size, config.dropout, config.bias)
            else:
                raise ValueError(f"Unknown attention type: {config.attention_type}")
            
            ffn = SwiGLUFeedForward(config.embed_dim, config.ffn_hidden_dim, config.dropout, config.bias)
            
            block = TransformerBlock(
                embed_dim=config.embed_dim,
                attn=attn,
                ffn=ffn,
                norm_cls=RMSNorm
            )
            self.blocks.append(block)
        
        # Final layers
        self.final_norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight typing: share weights between embedding and output layers
        self.token_embeddings.weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = idx.shape

        # 1. get token embeddings
        x = self.token_embeddings(idx)

        # 2. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final Normalization
        x = self.final_norm(x)

        # 4. Language model head
        logits = self.lm_head(x)

        # 5. Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy
            # Logits: (B, T, V) -> (B*T, V)
            # Targets: (B, T) -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


        return logits, loss
