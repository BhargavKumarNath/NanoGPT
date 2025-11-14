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

        # 1. Token Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)

        # 2. Positional Encoding (RoPE)
        head_dim = config.embed_dim // config.num_heads
        rope = RotaryPositionalEmbeddings(head_dim, max_seq_len=config.seq_len)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList()

        # Linearly increasing DropPath rates (0 → config.dropout)
        drop_path_rates = [x.item() for x in torch.linspace(0, config.dropout, config.num_layers)]

        for i, drop_rate in enumerate(drop_path_rates):
            # Select Attention Type
            if config.attention_type == 'mha':
                attn = MultiHeadAttention(
                    config.embed_dim,
                    config.num_heads,
                    config.dropout,
                    config.bias,
                    rope=rope
                )
            elif config.attention_type == 'gqa':
                attn = GroupedQueryAttention(
                    config.embed_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.dropout,
                    config.bias
                )
            elif config.attention_type == 'swa':
                attn = SlidingWindowAttention(
                    config.embed_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.window_size,
                    config.dropout,
                    config.bias
                )
            else:
                raise ValueError(f"Unknown attention type: {config.attention_type}")

            # Feed Forward Network
            ffn = SwiGLUFeedForward(
                config.embed_dim,
                config.ffn_hidden_dim,
                config.dropout,
                config.bias
            )

            # Transformer Block
            block = TransformerBlock(
                embed_dim=config.embed_dim,
                attn=attn,
                ffn=ffn,
                norm_cls=RMSNorm,
                drop_path_rate=drop_rate,  
            )
            self.blocks.append(block)

        # 4. Final Layers
        self.final_norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (sharing)
        self.token_embeddings.weight = self.lm_head.weight

        # 5. Initialize Weights
        self.apply(self._init_weights)

    # Weight Initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward Pass
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv_cache: list = None, use_cache: bool = False):
        batch_size, seq_len = idx.shape
        
        x = self.token_embeddings(idx)
        
        present_kv_cache = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_kv_cache[i] if past_kv_cache is not None else None
            
            h = block.norm1(x)
            attn_output = block.attn(h, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                attn_output, present_kv = attn_output
                present_kv_cache.append(present_kv)
            x = x + block.drop_path1(attn_output)
            
            h = block.norm2(x)
            ffn_output = block.ffn(h)
            x = x + block.drop_path2(ffn_output)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        if use_cache:
            return logits, loss, present_kv_cache
        else:
            return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        """
        Autoregressively generate a sequence of tokens using Top-k and/or Top-p (Nucleus) sampling.
        """
        self.eval()
        kv_cache = None
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            
            logits, _, kv_cache = self(idx_cond, use_cache=True, past_kv_cache=kv_cache)
            
            logits = logits[:, -1, :] / temperature
            
            # Optional Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-p (Nucleus) sampling
            if top_p is not None:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first one that exceeds the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask of tokens to remove
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # Apply the mask by setting the probability of removed tokens to 0
                probs[indices_to_remove] = 0
                # Renormalize the probabilities
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample from the (potentially modified) distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train()
        return idx

    # Optimizer Configuration
    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        Configures the AdamW optimizer with proper weight decay settings.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num. decayed tensors: {len(decay_params)}, parameters: {num_decay_params:,}")
        print(f"Num. non-decayed tensors: {len(nodecay_params)}, parameters: {num_nodecay_params:,}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
