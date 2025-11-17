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
                    config.bias,
                    rope=rope
                )
            elif config.attention_type == 'swa':
                attn = SlidingWindowAttention(
                    config.embed_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.window_size,
                    config.dropout,
                    config.bias,
                    rope=rope
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

    # Weight Initialisation
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward Pass
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv_cache: list = None, use_cache: bool = False, start_pos: int = 0):
        batch_size, seq_len = idx.shape
        x = self.token_embeddings(idx)
        present_kv_cache = [] if use_cache else None

        # CREATE CAUSAL MASK
        if use_cache and past_kv_cache is not None and len(past_kv_cache) > 0:
            # During generation with KV cache
            past_len = past_kv_cache[0][0].size(-2)
            full_len = past_len + seq_len
            # Mask shape: (seq_len, full_len)
            mask = torch.triu(torch.ones(seq_len, full_len, device=idx.device, dtype=torch.bool), diagonal=past_len + 1)
        else:
            # Normal forward pass: standard causal mask
            # Mask shape: (seq_len, seq_len)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool), diagonal=1)
        
        # Add batch and head dimensions for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len or full_len)

        # Pass through transformer blocks WITH MASK
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_cache[i] if past_kv_cache is not None else None
            
            # Call block with all parameters
            block_result = block(
                x, 
                mask=mask, 
                past_kv=past_kv, 
                use_cache=use_cache, 
                start_pos=start_pos
            )
            
            # Handle cache returns
            if use_cache:
                x, present_kv = block_result
                present_kv_cache.append(present_kv)
            else:
                x = block_result
        
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        if use_cache:
            return logits, loss, present_kv_cache
        else:
            return logits, loss



    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0
    ):
        """
        Autoregressively generate a sequence of tokens using Top-k and/or Top-p sampling.
        Tracks start_pos for RoPE correctly.
        """
        self.eval()
        kv_cache = None

        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, -1:]
                start_pos = idx.size(1) - 1

            logits, _, kv_cache = self(
                idx_cond, use_cache=True, past_kv_cache=kv_cache, start_pos=start_pos
            )

            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                # Create a view of logits for the current batch item
                for i in range(idx.shape[0]):
                    # Find unique tokens in the context
                    unique_tokens = torch.unique(idx[i, -self.config.seq_len:])
                    # Apply penalty: for positive logits, divide; for negative, multiply
                    logits[i, unique_tokens] = torch.where(
                        logits[i, unique_tokens] > 0,
                        logits[i, unique_tokens] / repetition_penalty,
                        logits[i, unique_tokens] * repetition_penalty
                    )
            
            logits = logits / temperature

            # Apply Top-K filtering if specified
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Apply Top-P (nucleus) filtering if specified
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
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
