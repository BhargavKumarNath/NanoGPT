import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from torch.utils.checkpoint import checkpoint 
from .block import TransformerBlock
from .attention import MultiHeadAttention, GroupedQueryAttention, SlidingWindowAttention, HybridAttention
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
        
        self.gradient_checkpointing = False

        # Token Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional Encoding (RoPE)
        head_dim = config.embed_dim // config.num_heads
        rope = RotaryPositionalEmbeddings(head_dim, max_seq_len=config.seq_len)

        # Transformer Blocks
        self.blocks = nn.ModuleList()

        # Linearly increasing DropPath rates (0 -> config.dropout)
        drop_path_rates = [x.item() for x in torch.linspace(0, config.dropout, config.num_layers)]

        for i, drop_rate in enumerate(drop_path_rates):
            # Select Attention Type
            if config.attention_type == 'mha':
                attn = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout, config.bias, rope=rope)
            elif config.attention_type == 'gqa':
                attn = GroupedQueryAttention(config.embed_dim, config.num_heads, config.num_kv_heads, config.dropout, config.bias, rope=rope)
            elif config.attention_type == 'swa':
                attn = SlidingWindowAttention(config.embed_dim, config.num_heads, config.num_kv_heads, config.window_size, config.dropout, config.bias, rope=rope)
            elif config.attention_type == 'hybrid':
                attn = HybridAttention(config.embed_dim, config.num_heads, config.num_global_heads, config.window_size, config.dropout, config.bias, rope=rope)
            else:
                raise ValueError(f"Unknown attention type: {config.attention_type}")

            # Feed Forward Network
            ffn = SwiGLUFeedForward(config.embed_dim, config.ffn_hidden_dim, config.dropout, config.bias)

            # Transformer Block
            block = TransformerBlock(
                embed_dim=config.embed_dim,
                attn=attn,
                ffn=ffn,
                norm_cls=RMSNorm,
                drop_path_rate=drop_rate,  
            )
            self.blocks.append(block)

        # Final Layers
        self.final_norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (sharing)
        self.token_embeddings.weight = self.lm_head.weight

        # Initialize Weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, value: bool):
        """
        Toggles gradient checkpointing to save VRAM at the cost of compute.
        """
        self.gradient_checkpointing = value

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv_cache: list = None, use_cache: bool = False, start_pos: int = 0):
        batch_size, seq_len = idx.shape
        x = self.token_embeddings(idx)
        present_kv_cache = [] if use_cache else None

        # CREATE CAUSAL MASK
        if use_cache and past_kv_cache is not None and len(past_kv_cache) > 0:
            past_len = past_kv_cache[0][0].size(-2)
            full_len = past_len + seq_len
            mask = torch.triu(torch.ones(seq_len, full_len, device=idx.device, dtype=torch.bool), diagonal=past_len + 1)
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool), diagonal=1)
        
        mask = mask.unsqueeze(0).unsqueeze(0) 

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_cache[i] if past_kv_cache is not None else None

            # GRADIENT CHECKPOINTING LOGIC
            if self.gradient_checkpointing and self.training and not use_cache:
                x = checkpoint(block, x, mask, past_kv, use_cache, start_pos, use_reentrant=False)
            else:
                block_result = block(x, mask=mask, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos)
            
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
        self.eval()
        kv_cache = None

        for step in range(max_new_tokens):
            if step == 0:
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
                batch_size = idx.shape[0]
                for batch_idx in range(batch_size):
                    context_length = min(self.config.seq_len, idx.shape[1])
                    context_tokens = idx[batch_idx, -context_length:]
                    unique_tokens = torch.unique(context_tokens)
                    
                    for token_id in unique_tokens:
                        if logits[batch_idx, token_id] > 0:
                            logits[batch_idx, token_id] /= repetition_penalty
                        else:
                            logits[batch_idx, token_id] *= repetition_penalty
            
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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