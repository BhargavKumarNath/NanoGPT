import torch
from torch.utils.data import DataLoader
import os
import sys
import time
import json
import math 
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.data.dataset import StreamingTextDataset

# Experiment Config
experiment_name = "hybrid_attention"

print(f"\n--- Running Experiment: {experiment_name} ---")

# Output folder
out_dir = os.path.join("out", experiment_name)
os.makedirs(out_dir, exist_ok=True)

# Logging
eval_interval = 500  
log_interval = 50   

# Training Config
max_steps = 30000
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_steps = 1000 

# Batch Config
micro_batch_size = 8  
gradient_accumulation_steps = 8
effective_batch_size = micro_batch_size * gradient_accumulation_steps

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
pt_dtype = torch.float16 if device == "cuda" else torch.float32

def get_lr(step):
    """Learning rate schedule with warmup and cosine decay"""
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

# Tokenizer
tokenizer_model_prefix = os.path.join(out_dir, "tinystories_tokenizer")
corpus_path = "data_corpus/tinystories.txt"
tokenizer = BpeTokenizer()

if not os.path.exists(f"{tokenizer_model_prefix}.vocab.json"):
    print("Training a new tokenizer on TinyStories...")
    tokenizer.train(corpus_path, vocab_size=512, special_tokens=["<|endoftext|>"])
    tokenizer.save(tokenizer_model_prefix)
else:
    print("Loading existing tokenizer...")
    tokenizer.load(tokenizer_model_prefix)

vocab_size = max(tokenizer.vocab.keys()) + 1
print(f"Tokenizer vocab size: {vocab_size}")

# Dataset
seq_len = 256
dataset = StreamingTextDataset(tokenizer, corpus_path, seq_len)
dataloader = DataLoader(dataset, batch_size=micro_batch_size)

# EXPERIMENT-BASED MODEL CONFIG
attention_type = "mha"
num_kv_heads = None
window_size = None
num_global_heads = None 

if experiment_name == "mha_baseline":
    attention_type = "mha"

elif experiment_name == "gqa":
    attention_type = "gqa"
    num_kv_heads = 2

elif experiment_name == "swa_long_context":
    attention_type = "swa"
    num_kv_heads = 2      
    window_size = 64      
    seq_len = 256         

elif experiment_name == "hybrid_attention":
    attention_type = "hybrid"   
    num_kv_heads = None         # HybridAttention uses full K,V per head
    window_size = 64            # Local window size
    num_global_heads = 2        # Number of global heads
    seq_len = 256

else:
    raise ValueError(f"Unknown experiment: {experiment_name}")

print(f"Experiment: {experiment_name}")
print(f" → attention_type = {attention_type}")
print(f" → num_kv_heads = {num_kv_heads}")
print(f" → window_size = {window_size}")
print(f" → num_global_heads = {num_global_heads}") 
print(f" → seq_len = {seq_len}")

model_config = NanoGptConfig(
    vocab_size=vocab_size,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    seq_len=seq_len,
    dropout=0.2,
    attention_type=attention_type,
    num_kv_heads=num_kv_heads,
    window_size=window_size,
    num_global_heads=num_global_heads, 
)

model = NanoGptModel(model_config).to(device)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# Optimizer + Grad Scaler
optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2))
scaler = torch.amp.GradScaler(device="cuda", enabled=(pt_dtype == torch.float16))

# Training Loop
step = 0
data_iter = iter(dataloader)
print(f"\nStarting training for {max_steps} steps...\n")
for step in range(max_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    start_time = time.time()
    
    # Evaluation + Checkpoint
    if step > 0 and step % eval_interval == 0:
        print(f"\nSaving checkpoint at step {step}")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "step": step
        }
        torch.save(ckpt, os.path.join(out_dir, f"ckpt_step_{step}.pt"))
        # Generate a sample
        model.eval()
        prompt = "Once upon"
        start_tokens = tokenizer.encode(prompt)
        start_tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)

        generated = model.generate(
            start_tokens,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9
        )
        print("Sample Generation:")
        generated_ids = generated[0].tolist()
        valid_ids = [tid for tid in generated_ids if tid in tokenizer.vocab]
        try:
            print(tokenizer.decode(valid_ids))
        except Exception as e:
            print(f"Decode error: {e}")
            print(f"Generated token IDs: {generated_ids[:20]}...")  
        print("-" * 50)
        model.train()
    
    # Gradient Accumulation Loop
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, dtype=pt_dtype):
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    
    # Logging
    if step % log_interval == 0:
        ms = (time.time() - start_time) * 1000
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item()*gradient_accumulation_steps:.4f} | LR: {lr:.6f} | {ms:.2f}ms")

print("\n--- Training Complete ---")
torch.save({"model_state_dict": model.state_dict(), "model_config": model_config},
           os.path.join(out_dir, "final_model.pt"))
print("Final model saved.")