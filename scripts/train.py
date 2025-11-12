import torch
from torch.utils.data import DataLoader
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.data.dataset import StreamingTextDataset

# Training config
max_steps = 200
learning_rate = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Batching config
micro_batch_size = 4
gradient_accumulation_steps = 4
effective_batch_size = micro_batch_size * gradient_accumulation_steps
print(f"Effective batch size: {effective_batch_size}")

# Curriculum Learning Config
initial_seq_len = 16
final_seq_len = 32
seq_len_warmup_steps = 50 
print(f"Using progressive sequence length: {initial_seq_len} for {seq_len_warmup_steps} steps, then {final_seq_len}.")


# System config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True 
print(f"Using device: {device}")

# Mixed Precision Setup
pt_dtype = torch.float16 if device == 'cuda' else torch.float32
ctx = torch.amp.autocast(device_type=device, dtype=pt_dtype)
scaler = torch.amp.GradScaler(device, enabled=(pt_dtype == torch.float16))
print(f"Using mixed precision with dtype: {pt_dtype}")

# Setup
# Tokenizer
tokenizer_prefix = "bpe_tokenizer"
assert os.path.exists(f"{tokenizer_prefix}.vocab.json"), "Tokenizer model not found."
tokenizer = BpeTokenizer()
tokenizer.load(tokenizer_prefix)
vocab_size = max(tokenizer.vocab.keys()) + 1

# Dataset and DataLoader
corpus_path = "data_corpus/sample.txt"
current_seq_len = initial_seq_len
dataset = StreamingTextDataset(tokenizer, corpus_path, current_seq_len)
dataloader = DataLoader(dataset, batch_size=micro_batch_size)

# Model
# Initialise the model with the FINAL sequence length so RoPE cache is large enough
model_config = NanoGptConfig(
    vocab_size=vocab_size,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    seq_len=final_seq_len, 
    attention_type='mha'
)
model = NanoGptModel(model_config)
model.to(device)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# Optimizer
optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2))

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=learning_rate/10)

# Training Loop
step = 0
running_loss = 0.0
data_iter = iter(dataloader)
start_time = time.time()

print("\n Starting Training (with Progressive Sequence Length) ")
while step < max_steps:
    
    # Sequence Length Transition Logic
    if step == seq_len_warmup_steps:
        print("\n" + "="*50)
        print(f"Warmup complete. Switching to sequence length: {final_seq_len}")
        print("="*50 + "\n")
        current_seq_len = final_seq_len
        # Recreate dataset and dataloader with the new sequence length
        dataset = StreamingTextDataset(tokenizer, corpus_path, current_seq_len)
        dataloader = DataLoader(dataset, batch_size=micro_batch_size)
        data_iter = iter(dataloader) # Reset the iterator
    
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    
    scheduler.step()
    
    running_loss += loss.item() * gradient_accumulation_steps
    if (step + 1) % 10 == 0:
        end_time = time.time()
        avg_loss = running_loss / 10
        steps_per_sec = 10 / (end_time - start_time)
        
        # Log the current sequence length being used
        print(f"Step {step+1:4d}/{max_steps} | SeqLen: {current_seq_len:2d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Steps/sec: {steps_per_sec:.2f}")
        running_loss = 0.0
        start_time = time.time()

    step += 1

print("\n--- Training Complete ---")