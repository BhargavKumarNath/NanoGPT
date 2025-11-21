import torch
import numpy as np
import os
import sys
import time
import json
import math
from pathlib import Path
import wandb
import tiktoken
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel

# Config
BATCH_SIZE = 8
ACCUM_STEPS = 32
SEQ_LEN = 512
LEARNING_RATE = 5e-4
MAX_ITERS = 10000
WARMUP_ITERS = 1000
DEVICE = 'cuda'
DTYPE = torch.bfloat16

experiment_name = "cosmopedia_pro_v1"
out_dir = Path("out")/experiment_name
out_dir.mkdir(parents=True, exist_ok=True)

# Data Loader
data_dir = 'data_corpus'
def get_batch(split):
    # Load memory mapped file
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Random offsets
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+SEQ_LEN]).astype(np.int64)) for i in ix])
    
    if DEVICE == 'cuda':
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    return x, y

# INITIALIZATION
print(f"Initializing {experiment_name} on {torch.cuda.get_device_name()}...")

# Load meta to get vocab size
with open(os.path.join(data_dir, 'meta.json'), 'r') as f:
    meta = json.load(f)
    vocab_size = meta['vocab_size']

config = NanoGptConfig(
    vocab_size=vocab_size,
    embed_dim=768,     
    num_layers=12,      
    num_heads=12,       
    seq_len=SEQ_LEN,
    dropout=0.0,        
    attention_type='gqa',
    num_kv_heads=4,     
    ffn_hidden_dim=int(768 * 8/3)
)

model = NanoGptModel(config)
model.to(DEVICE)

# COMPILE MODEL
print("Compiling model (this takes a minute)...")
model = torch.compile(model) 

optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=LEARNING_RATE, betas=(0.9, 0.95))

# TRAINING LOOP
scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16)) 
X, Y = get_batch('train')
t0 = time.time()

print("Training Started...")

for iter_num in range(MAX_ITERS):
    lr = LEARNING_RATE 
    
    # Gradient Accumulation
    for micro_step in range(ACCUM_STEPS):
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
            logits, loss = model(X, Y)
            loss = loss / ACCUM_STEPS 
        
        # Immediate backward
        scaler.scale(loss).backward()
        
        # Async prefetch next batch
        X, Y = get_batch('train')

    # Step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    if iter_num % 10 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = (BATCH_SIZE * ACCUM_STEPS * SEQ_LEN) / dt
        loss_f = loss.item() * ACCUM_STEPS
        print(f"Step {iter_num} | Loss: {loss_f:.4f} | Speed: {tokens_per_sec:.0f} tok/s")

    # Save
    if iter_num > 0 and iter_num % 500 == 0:
        ckpt_path = out_dir / f"ckpt_{iter_num}.pt"
        torch.save(model.state_dict(), ckpt_path) # Save weights
        print(f"Saved checkpoint: {ckpt_path}")
