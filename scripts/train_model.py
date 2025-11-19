import torch
from torch.utils.data import DataLoader
import os
import sys
import time
import json
import math
from pathlib import Path
import wandb 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.data.dataset import StreamingTextDataset

experiment_name = "enhanced_gpt_v1"
print(f"\n{'='*60}\n🚀 Enhanced Training: {experiment_name}\n{'='*60}")

# Output folder
out_dir = Path("out") / experiment_name
out_dir.mkdir(parents=True, exist_ok=True)


# Training Config
max_steps = 100000  
learning_rate = 3e-4  
min_lr = 3e-5  
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_steps = 2000  

# Batch Config
micro_batch_size = 4  
gradient_accumulation_steps = 16 
effective_batch_size = micro_batch_size * gradient_accumulation_steps
print(f"📊 Effective Batch Size: {effective_batch_size}")

# Logging & Checkpointing
eval_interval = 1000
log_interval = 100
save_checkpoint_interval = 5000
num_eval_batches = 50

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
pt_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"💻 Device: {device} | Precision: {pt_dtype}")


def get_lr(step):
    """Improved LR schedule: Linear warmup → Cosine decay with min_lr"""
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)
    elif step > max_steps:
        return min_lr
    else:
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

# Tokenizer
tokenizer_model_prefix = out_dir / "tinystories_tokenizer"
corpus_path = "data_corpus/tinystories.txt"
tokenizer = BpeTokenizer()

# Increase vocab size for better expressiveness
TARGET_VOCAB_SIZE = 2048  

if not Path(f"{tokenizer_model_prefix}.vocab.json").exists():
    print(f"🔤 Training tokenizer with vocab_size={TARGET_VOCAB_SIZE}...")
    tokenizer.train(corpus_path, vocab_size=TARGET_VOCAB_SIZE, special_tokens=["<|endoftext|>"])
    tokenizer.save(str(tokenizer_model_prefix))
else:
    print("🔤 Loading existing tokenizer...")
    tokenizer.load(str(tokenizer_model_prefix))

vocab_size = max(tokenizer.vocab.keys()) + 1
print(f"✅ Tokenizer vocab size: {vocab_size}")

# Dataset
seq_len = 384  # Increased for better context

# Create train/val split
train_dataset = StreamingTextDataset(tokenizer, corpus_path, seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, num_workers=2, pin_memory=True)

# For validation, use a separate file or subset
val_corpus_path = "data_corpus/tinystories_val.txt"
if Path(val_corpus_path).exists():
    val_dataset = StreamingTextDataset(tokenizer, val_corpus_path, seq_len)
    val_dataloader = DataLoader(val_dataset, batch_size=micro_batch_size, num_workers=2, pin_memory=True)
    has_validation = True
    print(f"✅ Validation dataset loaded")
else:
    print(f"⚠️  No validation data found. Skipping validation.")
    has_validation = False

# Model Config
model_config = NanoGptConfig(
    vocab_size=vocab_size,
    embed_dim=512,  
    num_layers=8,   
    num_heads=8,
    seq_len=seq_len,
    dropout=0.1,    
    attention_type='mha',  
    ffn_hidden_dim=2048,
)

model = NanoGptModel(model_config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Model initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2))
scaler = torch.amp.GradScaler("cuda", enabled=(pt_dtype == torch.float16))

@torch.no_grad()
def validate():
    """Run validation and return average loss"""
    if not has_validation:
        return None
    
    model.eval()
    total_loss = 0.0
    val_iter = iter(val_dataloader)
    
    for _ in range(num_eval_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, dtype=pt_dtype):
            logits, loss = model(x, y)
        total_loss += loss.item()
    
    model.train()
    avg_loss = total_loss / num_eval_batches
    return avg_loss

@torch.no_grad()
def generate_sample(prompt="Once upon a time", max_tokens=100, temperature=0.8):
    """Generate a sample for monitoring quality"""
    model.eval()
    
    start_tokens = tokenizer.encode(prompt)
    start_tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)
    
    generated = model.generate(
        start_tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2
    )
    
    generated_ids = generated[0].tolist()
    valid_ids = [tid for tid in generated_ids if tid in tokenizer.vocab]
    text = tokenizer.decode(valid_ids)
    
    model.train()
    return text

# Train loop
data_iter = iter(train_dataloader)
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"\n{'='*60}\n🎯 Starting Training\n{'='*60}\n")

for step in range(max_steps):
    step_start = time.time()
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Validation and checkpoint    
    if step > 0 and step % eval_interval == 0:
        val_loss = validate()
        
        if val_loss is not None:
            val_losses.append({'step': step, 'val_loss': val_loss})
            print(f"\n{'='*60}")
            print(f"📊 Step {step} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = out_dir / "best_model.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "step": step,
                    "val_loss": val_loss
                }, ckpt_path)
                print(f"💾 New best model saved! (val_loss={val_loss:.4f})")
        
        # Generate sample
        sample = generate_sample("Once upon a time, there was a little")
        print(f"\n📝 Sample Generation:\n{sample}\n")
        print(f"{'='*60}\n")
    
    # Regular checkpoint
    if step > 0 and step % save_checkpoint_interval == 0:
        ckpt_path = out_dir / f"ckpt_step_{step}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "step": step
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path.name}")
    
    # Training step
    
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    
    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=device, dtype=pt_dtype):
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
    
    # Gradient clipping & optimizer step
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    
    # Logging    
    if step % log_interval == 0:
        step_time = (time.time() - step_start) * 1000
        train_loss = accumulated_loss * gradient_accumulation_steps
        train_losses.append({'step': step, 'train_loss': train_loss})
        
        tokens_per_sec = (micro_batch_size * gradient_accumulation_steps * seq_len) / (step_time / 1000)
        
        print(
            f"Step {step:5d}/{max_steps} | "
            f"Loss: {train_loss:.4f} | "
            f"LR: {lr:.6f} | "
            f"GradNorm: {grad_norm:.2f} | "
            f"{step_time:.0f}ms | "
            f"{tokens_per_sec/1000:.1f}k tok/s"
        )

# Save final model and training history
print(f"\n{'='*60}\n✅ Training Complete!\n{'='*60}")

final_path = out_dir / "final_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": model_config,
    "step": max_steps
}, final_path)
print(f"💾 Final model saved: {final_path}")

# Save training history
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "config": {
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "effective_batch_size": effective_batch_size,
        "model_params": total_params
    }
}
with open(out_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n🎉 Training artifacts saved to: {out_dir}")