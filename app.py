import gradio as gr
import torch
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.config.model_config import NanoGptConfig
from torch.serialization import add_safe_globals

add_safe_globals([NanoGptConfig])

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "out/enhanced_gpt_v1/best_model.pt"
TOKENIZER_PATH = "out/enhanced_gpt_v1/tinystories_tokenizer"

print("🚀 Loading Story Generation AI...")

# Load tokenizer
tokenizer = BpeTokenizer()
assert Path(f"{TOKENIZER_PATH}.vocab.json").exists(), f"Tokenizer not found at {TOKENIZER_PATH}"
tokenizer.load(TOKENIZER_PATH)

# Load model
assert Path(CHECKPOINT_PATH).exists(), f"Model checkpoint not found at {CHECKPOINT_PATH}"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

config = checkpoint['model_config']
model = NanoGptModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Model loaded: {config.attention_type.upper()} | Layers: {config.num_layers} | Params: {sum(p.numel() for p in model.parameters()):,}")

def generate_story(prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, num_generations=1):
    """Enhanced generation with multiple samples and quality scoring"""
    
    if not prompt or len(prompt.strip()) == 0:
        return "⚠️ Please provide a starting prompt to generate a story.", None, None
    
    try:
        start_time = time.time()
        
        # Encode prompt
        start_tokens = tokenizer.encode(prompt)
        if len(start_tokens) == 0:
            return "⚠️ Unable to tokenize prompt. Please try different text.", None, None
        
        start_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)
        
        # Generate multiple candidates
        candidates = []
        for i in range(num_generations):
            generated_tokens = model.generate(
                start_tokens_tensor,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty
            )
            
            generated_ids = generated_tokens[0].tolist()
            valid_ids = [tid for tid in generated_ids if tid in tokenizer.vocab]
            generated_text = tokenizer.decode(valid_ids)
            
            candidates.append(generated_text)
        
        # Select best candidate (for now, just return first)
        best_story = candidates[0]
        
        # Calculate generation stats
        generation_time = time.time() - start_time
        num_tokens = len(valid_ids)
        tokens_per_sec = num_tokens / generation_time
        
        # Create stats display
        stats = f"""
### 📊 Generation Statistics
- **Tokens Generated**: {num_tokens}
- **Generation Time**: {generation_time:.2f}s
- **Speed**: {tokens_per_sec:.1f} tokens/sec
- **Model**: {config.attention_type.upper()} ({config.num_layers} layers)
- **Temperature**: {temperature}
"""
        
        # Create info display
        info = f"""
### ❕ Model Information
- **Architecture**: Custom GPT with {config.attention_type.upper()} attention
- **Parameters**: {sum(p.numel() for p in model.parameters()):,}
- **Vocabulary Size**: {config.vocab_size}
- **Context Length**: {config.seq_len} tokens
- **Training Steps**: {checkpoint.get('step', 'N/A')}
"""
        
        return best_story, stats, info
        
    except Exception as e:
        return f"❌ An error occurred: {str(e)}\n\nPlease try adjusting the parameters or using a different prompt.", None, None


# Custom CSS for professional styling
custom_css = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.header-title {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
    text-align: center;
}

.header-subtitle {
    color: rgba(255,255,255,0.95) !important;
    font-size: 1.1rem !important;
    text-align: center;
    font-weight: 400;
}

/* Input Styling */
.prompt-box textarea {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

.prompt-box textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Button Styling */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Output Styling */
.output-box {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    background: #fafafa !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
}

/* Stats Card */
.stats-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}

/* Accordion Styling */
.accordion {
    border-radius: 10px !important;
    border: 1px solid #e0e0e0 !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem;
    color: #666;
    font-size: 0.9rem;
}
"""

# Build interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")) as demo:
    
    # Header
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">🎭 NanoGPT Story Generator</h1>
            <p class="header-subtitle">
                A custom-built GPT model with advanced attention mechanisms for creative storytelling
            </p>
        </div>
    """)
    
    # Main content
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Story Prompt")
            prompt_box = gr.Textbox(
                label="",
                placeholder="Enter your story beginning here... (e.g., 'Once upon a time, in a magical forest...')",
                lines=4,
                elem_classes="prompt-box"
            )
            
            with gr.Accordion("⚙️ Advanced Generation Settings", open=False, elem_classes="accordion"):
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=50, maximum=500, value=150, step=10,
                        label="Max Tokens",
                        info="Length of generated story"
                    )
                    temperature = gr.Slider(
                        minimum=0.5, maximum=1.5, value=0.85, step=0.05,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.7, maximum=1.0, value=0.92, step=0.02,
                        label="Top-p (Nucleus Sampling)",
                        info="Controls diversity of word choices"
                    )
                    top_k = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Top-k",
                        info="Limits vocabulary per step"
                    )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.15, step=0.05,
                    label="Repetition Penalty",
                    info="Reduces repetitive text"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Story",
                variant="primary",
                elem_classes="generate-btn",
                size="lg"
            )
            
            # Example prompts
            gr.Examples(
                examples=[
                    ["Once upon a time, there was a brave little mouse named"],
                    ["In a kingdom far away, a young princess discovered"],
                    ["The old wizard looked into his crystal ball and saw"],
                    ["On a sunny morning, two best friends decided to"],
                    ["Deep in the enchanted forest, a mysterious creature"],
                ],
                inputs=prompt_box,
                label="💡 Example Prompts"
            )
        
        # Right column - Output
        with gr.Column(scale=6):
            gr.Markdown("### 📖 Generated Story")
            output_box = gr.Textbox(
                label="",
                lines=18,
                interactive=False,
                elem_classes="output-box",
                show_copy_button=True
            )
            
            with gr.Row():
                with gr.Column():
                    stats_box = gr.Markdown("", elem_classes="stats-card")
                with gr.Column():
                    info_box = gr.Markdown("", elem_classes="stats-card")
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p><strong>NanoGPT</strong> - A from-scratch GPT implementation with custom attention mechanisms</p>
            <p>Built with PyTorch | Featuring RoPE, SwiGLU, RMSNorm, and advanced sampling techniques</p>
        </div>
    """)
    
    # Connect generate button
    generate_btn.click(
        fn=generate_story,
        inputs=[
            prompt_box,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty
        ],
        outputs=[output_box, stats_box, info_box],
        api_name="generate"
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌟 Launching Professional Story Generator Interface")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False, 
        show_error=True,
        favicon_path=None
    )