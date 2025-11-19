import gradio as gr
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer

from nano_gpt.config.model_config import NanoGptConfig
from torch.serialization import add_safe_globals
add_safe_globals([NanoGptConfig])

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = "out/hybrid_attention/final_model.pt"
TOKENIZER_PATH = "out/hybrid_attention/tinystories_tokenizer"

# Model & Tokenizer Loading
print("Loading model and tokenizer...")

tokenizer = BpeTokenizer()
assert os.path.exists(TOKENIZER_PATH + ".vocab.json"), f"Tokenizer not found at {TOKENIZER_PATH}"
tokenizer.load(TOKENIZER_PATH)

assert os.path.exists(CHECKPOINT_PATH), f"Model checkpoint not found at {CHECKPOINT_PATH}"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

config = checkpoint['model_config']
model = NanoGptModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Model and tokenizer loaded successfully.")
print(f"Model: {config.attention_type}, Layers: {config.num_layers}, Heads: {config.num_heads}")

def generate_story(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    if not prompt:
        return "Please provide a starting prompt."
        
    try:
        start_tokens = tokenizer.encode(prompt)
        start_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)
        
        generated_tokens = model.generate(
            start_tokens_tensor,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty
        )
        
        generated_ids = generated_tokens[0].tolist()
        valid_ids = [tid for tid in generated_ids if tid in tokenizer.vocab]
        generated_text = tokenizer.decode(valid_ids)
        
        return generated_text
        
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Gradio Interface
print("Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Nano-GPT: Interactive Story Generator (GQA Model)")
    gr.Markdown("This demo uses a from-scratch GPT model with Grouped-Query Attention trained on TinyStories.")

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="Your Story Prompt",
                placeholder="e.g., 'Once upon a time, there was a brave little rabbit'",
                lines=3
            )

            with gr.Accordion("Generation Parameters", open=False):
                max_tokens_slider = gr.Slider(10, 200, 100, 1, label="Max New Tokens")

                temp_slider = gr.Slider(
                    minimum=0.5, maximum=1.2, value=0.9, step=0.05,
                    label="Temperature (higher = more creative)"
                )

                top_p_slider = gr.Slider(
                    minimum=0.7, maximum=1.0, value=0.95, step=0.05,
                    label="Top-p (Nucleus Sampling)"
                )

                top_k_slider = gr.Slider(
                    minimum=10, maximum=100, value=50, step=5,
                    label="Top-k (limit vocabulary)"
                )

                rep_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                    label="Repetition Penalty (higher = less repetition)"
                )

            generate_button = gr.Button("Generate Story", variant="primary")

        with gr.Column(scale=3):
            output_box = gr.Textbox(label="Generated Story", lines=15, interactive=False)

    gr.Examples(
        examples=[
            ["Once upon a time, there was a little girl named"],
            ["One day, a brave knight went to"],
            ["In a magical forest, there lived"],
        ],
        inputs=prompt_box,
    )

    generate_button.click(
        fn=generate_story,
        inputs=[
            prompt_box,
            max_tokens_slider,
            temp_slider,
            top_p_slider,
            top_k_slider,
            rep_penalty_slider
        ],
        outputs=output_box
    )

demo.launch()
