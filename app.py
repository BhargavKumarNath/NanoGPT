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
CHECKPOINT_PATH = "out/mha_baseline/final_model.pt"
TOKENIZER_PATH = "out/mha_baseline/tinystories_tokenizer"

# Model and Tokenizer Loading
print("Loading model and tokenizer...")

# Load tokenizer
tokenizer = BpeTokenizer()
assert os.path.exists(TOKENIZER_PATH + ".vocab.json"), "Tokenizer not found"
tokenizer.load(TOKENIZER_PATH)

# Load model
assert os.path.exists(CHECKPOINT_PATH), "Model checkpoint not found"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

config = checkpoint['model_config']
model = NanoGptModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Model and tokenizer loaded successfully.")

# Generation Function
def generate_story(prompt, max_new_tokens, temperature, top_p):
    """The main function that Gradio will call."""
    if not prompt:
        return "Please provide a starting prompt."
        
    try:
        # Encode the prompt
        start_tokens = tokenizer.encode(prompt)
        start_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)
        
        # Generate
        generated_tokens = model.generate(
            start_tokens_tensor,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=top_p
        )
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        
        return generated_text
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
print("Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultimate Nano-GPT: Interactive Story Generator")
    gr.Markdown("This demo uses a 100% from-scratch GPT model trained on the TinyStories dataset. Provide a prompt and watch it generate a story!")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(label="Your Story Prompt", placeholder="e.g., 'Once upon a time, there was a brave little rabbit named Pip.'", lines=3)
            
            with gr.Accordion("Generation Parameters", open=False):
                max_tokens_slider = gr.Slider(minimum=10, maximum=250, value=100, step=1, label="Max New Tokens")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)")

            generate_button = gr.Button("Generate Story", variant="primary")
            
        with gr.Column(scale=3):
            output_box = gr.Textbox(label="Generated Story", lines=15, interactive=False)
            
    generate_button.click(
        fn=generate_story,
        inputs=[prompt_box, max_tokens_slider, temp_slider, top_p_slider],
        outputs=output_box
    )

# Launch the app
demo.launch()
