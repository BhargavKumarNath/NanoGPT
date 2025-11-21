import torch
import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.model import NanoGptModel
from nano_gpt.config.model_config import NanoGptConfig
import tiktoken

class RAGEngine:
    def __init__(self, model_path, doc_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Load LLM
        print("Loading LLM...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # These values must match your train_model.py exactly
        config = NanoGptConfig(
            vocab_size=50257, # Standard GPT-2 vocab size, or check your meta.json
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            seq_len=512,
            dropout=0.0,
            attention_type='gqa',
            num_kv_heads=4,
            ffn_hidden_dim=int(768 * 8/3)
        )
        
        self.model = NanoGptModel(config).to(self.device) 

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.enc = tiktoken.get_encoding("gpt2")

        # Load Embedding Model 
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Index Document
        print(f"Indexing {doc_path}...")
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        self.chunk_embeddings = self.embedder.encode(self.chunks)

    def retrieve(self, query, k=2):
        query_emb = self.embedder.encode([query])
        # Cosine Similarity
        scores = np.dot(self.chunk_embeddings, query_emb.T).flatten()
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in top_k_indices]
    
    def generate_answer(self, query):
        context_chunks = self.retrieve(query)
        context_text = "\n".join(context_chunks)

        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Tokenize
        input_ids = self.enc.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate 
        with torch.no_grad():
            out_ids = self.model.generate(x, max_new_tokens=100, temperature=0.7)[0].tolist()
        
        generated_text = self.enc.decode(out_ids)
        
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):]
        else:
            answer = generated_text

        return answer

if __name__ == "__main__":
    # Create dummy file for testing
    with open("research_paper.txt", "w") as f:
        f.write("""The Transformer model relies on self-attention mechanisms. Unlike RNNs, Transformers process the entire sequence in parallel. The key components are Query, Key and Value matrices. Attention scores are calculated using the dot product on Query and Key.""")
    
    # Make sure the path exists before running
    model_path = "out/cosmopedia_pro_v1/ckpt_500.pt"
    if os.path.exists(model_path):
        rag = RAGEngine(model_path, "research_paper.txt")
        print(f"\n--- RAG DEMO ---")
        print("Q: How do Transformers differ from RNNs?")
        print(f"A {rag.generate_answer('How do Transformers differ from RNNs?')}")
    else:
        print(f"Please run train_model.py first to generate {model_path}")