import os
import json
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
import torch

EMBED_MODEL = "all-MiniLM-L6-v2"  
BATCH_SIZE = 128
CHUNK_SIZE = 600           
CHUNK_OVERLAP = 150        
EMBED_DTYPE = np.float32
INDEX_FACTORY = None       
GPT2_ENCODING = "gpt2"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_SEQ_LEN = 1024     

PROMPT_TEMPLATE = (
    "You are an expert AI assistant. Use ONLY the information in the 'Retrieved Context' sections "
    "to answer the question. If the context does not contain the answer, say 'I don't know' and suggest next steps."
    "\n\nRetrieved Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def read_documents_from_path(path: str) -> List[Tuple[str, str]]:
    """Read text documents from a directory or a single file.
    Returns: list of tuples (doc_id, text)
    """
    p = Path(path)
    docs = []
    if p.is_file():
        docs.append((p.name, p.read_text(encoding="utf-8")))
    else:
        for f in sorted(p.glob("**/*.txt")):
            docs.append((str(f.relative_to(p)), f.read_text(encoding="utf-8")))
    return docs


def sentence_aware_chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk text in a way that tries to keep sentences together.
    - Splits by newlines and sentences heuristically, then packs into chunks
    - Character-based sizes used to be robust to technical content
    """
    # Normalize whitespace
    txt = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # Split by paragraphs first
    paras = txt.split("\n\n")

    chunks = []
    current = ""
    for para in paras:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            # If paragraph alone exceeds chunk_size, slice it
            if current:
                chunks.append(current.strip())
            if len(para) > chunk_size:
                # fallback: split paragraph into sliding windows
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end].strip())
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current.strip())

    # Add overlap at boundaries (character-level) to preserve context
    merged = []
    for i, c in enumerate(chunks):
        if i == 0:
            merged.append(c)
        else:
            prev = merged[-1]
            # create overlap
            overlap_text = (prev + " ")[-overlap:]
            merged.append((overlap_text + c).strip())

    # final clean: trim whitespace
    return [m.strip() for m in merged if len(m.strip()) > 0]


# Embedding
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        # Returns numpy array of shape (N, D) float32
        emb = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype(EMBED_DTYPE)
        # Normalize for cosine via inner product in FAISS
        faiss.normalize_L2(emb)
        return emb


# Indexing
class RAGIndex:
    def __init__(self, d: int):
        self.d = d
        # If INDEX_FACTORY is provided, use Faiss index factory; else use flat IP index
        if INDEX_FACTORY:
            self.index = faiss.index_factory(d, INDEX_FACTORY, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(d)

        self.doc_id_to_chunk = []  

    def add(self, emb: np.ndarray, meta: List[Tuple[str, str]]):
        """emb: (N, d) normalized embeddings, meta: list of (doc_id, chunk_text) length N"""
        assert emb.shape[0] == len(meta)
        self.index.add(emb)
        self.doc_id_to_chunk.extend(meta)

    def search(self, q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Tuple[str, str]]]:
        """Return list of (index, score, (doc_id, chunk_text))"""
        # q_emb is (1, d) normalized
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((int(idx), float(score), self.doc_id_to_chunk[idx]))
        return results

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        # Save doc_id_to_chunk mapping as numpy npz
        ids = [d for d, _ in self.doc_id_to_chunk]
        chunks = [c for _, c in self.doc_id_to_chunk]
        np.savez_compressed(meta_path, ids=ids, chunks=chunks)

    @classmethod
    def load(cls, index_path: str, meta_path: str):
        idx = faiss.read_index(index_path)
        # load meta
        dat = np.load(meta_path, allow_pickle=True)
        ids = dat['ids'].tolist()
        chunks = dat['chunks'].tolist()
        obj = cls(idx.d)
        obj.index = idx
        obj.doc_id_to_chunk = list(zip(ids, chunks))
        return obj


# RAG Engine
class RAGEngineV2:
    def __init__(self, index: RAGIndex, embedder: Embedder, tokenizer_name: str = GPT2_ENCODING, seq_len: int = DEFAULT_SEQ_LEN):
        self.index = index
        self.embedder = embedder
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.seq_len = seq_len

    def _encode_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode_batch([query])
        return emb

    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[float, str, str]]:
        q_emb = self._encode_query(query)
        res = self.index.search(q_emb, top_k)
        return [(score, doc_id, chunk_text) for (_, score, (doc_id, chunk_text)) in res]

    def assemble_context(self, retrieved: List[Tuple[float, str, str]], max_context_tokens: int = None) -> str:
        """Assemble context by concatenating top retrieved chunks until token budget is reached."""
        if max_context_tokens is None:
            max_context_tokens = self.seq_len - 64  

        context_parts = []
        used_tokens = 0
        for score, doc_id, chunk in retrieved:
            tok_count = len(self.tokenizer.encode(chunk))
            if used_tokens + tok_count > max_context_tokens:
                continue
            context_parts.append(f"[Source: {doc_id}, score={score:.4f}]\n" + chunk)
            used_tokens += tok_count
            if used_tokens >= max_context_tokens:
                break
        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate_answer(self, model, question: str, top_k: int = 8, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = 0.7):
        retrieved = self.retrieve(question, top_k=top_k)
        context = self.assemble_context(retrieved)
        prompt = self.build_prompt(question, context)

        # Tokenize prompt and ensure length
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) >= self.seq_len:
            input_ids = input_ids[-(self.seq_len - max_new_tokens - 1):]

        x = torch.tensor([input_ids], dtype=torch.long)
        device = next(model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()

        generated = self.tokenizer.decode(out)
        if generated.startswith(prompt):
            answer = generated[len(prompt):]
        else:
            answer = generated
        return answer.strip()


# Index Builder

def build_index_from_documents(docs: List[Tuple[str, str]], embedder: Embedder, index_path: str = None, meta_path: str = None) -> RAGIndex:
    """Docs: list of (doc_id, text). Returns RAGIndex and optionally saves to disk."""
    all_chunks = []
    for doc_id, txt in docs:
        chunks = sentence_aware_chunking(txt)
        for c in chunks:
            all_chunks.append((doc_id, c))

    print(f"Prepared {len(all_chunks)} chunks from {len(docs)} documents.")

    # Batch embed
    texts = [c for (_, c) in all_chunks]
    dummies = embedder.encode_batch([""])
    dim = dummies.shape[1]

    idx = RAGIndex(dim)

    # embed in batches to avoid memory spike
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        emb = embedder.encode_batch(batch_texts)
        meta = all_chunks[i:i+len(batch_texts)]
        idx.add(emb, meta)
        print(f"Indexed {i+len(batch_texts)}/{len(texts)} chunks.")

    if index_path and meta_path:
        idx.save(index_path, meta_path)
        print(f"Saved index to {index_path} and meta to {meta_path}")

    return idx


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Build FAISS index")
    parser.add_argument("--docs", type=str, help="Path to docs")
    parser.add_argument("--index_path", type=str, default="rag_index.faiss")
    parser.add_argument("--meta_path", type=str, default="rag_meta.npz")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5) 
    args = parser.parse_args()

    embedder = Embedder(EMBED_MODEL)

    if args.index:
        if not args.docs:
            print("Error: --docs argument is required for indexing.")
            return
        docs = read_documents_from_path(args.docs)
        build_index_from_documents(docs, embedder, args.index_path, args.meta_path)
        return

    # Check if files exist
    if not os.path.exists(args.index_path):
        print(f"Index file not found at {args.index_path}. Run with --index --docs <folder> first.")
        return

    idx = RAGIndex.load(args.index_path, args.meta_path)
    rag = RAGEngineV2(idx, embedder)

    if args.query:
        # Pure Retrieval Mode (No Model)
        if not args.model_path:
            print(f"Retrieving top {args.top_k} chunks for: '{args.query}'\n")
            retrieved = rag.retrieve(args.query, top_k=args.top_k)
            for score, doc_id, chunk in retrieved:
                print(f"--- [{score:.4f}] {doc_id} ---\n{chunk}\n")
            return

        # Generation Mode (With Model)
        # DYNAMIC CONFIG LOADING (Fixes the crash)
        from nano_gpt.model.model import NanoGptModel
        from nano_gpt.config.model_config import NanoGptConfig
        
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        
        vocab_size = 50257 
        
        cfg = NanoGptConfig(
            vocab_size=vocab_size, 
            embed_dim=768, 
            num_layers=12, 
            num_heads=12, 
            seq_len=DEFAULT_SEQ_LEN
        )
        
        model = NanoGptModel(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Safe state_dict loading
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        print("Generating answer...")
        answer = rag.generate_answer(model, args.query, top_k=args.top_k)
        print(f"\nQ: {args.query}\nA: {answer}")

if __name__ == "__main__":
    main()
