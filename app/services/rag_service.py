import uuid
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.pdf_service import ingest_pdf
from app.config import (
    RAG_INDEX_FILE, 
    RAG_META_FILE, 
    RAG_EMBEDDING_MODEL, 
    RAG_EMBEDDING_DIM,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K
)

# Lazy-load embedding model to avoid import errors at startup
embed_model = None
DIM = RAG_EMBEDDING_DIM

def get_embed_model():
    """Lazy-load the embedding model on first use."""
    global embed_model, DIM
    if embed_model is None:
        try:
            embed_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
            DIM = embed_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            print("RAG will not work until model is loaded manually.")
            raise
    return embed_model

# Helper to normalize embeddings (unit length)
def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# Load FAISS index (IP = cosine similarity)
if RAG_INDEX_FILE.exists():
    rag_index = faiss.read_index(str(RAG_INDEX_FILE))
else:
    rag_index = faiss.IndexFlatIP(DIM)

# Load metadata
if RAG_META_FILE.exists():
    rag_meta = json.loads(RAG_META_FILE.read_text())
else:
    rag_meta = {}

def save_rag_state():
    faiss.write_index(rag_index, str(RAG_INDEX_FILE))
    RAG_META_FILE.write_text(json.dumps(rag_meta, indent=2))

def rag_add(text: str):
    doc_id = str(uuid.uuid4())
    model = get_embed_model()
    emb = model.encode([text])
    emb = normalize(emb)
    rag_index.add(emb)
    rag_meta[doc_id] = text
    save_rag_state()
    return doc_id

def rag_remove(doc_id: str):
    if doc_id not in rag_meta:
        return False
    
    del rag_meta[doc_id]

    # Rebuild a fresh index
    new_index = faiss.IndexFlatIP(DIM)
    model = get_embed_model()
    for t in rag_meta.values():
        emb = model.encode([t])
        emb = normalize(emb)
        new_index.add(emb)
    
    global rag_index
    rag_index = new_index
    save_rag_state()
    return True

def rag_retrieve(query: str, top_k=None, similarity_threshold=None):
    if top_k is None:
        top_k = RAG_TOP_K
    if similarity_threshold is None:
        similarity_threshold = RAG_SIMILARITY_THRESHOLD
    
    if len(rag_meta) == 0:
        return []
    
    model = get_embed_model()
    q_emb = model.encode([query])
    q_emb = normalize(q_emb)

    sims, idxs = rag_index.search(q_emb, top_k)

    results = []
    all_docs = list(rag_meta.values())
    
    for sim, idx in zip(sims[0], idxs[0]):
        if idx >= len(all_docs):
            continue
        if sim >= similarity_threshold:
            results.append(all_docs[idx])
    
    return results

def rag_list():
    return [{"id": doc_id, "text": text} for doc_id, text in rag_meta.items()]

def rag_clear():
    global rag_index, rag_meta
    rag_index = faiss.IndexFlatIP(DIM)
    rag_meta = {}
    save_rag_state()

def add_pdf_to_rag(pdf_path: str):
    """
    Ingest a PDF, chunk it, embed chunks, and add them to the RAG index.

    Returns:
        dict with pdf_id and number of chunks added
    """
    chunks = ingest_pdf(pdf_path)
    if not chunks:
        return {"pdf_id": None, "chunks_added": 0}

    model = get_embed_model()

    # Embed in batch (critical for performance)
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False
    )
    embeddings = normalize(embeddings)

    # Track starting index position
    start_idx = len(rag_meta)

    # Add to FAISS
    rag_index.add(embeddings)

    pdf_id = str(uuid.uuid4())

    # Store metadata indexed by FAISS position (NOT doc_id)
    for i, text in enumerate(chunks):
        rag_meta[start_idx + i] = {
            "pdf_id": pdf_id,
            "chunk_id": i,
            "text": text,
            "source": pdf_path,
        }

    save_rag_state()

    return {
        "pdf_id": pdf_id,
        "chunks_added": len(chunks),
    }
