import uuid
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.pdf_service import ingest_pdf
from app.services.rag_backend import (
    available_backends,
    load_backend,
    get_active_backend,
)
from app.config import (
    RAG_META_FILE,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
)

embed_model = None


def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
    return embed_model


def normalize(v):
    v = np.array(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


# Initialize backend
try:
    load_backend("faiss")
except Exception:
    load_backend(available_backends()[0])


# ---------- LOAD META ----------

if RAG_META_FILE.exists():
    rag_meta = json.loads(RAG_META_FILE.read_text())
else:
    rag_meta = {"documents": {}, "chunks": []}


def save_state():
    backend = get_active_backend()
    backend.save()
    RAG_META_FILE.write_text(json.dumps(rag_meta, indent=2))

def rag_remove(doc_id: str):
    """
    Remove a full document (PDF or manual) and rebuild index.
    """
    if doc_id not in rag_meta["documents"]:
        return False

    # Remove document metadata
    del rag_meta["documents"][doc_id]

    # Remove all its chunks
    rag_meta["chunks"] = [
        c for c in rag_meta["chunks"] if c["doc_id"] != doc_id
    ]

    # Rebuild index from remaining chunks
    backend = get_active_backend()
    backend.reset()

    if rag_meta["chunks"]:
        model = get_embed_model()
        texts = [c["text"] for c in rag_meta["chunks"]]
        embeddings = model.encode(texts, batch_size=32)
        embeddings = normalize(embeddings)
        backend.add(embeddings)

    save_state()
    return True



def rag_list():
    """Return only high-level documents (no chunks)."""
    return [
        {
            "id": doc_id,
            "source": doc["source"],
            "type": doc["type"],
        }
        for doc_id, doc in rag_meta["documents"].items()
    ]


def rag_clear():
    global rag_meta
    backend = get_active_backend()
    backend.reset()
    rag_meta = {"documents": {}, "chunks": []}
    save_state()


# ---------- ADD MANUAL DOCUMENT ----------

def rag_add(text: str):
    model = get_embed_model()
    doc_id = str(uuid.uuid4())

    emb = model.encode([text])
    emb = normalize(emb)

    backend = get_active_backend()
    backend.add(emb)

    rag_meta["documents"][doc_id] = {
        "type": "manual",
        "source": "manual_entry",
        "num_chunks": 1,
    }

    rag_meta["chunks"].append({
        "doc_id": doc_id,
        "text": text,
    })

    save_state()
    return doc_id


# ---------- ADD PDF ----------

def add_pdf_to_rag(pdf_path: str):
    chunks = ingest_pdf(pdf_path)
    if not chunks:
        return {"pdf_id": None, "chunks_added": 0}

    model = get_embed_model()
    embeddings = model.encode(chunks, batch_size=32)
    embeddings = normalize(embeddings)

    backend = get_active_backend()
    backend.add(embeddings)

    doc_id = str(uuid.uuid4())

    rag_meta["documents"][doc_id] = {
        "type": "pdf",
        "source": pdf_path,
        "num_chunks": len(chunks),
    }

    for chunk in chunks:
        rag_meta["chunks"].append({
            "doc_id": doc_id,
            "text": chunk,
        })

    save_state()

    return {"pdf_id": doc_id, "chunks_added": len(chunks)}


# ---------- RETRIEVE ----------

def rag_retrieve(query: str, top_k=None, similarity_threshold=None):
    if top_k is None:
        top_k = RAG_TOP_K
    if similarity_threshold is None:    
        similarity_threshold = RAG_SIMILARITY_THRESHOLD

    if not rag_meta["chunks"]:
        return []

    model = get_embed_model()
    q_emb = model.encode([query])
    q_emb = normalize(q_emb)

    backend = get_active_backend()
    sims, idxs = backend.search(q_emb, top_k)

    results = []

    for sim, idx in zip(sims[0], idxs[0]):
        print("SIM SCORE:", sim)

        if idx >= len(rag_meta["chunks"]):
            continue
        if sim < similarity_threshold:
            continue

        chunk = rag_meta["chunks"][idx]
        results.append({
            "text": chunk["text"],
            "source": rag_meta["documents"][chunk["doc_id"]]["source"],
            "similarity": float(sim)
        })

    return results
