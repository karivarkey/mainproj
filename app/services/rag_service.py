import uuid
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.pdf_service import ingest_pdf
from app.services.rag_backend import (
    available_backends,
    load_backend,
    get_active_backend,
    get_active_backend_name,
)
from app.config import (
    RAG_META_FILE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K,
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


def normalize(v):
    """Normalize embeddings (unit length)."""
    v = np.array(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


# Initialize backend (default to faiss if available)
try:
    load_backend("faiss")
except Exception:
    backends = available_backends()
    if backends:
        load_backend(backends[0])


# Load metadata
if RAG_META_FILE.exists():
    rag_meta = json.loads(RAG_META_FILE.read_text())
else:
    rag_meta = {}


def _get_backend():
    b = get_active_backend()
    if b is None:
        raise RuntimeError("No RAG backend loaded")
    return b


def save_rag_state():
    """Persist backend index/state and metadata."""
    backend = _get_backend()
    try:
        backend.save()
    except Exception:
        pass

    RAG_META_FILE.write_text(json.dumps(rag_meta, indent=2))


def reindex_backend():
    """Rebuild the currently loaded backend from `rag_meta` contents."""
    backend = _get_backend()
    backend.reset()

    model = get_embed_model()

    texts = []
    for obj in rag_meta.values():
        if isinstance(obj, str):
            # backward compatibility if old data stored plain strings
            texts.append(obj)
        else:
            texts.append(obj.get("text", ""))

    # clean empty texts
    texts = [t for t in texts if isinstance(t, str) and t.strip()]

    if texts:
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = normalize(embeddings)
        backend.add(embeddings)

    save_rag_state()


def rag_add(text: str):
    doc_id = str(uuid.uuid4())
    model = get_embed_model()

    emb = model.encode([text])
    emb = normalize(emb)

    backend = _get_backend()
    backend.add(emb)

    rag_meta[doc_id] = {
        "type": "text",
        "text": text,
        "source": "manual",
    }

    save_rag_state()
    return doc_id


def rag_remove(doc_id: str):
    if doc_id not in rag_meta:
        return False

    del rag_meta[doc_id]

    # Rebuild backend index from remaining metadata
    reindex_backend()
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

    backend = _get_backend()
    sims, idxs = backend.search(q_emb, top_k)

    results = []

    # IMPORTANT: list order matters because FAISS idx refers to insertion order
    all_docs = list(rag_meta.values())

    for sim, idx in zip(sims[0], idxs[0]):
        if idx >= len(all_docs):
            continue
        if sim < similarity_threshold:
            continue

        doc = all_docs[idx]
        if isinstance(doc, str):
            results.append(doc)
        else:
            results.append(doc.get("text", ""))

    # remove blanks
    results = [r for r in results if isinstance(r, str) and r.strip()]
    return results


def rag_list():
    docs = []
    for doc_id, obj in rag_meta.items():
        if isinstance(obj, str):
            docs.append({"id": doc_id, "text": obj})
        else:
            docs.append({"id": doc_id, "text": obj.get("text", "")})
    return docs


def rag_clear():
    global rag_meta
    backend = _get_backend()
    backend.reset()

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

    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
    embeddings = normalize(embeddings)

    backend = _get_backend()
    backend.add(embeddings)

    pdf_id = str(uuid.uuid4())

    for i, text in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        rag_meta[doc_id] = {
            "type": "pdf",
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
