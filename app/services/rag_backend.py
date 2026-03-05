import os
from pathlib import Path
import numpy as np

from app.config import RAG_INDEX_FILE, RAG_EMBEDDING_DIM


class FaissBackend:
    def __init__(self, index_path: Path, dim: int):
        import faiss

        self.faiss = faiss
        self.index_path = index_path
        self.dim = dim
        self.index = None

    def load(self):
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
        else:
            self.index = self.faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings.astype('float32'))

    def search(self, q_emb: np.ndarray, top_k: int):
        return self.index.search(q_emb.astype('float32'), top_k)

    def save(self):
        self.faiss.write_index(self.index, str(self.index_path))

    def reset(self):
        self.index = self.faiss.IndexFlatIP(self.dim)
        self.save()


class BruteBackend:
    """A simple numpy-backed brute-force cosine similarity backend."""
    def __init__(self, index_path: Path, dim: int):
        self.index_path = index_path.with_suffix('.npy')
        self.dim = dim
        self.embeddings = None

    def load(self):
        if self.index_path.exists():
            self.embeddings = np.load(self.index_path)
        else:
            self.embeddings = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, embeddings: np.ndarray):
        emb = embeddings.astype(np.float32)
        if self.embeddings.size == 0:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def search(self, q_emb: np.ndarray, top_k: int):
        # Normalize both
        def _normalize(a):
            a = np.array(a, dtype=np.float32)
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return a / norms

        if self.embeddings.size == 0:
            # return empty arrays compatible with faiss.search
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        emb_norm = _normalize(self.embeddings)
        qn = _normalize(q_emb)
        sims = np.dot(qn, emb_norm.T)
        # get top_k indices
        idxs = np.argsort(-sims, axis=1)[:, :top_k]
        top_sims = np.take_along_axis(sims, idxs, axis=1)
        return top_sims, idxs

    def save(self):
        np.save(self.index_path, self.embeddings)

    def reset(self):
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.save()


class BackendManager:
    def __init__(self, index_path: Path, dim: int):
        self.index_path = index_path
        self.dim = dim
        self.backends = {
            'faiss': lambda: FaissBackend(index_path, dim),
            'brute': lambda: BruteBackend(index_path, dim),
        }
        self.active_name = None
        self.active = None

    def list_backends(self):
        return list(self.backends.keys())

    def load(self, name: str):
        if name not in self.backends:
            raise ValueError(f"unknown backend: {name}")
        self.active_name = name
        self.active = self.backends[name]()
        self.active.load()
        return self.active

    def select(self, name: str):
        if name not in self.backends:
            raise ValueError(f"unknown backend: {name}")
        self.active_name = name
        self.active = None
        return self.active_name

    def unload(self):
        self.active = None
        self.active_name = None

    def ensure_loaded(self):
        if self.active is not None:
            return self.active
        if not self.active_name:
            return None
        return self.load(self.active_name)

    def get_active(self):
        return self.active

    def get_active_name(self):
        return self.active_name

    def get_loaded_name(self):
        if self.active is None:
            return None
        return self.active_name


# Singleton manager
backend_manager = BackendManager(RAG_INDEX_FILE, RAG_EMBEDDING_DIM)

def available_backends():
    return backend_manager.list_backends()

def load_backend(name: str):
    return backend_manager.load(name)

def select_backend(name: str):
    return backend_manager.select(name)

def unload_backend():
    return backend_manager.unload()

def ensure_active_backend_loaded():
    return backend_manager.ensure_loaded()

def get_active_backend():
    return backend_manager.get_active()

def get_active_backend_name():
    return backend_manager.get_active_name()

def get_loaded_backend_name():
    return backend_manager.get_loaded_name()
