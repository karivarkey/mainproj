"""
Query-level caching for RAG results.

Caches query embeddings → RAG documents mapping to avoid re-searching when similar queries arrive.
Uses cosine similarity on query embeddings (384-dim from all-MiniLM-L6-v2).

Cache structure (in-memory + disk):
  {
    "queries": [
      {
        "text": "original english query",
        "embedding": [float, ...],  # 768-dim
        "rag_docs": ["doc 1", "doc 2", ...],
        "timestamp": 1708000000
      },
      ...
    ]
  }
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class QueryCache:
    def __init__(self, cache_file: Path, similarity_threshold: float = 0.80, max_entries: int = 1000):
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.max_entries = max(1, int(max_entries))
        self.queries = []  # List of {text, embedding, rag_docs, timestamp}
        
        # Load cache from disk if it exists
        self.load()

    def _enforce_max_entries(self) -> bool:
        """Keep only the latest max_entries records. Returns True if trimmed."""
        if len(self.queries) <= self.max_entries:
            return False
        self.queries = self.queries[-self.max_entries:]
        return True
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors (both assumed normalized)."""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        
        # Normalize
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        
        return float(np.dot(a_norm, b_norm))
    
    def find_similar_query(self, query_embedding: List[float]) -> Optional[Tuple[List[str], float]]:
        """
        Search cache for a similar query embedding.
        
        Args:
            query_embedding: 384-dim embedding of the query
        
        Returns:
            (rag_docs, similarity_score) if found, else None
        """
        if not self.queries:
            return None
        
        best_sim = -1.0
        best_docs = None
        
        for cached in self.queries:
            cached_emb = cached["embedding"]
            sim = self._cosine_similarity(query_embedding, cached_emb)
            
            if sim > best_sim:
                best_sim = sim
                best_docs = cached["rag_docs"]
        
        if best_sim >= self.similarity_threshold:
            return best_docs, best_sim
        
        return None
    
    def add_query(self, text: str, query_embedding: List[float], rag_docs: List[str]) -> None:
        """
        Add a query and its RAG results to the cache.
        
        Args:
            text: Original English query text
            query_embedding: 384-dim embedding
            rag_docs: List of RAG documents retrieved
        """
        self.queries.append({
            "text": text,
            "embedding": query_embedding,
            "rag_docs": rag_docs,
            "timestamp": time.time(),
        })
        self._enforce_max_entries()
        
        # Persist to disk
        self.save()
    
    def save(self) -> None:
        """Persist cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize embeddings as lists for JSON
            data = {
                "queries": [
                    {
                        "text": q["text"],
                        "embedding": q["embedding"] if isinstance(q["embedding"], list) else q["embedding"].tolist(),
                        "rag_docs": q["rag_docs"],
                        "timestamp": q["timestamp"],
                    }
                    for q in self.queries
                ]
            }
            
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[QueryCache] Warning: Failed to save cache: {e}")
    
    def load(self) -> None:
        """Load cache from disk if it exists."""
        if not self.cache_file.exists():
            return
        
        try:
            data = json.loads(self.cache_file.read_text())
            self.queries = data.get("queries", [])
            # validate dimensions; if mismatch flush cache
            from app.config import RAG_EMBEDDING_DIM
            for q in self.queries:
                emb = q.get("embedding")
                if emb and len(emb) != RAG_EMBEDDING_DIM:
                    print(f"[QueryCache] Info: embedding dim mismatch ({len(emb)} vs {RAG_EMBEDDING_DIM}), clearing cache")
                    self.clear()
                    break
            if self._enforce_max_entries():
                self.save()
        except Exception as e:
            print(f"[QueryCache] Warning: Failed to load cache: {e}")
            self.queries = []
    
    def clear(self) -> None:
        """Clear all cached queries."""
        self.queries = []
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except Exception as e:
            print(f"[QueryCache] Warning: Failed to clear cache: {e}")
    
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "num_cached_queries": len(self.queries),
            "max_entries": self.max_entries,
            "similarity_threshold": self.similarity_threshold,
            "cache_file": str(self.cache_file),
        }
