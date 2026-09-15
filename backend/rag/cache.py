import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from logger import logging


class RAGQueryCache:
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.95, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        # Structure: list of dicts:
        # { "query": str, "query_vec": np.ndarray, "file_name": str, "session_id": str, "answer": str, "sources": list, "timestamp": float }
        self.cache: List[Dict[str, Any]] = []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get(
        self,
        query: str,
        query_vec: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Tuple[str, List]]:
        now = time.time()
        clean_query = query.strip().lower()

        # Evict expired entries
        self.cache = [entry for entry in self.cache if now - entry["timestamp"] < self.ttl_seconds]

        # 1. Exact Match Check
        for entry in reversed(self.cache):
            if (
                entry["query"].strip().lower() == clean_query
                and entry["file_name"] == file_name
                and entry["session_id"] == session_id
            ):
                logging.info(f"⚡ [CACHE HIT - EXACT] Query: '{query}'")
                return entry["answer"], entry["sources"]

        # 2. Semantic Similarity Check (if query_vec provided)
        if query_vec is not None and len(query_vec) > 0:
            best_sim = 0.0
            best_entry = None

            for entry in reversed(self.cache):
                if entry["file_name"] == file_name and entry["session_id"] == session_id:
                    cached_vec = entry.get("query_vec")
                    if cached_vec is not None and len(cached_vec) == len(query_vec):
                        sim = self._cosine_similarity(query_vec, cached_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = entry

            if best_sim >= self.similarity_threshold and best_entry is not None:
                logging.info(f"⚡ [CACHE HIT - SEMANTIC {round(best_sim, 4)}] Query: '{query}' matching '{best_entry['query']}'")
                return best_entry["answer"], best_entry["sources"]

        logging.info(f"❌ [CACHE MISS] Query: '{query}'")
        return None

    def put(
        self,
        query: str,
        answer: str,
        sources: List,
        query_vec: Optional[np.ndarray] = None,
        file_name: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        now = time.time()
        # LRU eviction if max_size reached
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)

        entry = {
            "query": query,
            "query_vec": query_vec,
            "file_name": file_name,
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "timestamp": now
        }
        self.cache.append(entry)
        logging.info(f"💾 [CACHE STORED] Saved query '{query}' to RAG cache.")
