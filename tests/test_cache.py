import sys
import time
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from rag.cache import RAGQueryCache


class TestQueryCache(unittest.TestCase):
    def test_exact_query_hash_cache_hit(self):
        cache = RAGQueryCache(max_size=10, similarity_threshold=0.95, ttl_seconds=3600)

        query = "What is the recommended dosage of Metformin?"
        answer = "The standard starting dosage of Metformin is 500 mg once or twice daily."
        sources = ["page: 12"]

        cache.put(query=query, answer=answer, sources=sources)

        res = cache.get(query=query)
        self.assertIsNotNone(res)
        cached_answer, cached_sources = res
        self.assertEqual(cached_answer, answer)
        self.assertEqual(cached_sources, sources)

    def test_semantic_vector_similarity_cache_hit(self):
        cache = RAGQueryCache(max_size=10, similarity_threshold=0.95, ttl_seconds=3600)

        vec1 = np.array([0.1, 0.2, 0.8, 0.5], dtype=np.float32)
        vec2 = np.array([0.101, 0.199, 0.802, 0.498], dtype=np.float32)

        query1 = "How do I treat diabetes with Metformin?"
        query2 = "How to treat diabetes using Metformin?"
        answer = "Metformin is used as first-line therapy."

        cache.put(query=query1, answer=answer, sources=["file: diabetes.pdf"], query_vec=vec1)

        res = cache.get(query=query2, query_vec=vec2)
        self.assertIsNotNone(res)
        cached_answer, _ = res
        self.assertEqual(cached_answer, answer)

    def test_cache_ttl_eviction(self):
        cache = RAGQueryCache(max_size=10, similarity_threshold=0.95, ttl_seconds=1)

        cache.put(query="Short lived query", answer="Temp answer", sources=[])
        time.sleep(1.1)

        res = cache.get(query="Short lived query")
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
