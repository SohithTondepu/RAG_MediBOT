import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from rag.decomposer import QueryDecomposer


class TestDecomposer(unittest.TestCase):
    def test_single_direct_query_decomposition(self):
        decomposer = QueryDecomposer(llm=None)

        query = "What is hypertension?"
        sub_queries = decomposer.decompose(query)

        self.assertEqual(len(sub_queries), 1)
        self.assertEqual(sub_queries[0], "What is hypertension?")

    def test_empty_query_decomposition(self):
        decomposer = QueryDecomposer(llm=None)
        self.assertEqual(decomposer.decompose(""), [])
        self.assertEqual(decomposer.decompose("   "), [])


if __name__ == "__main__":
    unittest.main()
