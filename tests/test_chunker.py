import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from langchain_core.documents import Document
from core.chunker import TextChunker


class TestChunker(unittest.TestCase):
    def test_header_aware_section_tracking(self):
        chunker = TextChunker(chunk_size=150, chunk_overlap=20)

        doc_content = """# Oncology Protocol Guidelines

## Chemotherapy Administration
Chemotherapy drugs should be administered strictly under oncology supervision.

## Patient Monitoring
Monitor white blood cell count and vital signs every 4 hours."""

        doc = Document(page_content=doc_content, metadata={"source": "oncology.pdf"})
        chunks = chunker.chunk_documents([doc])

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("section_path", chunk.metadata)
            self.assertTrue(chunk.page_content.startswith("[Section:"))

    def test_structured_table_row_preservation(self):
        chunker = TextChunker()

        table_doc = Document(
            page_content="[Table Row 1] PatientID: 101 | Drug: Metformin | Dosage: 500mg",
            metadata={"source": "patients.csv", "row": 1, "is_table": True}
        )

        chunks = chunker.chunk_documents([table_doc])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, table_doc.page_content)
        self.assertTrue(chunks[0].metadata["is_table"])


if __name__ == "__main__":
    unittest.main()
