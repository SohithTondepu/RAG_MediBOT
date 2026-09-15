import sys
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logging
from exception import MedicalAssistantException


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n### ", "\n## ", "\n# ", ". ", "? ", "! ", "\n", " "]
        )
        logging.info(f"HeaderAware TextChunker initialized | size={chunk_size} overlap={chunk_overlap}")

    def _extract_header(self, text: str) -> Optional[str]:
        """Detects section headings in text lines."""
        lines = text.split("\n")
        header_pattern = re.compile(
            r"^(#{1,6}\s+.+|SECTION\s+[\dA-Z]+.*|CHAPTER\s+[\dA-Z]+.*|[A-Z\s]{4,40}:)$",
            re.IGNORECASE
        )
        for line in lines[:3]:
            cleaned = line.strip()
            if header_pattern.match(cleaned):
                return cleaned.lstrip("#").strip()
        return None

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            logging.info(f"Chunking {len(documents)} document pages with Header-Aware Splitter...")
            processed_chunks: List[Document] = []
            current_section = "General Section"

            for doc in documents:
                if doc.metadata.get("is_table"):
                    processed_chunks.append(doc)
                    continue

                page_chunks = self.splitter.split_documents([doc])

                for chunk in page_chunks:
                    header = self._extract_header(chunk.page_content)
                    if header:
                        current_section = header

                    chunk.metadata["section_path"] = current_section

                    if not chunk.page_content.startswith("[Section:"):
                        chunk.page_content = f"[Section: {current_section}]\n{chunk.page_content}"

                    processed_chunks.append(chunk)

            logging.info(f"Successfully split {len(documents)} document pages into {len(processed_chunks)} Header-Aware chunks.")
            return processed_chunks

        except Exception as e:
            logging.error(f"Error chunking documents: {e}")
            raise MedicalAssistantException(e, sys)
