import sys
import csv
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader
)
from logger import logging
from exception import MedicalAssistantException


class DocumentLoader:

    def __init__(self, file_path: str):
        self.file_path = Path(file_path).resolve()

        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} does not exist")

        logging.info(f"DocumentLoader initialized for file: {self.file_path}")

    def _load_csv_structured(self) -> List[Document]:
        """Loads CSV files formatting rows as structured key-value pairs to preserve header context."""
        documents = []
        try:
            with open(self.file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                for row_idx, row in enumerate(reader, 1):
                    row_str = " | ".join([f"{k}: {v}" for k, v in row.items() if v and v.strip()])
                    doc_text = f"[Table Row {row_idx}] {row_str}"
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "source": str(self.file_path),
                            "row": row_idx,
                            "headers": ", ".join(headers),
                            "is_table": True
                        }
                    )
                    documents.append(doc)
            return documents
        except Exception as e:
            logging.warning(f"Structured CSV loading failed ({e}). Falling back to standard loader.")
            loader = CSVLoader(str(self.file_path))
            return loader.load()

    def get_loaders(self):
        suffix = self.file_path.suffix.lower()
        loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader
        }
        if suffix == ".csv":
            return None  # Handled by _load_csv_structured
        if suffix in loader_map:
            return loader_map[suffix](str(self.file_path))
        return UnstructuredFileLoader(str(self.file_path))

    def load_documents(self) -> List[Document]:
        try:
            suffix = self.file_path.suffix.lower()
            if suffix == ".csv":
                documents = self._load_csv_structured()
            else:
                loader = self.get_loaders()
                documents = loader.load() if loader else []

            for doc in documents:
                doc.metadata["source"] = str(self.file_path)
                doc.metadata["filename"] = self.file_path.name

            logging.info(f"Loaded {len(documents)} document pages/rows from {self.file_path}")
            return documents

        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            raise MedicalAssistantException(e, sys)