import sys
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
        """
        Initialize loader with directory containing PDF files
        """
        self.file_path = Path(file_path).resolve()

        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} does not exist")

        logging.info(f"DocumentLoader initialized for directory: {self.file_path}")

    
    def get_loaders(self):
        suffix=self.file_path.suffix.lower()
        loader_map={
            ".pdf":PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader
        }
        if suffix in loader_map:
            return loader_map[suffix](str(self.file_path))
        return UnstructuredFileLoader(str(self.file_path))



    def load_documents(self) -> List[Document]:
        try:
            loader=self.get_loaders()
            
            documents=loader.load()
            for doc in documents:
                doc.metadata["source"] = str(self.file_path)
            logging.info(f"Loaded {len(documents)} pages from {self.file_path}")
            return documents

        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            raise MedicalAssistantException(e, sys)