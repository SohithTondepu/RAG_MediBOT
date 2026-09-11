import sys
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logging
from exception import MedicalAssistantException

class TextChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        logging.info("chunker initialised")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        print(f"[INFO] Chunker initialized | size={chunk_size} overlap={chunk_overlap}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            logging.info("chunking documents")
            chunks = self.splitter.split_documents(documents)

            print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")

            return chunks
        except Exception as e:
            logging.error("error in chunking documents {}".format(e))
            raise MedicalAssistantException(e,sys)
    