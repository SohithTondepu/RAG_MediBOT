import os
import sys
import uuid
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from config import settings
from logger import logging
from exception import MedicalAssistantException
from core.embedder import Embedder


class PineconeStore:
    def __init__(self, index_name: Optional[str] = None):
        api_key = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")

        if not api_key:
            logging.warning("PINECONE_API_KEY is not set in environment or config!")
            raise ValueError("PINECONE_API_KEY not found")

        self.embedder = Embedder()
        self.embeddings = self.embedder.get_embedding_model()

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name or settings.PINECONE_INDEX_NAME

        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

        logging.info(f"Connected to Pinecone index: {self.index_name}")

    def store_document(self, chunks: List[Document]):
        try:
            ids = [str(uuid.uuid4()) for _ in chunks]
            for chunk in chunks:
                chunk.metadata["chunk_length"] = len(chunk.page_content)
                if "filename" not in chunk.metadata and "source" in chunk.metadata:
                    chunk.metadata["filename"] = os.path.basename(str(chunk.metadata["source"]))
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i: i + batch_size]
                batch_ids = ids[i: i + batch_size]
                self.vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            logging.info(f"Stored {len(chunks)} chunks in Pinecone in batches of {batch_size}")

        except Exception as e:
            logging.error(f"Error in storing document in Pinecone: {e}")
            raise MedicalAssistantException(e, sys)

    def delete_document_vectors(self, doc_id: str):
        try:
            logging.info(f"Deleting vectors for doc_id '{doc_id}' from Pinecone index...")
            self.index.delete(filter={"doc_id": doc_id})
            logging.info(f"Successfully deleted vectors for doc_id '{doc_id}'")
        except Exception as e:
            logging.error(f"Error deleting vectors for doc_id '{doc_id}': {e}")
            raise MedicalAssistantException(e, sys)
