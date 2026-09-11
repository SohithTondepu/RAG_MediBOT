from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document   
from logger import logging
from exception import MedicalAssistantException
import numpy as np,sys

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)
        logging.info(f"model loaded successfully with dimension : {self.model.get_sentence_embedding_dimension()}")

    def embed_chunks(self, chunks: List[Document]) -> np.ndarray:

        try:
            texts = [chunk.page_content for chunk in chunks]
            logging.info("Generating embeddings...")
            print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

            embeddings = self.model.encode(texts, show_progress_bar=True)
            

            print(f"[INFO] Embeddings shape: {embeddings.shape}")

            return embeddings
        except Exception as e:
            raise MedicalAssistantException(e,sys)