import sys
from typing import List, Tuple, Optional
from functools import lru_cache
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from logger import logging
from exception import MedicalAssistantException


class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        try:
            self.model_name = model_name or settings.EMBEDDING_MODEL
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logging.info(f"Embedder model loaded successfully: {self.model_name}")
        except Exception as e:
            raise MedicalAssistantException(e, sys)

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        return self.embeddings

    def embed_documents(self, texts: List[str]):
        return self.embeddings.embed_documents(texts)

    @lru_cache(maxsize=1024)
    def _cached_embed_query(self, text: str) -> Tuple[float, ...]:
        return tuple(self.embeddings.embed_query(text))

    def embed_query(self, text: str) -> List[float]:
        return list(self._cached_embed_query(text))

    def embed_query_vector(self, text: str) -> np.ndarray:
        return np.array(self._cached_embed_query(text), dtype=np.float32)
