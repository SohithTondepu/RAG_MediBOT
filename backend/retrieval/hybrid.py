import os
import sys
from typing import Optional, Dict, Any, List
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from config import settings
from logger import logging
from exception import MedicalAssistantException
from core.embedder import Embedder
from retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        index_name: Optional[str] = None,
        top_k: int = 5,
        fetch_k: int = 20,
        enable_rerank: bool = True
    ):
        logging.info("Initializing HybridRetriever (Dense + Sparse BM25 + CrossEncoder)...")

        self.embedder = Embedder()
        self.embeddings = self.embedder.get_embedding_model()
        self.bm25_retriever = BM25Retriever()
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.enable_rerank = enable_rerank

        api_key = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")
        if not api_key:
            logging.warning("PINECONE_API_KEY is not set in environment or config!")
            raise ValueError("PINECONE_API_KEY not set")

        idx_name = index_name or settings.PINECONE_INDEX_NAME
        pc = Pinecone(api_key=api_key)

        existing_indexes = [i.name for i in pc.list_indexes()]
        if idx_name not in existing_indexes:
            logging.info(f"Pinecone index '{idx_name}' not found. Creating serverless index (384d, cosine)...")
            from pinecone import ServerlessSpec
            pc.create_index(
                name=idx_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logging.info(f"Pinecone index '{idx_name}' created successfully.")

        index = pc.Index(idx_name)

        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings
        )


        self.reranker = None
        if self.enable_rerank:
            try:
                logging.info(f"Loading CrossEncoder model ('{settings.RERANK_MODEL}')...")
                self.reranker = CrossEncoder(settings.RERANK_MODEL)
                logging.info("CrossEncoder reranker loaded successfully.")
            except Exception as e:
                logging.warning(f"Could not load CrossEncoder: {e}. Falling back to standard search.")
                self.reranker = None

        logging.info(f"Initialized HybridRetriever with top_k={top_k}, fetch_k={fetch_k}, rerank={self.reranker is not None}")

    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        rrf_k: int = 60
    ) -> List[Document]:
        """Combines Dense Vector and BM25 Sparse search rankings using Reciprocal Rank Fusion."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, 1):
            content_key = doc.page_content.strip()
            doc_map[content_key] = doc
            rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + (1.0 / (rrf_k + rank))

        for rank, doc in enumerate(sparse_docs, 1):
            content_key = doc.page_content.strip()
            if content_key not in doc_map:
                doc_map[content_key] = doc
            rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + (1.0 / (rrf_k + rank))

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        fused_docs = []
        for k in sorted_keys:
            d = doc_map[k]
            d.metadata["rrf_score"] = float(rrf_scores[k])
            fused_docs.append(d)

        logging.info(f"RRF fused {len(dense_docs)} dense + {len(sparse_docs)} sparse chunks into {len(fused_docs)} candidates.")
        return fused_docs

    def retrieve(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        try:
            logging.info(f"Retrieving user query using Hybrid Search with filter: {filter_dict}")

            search_kwargs: Dict[str, Any] = {"k": self.fetch_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict

            # 1. Dense Vector Search (Pinecone)
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            dense_docs = retriever.invoke(query)

            # 2. Sparse Keyword Search (BM25)
            sparse_docs = self.bm25_retriever.search(query, filter_dict=filter_dict, top_k=self.fetch_k)

            # 3. Reciprocal Rank Fusion (RRF)
            hybrid_candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)

            if not hybrid_candidates:
                return []

            # 4. Cross-Encoder Reranking
            if self.reranker and len(hybrid_candidates) > 1:
                logging.info(f"Cross-Encoder Reranking {len(hybrid_candidates)} hybrid candidates for query: '{query}'")
                pairs = [[query, doc.page_content] for doc in hybrid_candidates]
                scores = self.reranker.predict(pairs)

                for doc, score in zip(hybrid_candidates, scores):
                    doc.metadata["rerank_score"] = float(score)

                reranked_docs = sorted(hybrid_candidates, key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
                final_docs = reranked_docs[:self.top_k]
                logging.info(f"Top hybrid reranked scores: {[round(d.metadata['rerank_score'], 4) for d in final_docs]}")
                return final_docs

            return hybrid_candidates[:self.top_k]

        except Exception as e:
            logging.error(f"Error in hybrid retrieval: {e}")
            raise MedicalAssistantException(e, sys)

    def retrieve_multi(
        self,
        queries: List[str],
        main_query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 7
    ) -> List[Document]:
        try:
            logging.info(f"Executing hybrid multi-query retrieval across {len(queries)} queries with filter: {filter_dict}")
            all_candidate_docs = []
            seen_contents = set()

            for q in queries:
                q_docs = self.retrieve(q, filter_dict=filter_dict)
                for doc in q_docs:
                    content_hash = hash(doc.page_content.strip())
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_candidate_docs.append(doc)

            if not all_candidate_docs:
                return []

            if self.reranker and len(all_candidate_docs) > 1:
                logging.info(f"Re-ranking {len(all_candidate_docs)} hybrid deduplicated chunks against main query: '{main_query}'")
                pairs = [[main_query, doc.page_content] for doc in all_candidate_docs]
                scores = self.reranker.predict(pairs)

                for doc, score in zip(all_candidate_docs, scores):
                    doc.metadata["rerank_score"] = float(score)

                reranked = sorted(all_candidate_docs, key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
                final_docs = reranked[:top_k]
                logging.info(f"Multi-query hybrid top reranked scores: {[round(d.metadata['rerank_score'], 4) for d in final_docs]}")
                return final_docs

            return all_candidate_docs[:top_k]

        except Exception as e:
            logging.error(f"Error in multi-query hybrid retrieval: {e}")
            raise MedicalAssistantException(e, sys)
