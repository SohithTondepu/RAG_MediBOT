import re
import sys
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from logger import logging
from exception import MedicalAssistantException


class BM25Retriever:
    def __init__(self):
        self.documents: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild_index(self):
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logging.info(f"Rebuilt BM25 index with {len(self.documents)} total chunks.")
        else:
            self.bm25 = None

    def add_documents(self, chunks: List[Document]):
        try:
            for chunk in chunks:
                tokens = self._tokenize(chunk.page_content)
                self.documents.append(chunk)
                self.corpus_tokens.append(tokens)

            self._rebuild_index()
            logging.info(f"Added {len(chunks)} chunks to BM25 index.")
        except Exception as e:
            logging.error(f"Error adding documents to BM25 index: {e}")
            raise MedicalAssistantException(e, sys)

    def remove_document_by_id(self, doc_id: str):
        try:
            new_documents = []
            new_tokens = []

            for doc, tokens in zip(self.documents, self.corpus_tokens):
                if doc.metadata.get("doc_id") != doc_id:
                    new_documents.append(doc)
                    new_tokens.append(tokens)

            removed_count = len(self.documents) - len(new_documents)
            self.documents = new_documents
            self.corpus_tokens = new_tokens
            self._rebuild_index()

            logging.info(f"Removed {removed_count} chunks for doc_id '{doc_id}' from BM25 index.")
        except Exception as e:
            logging.error(f"Error removing document from BM25 index: {e}")
            raise MedicalAssistantException(e, sys)

    def search(self, query: str, filter_dict: Optional[Dict[str, Any]] = None, top_k: int = 20) -> List[Document]:
        try:
            if not self.bm25 or not self.documents:
                return []

            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            scores = self.bm25.get_scores(query_tokens)

            query_set = set(query_tokens)
            scored_docs = []
            for idx, score in enumerate(scores):
                doc_tokens = set(self.corpus_tokens[idx])
                if not query_set.intersection(doc_tokens):
                    continue
                doc = self.documents[idx]


                if filter_dict:
                    match = True
                    for k, v in filter_dict.items():
                        if doc.metadata.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                doc_copy = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                doc_copy.metadata["bm25_score"] = float(score)
                scored_docs.append((doc_copy, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in scored_docs[:top_k]]
            logging.info(f"BM25 search found {len(results)} keyword chunks for query: '{query}'")
            return results

        except Exception as e:
            logging.error(f"Error searching BM25 index: {e}")
            raise MedicalAssistantException(e, sys)
