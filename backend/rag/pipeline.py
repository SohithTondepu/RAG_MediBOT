import os
import sys
from typing import Optional, List, Tuple, Dict

from config import settings
from logger import logging
from exception import MedicalAssistantException
from core.llm import load_llm
from retrieval.hybrid import HybridRetriever
from security.guardrails import RAGGuardrails
from rag.cache import RAGQueryCache
from rag.decomposer import QueryDecomposer


class RAGPipeline:
    def __init__(self):
        logging.info("Initializing RAG Pipeline...")
        self.retriever = HybridRetriever()
        self.llm = load_llm()
        self.guardrails = RAGGuardrails()
        self.cache = RAGQueryCache(
            similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
            ttl_seconds=settings.CACHE_TTL_SECONDS
        )
        self.decomposer = QueryDecomposer(llm=self.llm)
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}

    def _get_session_history(self, session_id: Optional[str]) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def _rewrite_question(self, current_question: str, history: List[Tuple[str, str]]) -> str:
        if not history:
            return current_question

        history_text = ""
        for q, a in history[-2:]:
            history_text += f"User: {q}\nAssistant: {a}\n"

        rewrite_prompt = f"""Given the conversation history and a new user question, rewrite the new question to be a self-contained, standalone query that includes all necessary context.

Conversation History:
{history_text}

New Question: {current_question}

Standalone Question:"""

        try:
            response = self.llm.invoke(rewrite_prompt)
            rewritten = response.content.strip()
            logging.info(f"Original question: '{current_question}' -> Rewritten: '{rewritten}'")
            return rewritten
        except Exception as e:
            logging.warning(f"Question rewriting failed: {e}. Using original question.")
            return current_question

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        file_name: Optional[str] = None,
        category: Optional[str] = None
    ) -> Tuple[str, List]:
        try:
            logging.info(f"Answering question for session_id='{session_id}', file_name='{file_name}', category='{category}'")

            # --- 1. GUARDRAILS VALIDATION ---
            is_blocked, block_reason, emergency_warning = self.guardrails.validate_input(question)
            if is_blocked and block_reason:
                logging.info(f"Query blocked by guardrails: {block_reason}")
                return block_reason, []

            # --- 2. CACHE LOOKUP ---
            query_vec = None
            try:
                query_vec = self.retriever.embedder.embed_query_vector(question)
            except Exception as e:
                logging.warning(f"Failed to compute query vector for cache lookup: {e}")

            cached_res = self.cache.get(
                query=question,
                query_vec=query_vec,
                file_name=file_name,
                session_id=session_id
            )
            if cached_res:
                cached_answer, cached_chunks = cached_res
                if emergency_warning and not cached_answer.startswith("⚠️ **URGENT"):
                    cached_answer = emergency_warning + cached_answer
                return cached_answer, cached_chunks

            # --- 3. CONVERSATIONAL REWRITING ---
            history = self._get_session_history(session_id)
            rewritten_question = self._rewrite_question(question, history)

            # Evaluate guardrails on rewritten question if raw input did not trigger emergency warning
            if not emergency_warning and rewritten_question != question:
                _, _, rewritten_emergency = self.guardrails.validate_input(rewritten_question)
                if rewritten_emergency:
                    emergency_warning = rewritten_emergency


            # --- 4. MULTI-HOP DECOMPOSITION & HYBRID RETRIEVAL ---
            filter_dict = {}
            if file_name and file_name != "all":
                filter_dict["filename"] = file_name
            if category:
                filter_dict["category"] = category.lower()
            if not filter_dict:
                filter_dict = None

            sub_queries = self.decomposer.decompose(rewritten_question)
            top_k_chunks = self.retriever.retrieve_multi(
                queries=sub_queries,
                main_query=rewritten_question,
                filter_dict=filter_dict,
                top_k=7
            )

            if not top_k_chunks and filter_dict:
                logging.info(f"No chunks found with filter {filter_dict}. Retrying without filter.")
                top_k_chunks = self.retriever.retrieve_multi(
                    queries=sub_queries,
                    main_query=rewritten_question,
                    top_k=7
                )

            formatted_context_list = []
            for idx, doc in enumerate(top_k_chunks, 1):
                src = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "doc"))
                page = doc.metadata.get("page")
                page_str = f", Page {page + 1 if isinstance(page, int) and page == 0 else page}" if page is not None else ""
                score_str = f" [Score: {round(doc.metadata['rerank_score'], 3)}]" if "rerank_score" in doc.metadata else ""
                formatted_context_list.append(f"--- Document Chunk {idx} (Source: {src}{page_str}{score_str}) ---\n{doc.page_content}")

            context = "\n\n".join(formatted_context_list)

            history_text = ""
            if history:
                recent_history = history[-3:]
                for q, a in recent_history:
                    history_text += f"User: {q}\nAssistant: {a}\n"

            prompt = f"""You are **MediBot**, an AI-powered medical assistant.

{f"📜 **Previous Conversation**:\n{history_text}" if history_text else ""}

🔍 **Medical Context**:
{context if context else "No context available."}

🙋‍♂️ **Current Question**:
{question}

---
💬 **Instructions**:
- Use previous conversation context if relevant.
- Answer primarily using the provided medical context across all referenced documents.
- If the answer cannot be found or inferred from the context, state clearly: "I could not find this information in the uploaded medical documents."
- Keep answers concise, clear, and medically informative with explicit source citations.
"""

            response = self.llm.invoke(prompt)
            answer = response.content

            if emergency_warning:
                answer = emergency_warning + answer

            # --- 5. STORE IN CACHE & SESSION HISTORY ---
            self.cache.put(
                query=question,
                answer=answer,
                sources=top_k_chunks,
                query_vec=query_vec,
                file_name=file_name,
                session_id=session_id
            )

            if session_id:
                self.sessions[session_id].append((question, answer))
                if len(self.sessions[session_id]) > 10:
                    self.sessions[session_id] = self.sessions[session_id][-10:]

            logging.info("Answer generated successfully")
            return answer, top_k_chunks

        except Exception as e:
            logging.error(f"Error in RAG ask pipeline: {e}")
            raise MedicalAssistantException(e, sys)
