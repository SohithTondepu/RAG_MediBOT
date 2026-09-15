import re
from typing import List
from logger import logging


class QueryDecomposer:
    def __init__(self, llm=None):
        self.llm = llm

    def decompose(self, query: str) -> List[str]:
        clean_query = query.strip()
        if not clean_query:
            return []

        # Heuristic check: triggers if query has comparative/multi-part keywords
        multi_part_keywords = ["versus", " vs ", " vs. ", "compare", "difference between", " both ", " and "]
        is_multi = any(kw in clean_query.lower() for kw in multi_part_keywords) or len(clean_query.split()) > 15

        if not is_multi or not self.llm:
            return [clean_query]

        prompt = f"""Given the following medical user question, evaluate if it asks about multiple distinct entities, conditions, or comparative aspects.
If it does, decompose it into 2 to 3 concise, self-contained sub-queries that cover all parts of the user's question.
If it is a single direct question, return ONLY the original question.

Rules:
- Output ONLY the sub-queries, one per line.
- Do NOT include numbering, bullet points, or markdown.

User Question: {clean_query}

Sub-queries:"""

        try:
            logging.info(f"Decomposing query: '{clean_query}'")
            response = self.llm.invoke(prompt)
            lines = [line.strip("- *•0123456789. ") for line in response.content.split("\n") if line.strip()]
            sub_queries = [line for line in lines if len(line) > 5]

            if sub_queries:
                logging.info(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")
                return sub_queries
            return [clean_query]
        except Exception as e:
            logging.warning(f"Query decomposition failed: {e}. Falling back to single query.")
            return [clean_query]
