import re
from typing import Tuple, Optional
from logger import logging


class PromptInjectionDetector:
    def __init__(self):
        self.injection_patterns = [
            r"(ignore|disregard|forget|override|bypass|drop)\s+.*(instruction|direction|rule|guideline|prompt|constraint|system|context)",
            r"(ignore|disregard|forget)\s+(everything|all)",
            r"you\s+are\s+now\s+(a|an)",
            r"act\s+as\s+(a|an)",
            r"pretend\s+(you\s+are|to\s+be)",
            r"simulate\s+(a|an)",
            r"system\s+prompt",
            r"reveal\s+.*prompt",
            r"show\s+.*prompt",
            r"jailbreak",
            r"dan\s+mode",
            r"developer\s+mode",
            r"repeat\s+after\s+me",
            r"bypass\s+safety",
        ]

    def detect(self, query: str) -> Tuple[bool, Optional[str]]:
        clean_query = query.strip()
        for pattern in self.injection_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                logging.warning(f"Prompt injection detected for pattern '{pattern}' in query: '{clean_query}'")
                return True, "🚨 **Security Alert**: Your query was flagged by system safety guardrails. Please enter a valid medical question."
        return False, None
