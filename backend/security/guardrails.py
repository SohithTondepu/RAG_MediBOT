from typing import Tuple, Optional
from security.prompt_injection import PromptInjectionDetector
from security.medical_safety import MedicalSafetyDetector


class RAGGuardrails:
    def __init__(self):
        self.injection_detector = PromptInjectionDetector()
        self.medical_safety = MedicalSafetyDetector()

    def validate_input(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates user query against safety guardrails.
        Returns:
            (is_blocked: bool, block_reason: Optional[str], emergency_warning: Optional[str])
        """
        clean_query = query.strip()
        if not clean_query:
            return True, "Empty query provided.", None

        # 1. Prompt Injection Defense
        is_blocked, block_reason = self.injection_detector.detect(clean_query)
        if is_blocked:
            return True, block_reason, None

        # 2. Medical Emergency Advisory Check
        emergency_warning = self.medical_safety.check_emergency(clean_query)

        return False, None, emergency_warning
