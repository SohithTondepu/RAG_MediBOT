import re
from typing import Optional
from logger import logging


class MedicalSafetyDetector:
    def __init__(self):
        self.emergency_patterns = [
            r"chest\s*pain",
            r"heart\s*attack",
            r"cardiac\s*arrest",
            r"stroke",
            r"cannot\s*breathe",
            r"can't\s*breathe",
            r"unable\s+to\s+breathe",
            r"trouble\s+breathing",
            r"difficulty\s+breathing",
            r"shortness\s+of\s+breath",
            r"gasping",
            r"choking",
            r"severe\s+bleeding",
            r"unconscious",
            r"loss\s+of\s+consciousness",
            r"passed\s*out",
            r"fainted",
            r"anaphylaxis",
            r"overdose",
            r"poisoning",
            r"suicide",
            r"self-harm",
            r"seizure",
        ]


    def check_emergency(self, query: str) -> Optional[str]:
        clean_query = query.strip()
        for pattern in self.emergency_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                logging.info(f"Emergency intent detected for pattern '{pattern}'")
                return (
                    "⚠️ **URGENT MEDICAL NOTICE**: If you or someone else is experiencing a life-threatening medical emergency "
                    "(such as severe chest pain, sudden numbness/weakness, severe shortness of breath, or heavy bleeding), "
                    "please call emergency services (911 or your local emergency number) or go to the nearest emergency room immediately.\n\n"
                )
        return None
