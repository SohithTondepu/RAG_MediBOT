import os
import sys
from langchain_groq import ChatGroq
from config import settings
from logger import logging
from exception import MedicalAssistantException


def load_llm():
    try:
        logging.info("Loading Groq LLM...")
        groq_api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.warning("GROQ_API_KEY is not set in environment or config!")
            raise ValueError("GROQ_API_KEY is not set")
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )

        logging.info("LLM model loaded successfully.")
        return llm
    except Exception as e:
        raise MedicalAssistantException(e, sys)
