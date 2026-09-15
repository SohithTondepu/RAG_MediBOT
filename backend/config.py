import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from backend/ or root workspace directory
backend_dir = Path(__file__).resolve().parent
root_dir = backend_dir.parent

load_dotenv(backend_dir / ".env")
load_dotenv(root_dir / ".env")


class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "medicalassistant"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL: str = "openai/gpt-oss-20b"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 3600
    UPLOAD_DIR: str = "upload"
    API_PORT: int = 8000


    model_config = SettingsConfigDict(
        env_file=(str(backend_dir / ".env"), str(root_dir / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )



settings = Settings()
