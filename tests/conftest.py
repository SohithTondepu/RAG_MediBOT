import os
import sys
from pathlib import Path

# Add backend directory to sys.path so tests can import all backend packages cleanly
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Mock default env vars for testing if missing
os.environ.setdefault("PINECONE_API_KEY", "mock_pinecone_key_for_testing")
os.environ.setdefault("GROQ_API_KEY", "mock_groq_key_for_testing")
