FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies manifest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

ENV PYTHONPATH=/app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
