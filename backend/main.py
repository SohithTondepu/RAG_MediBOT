import os
import sys
from pathlib import Path

# Add backend directory to sys.path to support launching from root or backend directory
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from logger import logging
from exception import MedicalAssistantException
from loader.pdf_loader import DocumentLoader
from loader.catalog import DocumentCatalog
from core.chunker import TextChunker
from retrieval.pinecone_store import PineconeStore
from rag.pipeline import RAGPipeline
from schemas.rag import IngestURLRequest, QueryRequest, QueryResponse


app = FastAPI(
    title="MediBot RAG API",
    description="Enterprise Multi-Document Medical RAG Engine with Hybrid Search & Security Guardrails",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
doc_catalog = DocumentCatalog()

upload_dir = Path(settings.UPLOAD_DIR).resolve()
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static-files", StaticFiles(directory=str(upload_dir)), name="static_files")



@app.get("/")
async def root():
    return {"message": "Medical Assistant API running", "version": "2.0.0"}


# ---------------- DOCUMENT INGESTION & CATALOG ---------------- #

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), category: Optional[str] = "general"):
    try:
        file_path = upload_dir / file.filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logging.info(f"File saved: {file_path}")

        document_loader = DocumentLoader(file_path=file_path)
        documents = document_loader.load_documents()
        chunker = TextChunker()
        chunks = chunker.chunk_documents(documents)

        doc_id, is_new = doc_catalog.register_document(
            file_path=file_path,
            filename=file.filename,
            category=category,
            source_url=str(file_path),
            chunks_count=len(chunks)
        )

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = file.filename
            chunk.metadata["category"] = category.lower()

        store = PineconeStore()
        store.store_document(chunks)
        rag.retriever.bm25_retriever.add_documents(chunks)

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and cataloged successfully",
                "doc_id": doc_id,
                "chunks_created": len(chunks)
            }
        )

    except Exception as e:
        logging.error(str(e))
        raise MedicalAssistantException(e, sys)


@app.post("/ingest-url/")
async def ingest_url(request: IngestURLRequest):
    try:
        file_path = doc_catalog.download_url(request.url, upload_dir)

        document_loader = DocumentLoader(file_path=file_path)
        documents = document_loader.load_documents()
        chunker = TextChunker()
        chunks = chunker.chunk_documents(documents)

        doc_id, is_new = doc_catalog.register_document(
            file_path=file_path,
            filename=file_path.name,
            category=request.category,
            source_url=request.url,
            chunks_count=len(chunks)
        )

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = file_path.name
            chunk.metadata["category"] = request.category.lower()

        store = PineconeStore()
        store.store_document(chunks)
        rag.retriever.bm25_retriever.add_documents(chunks)

        return JSONResponse(
            status_code=200,
            content={
                "message": "URL document ingested and cataloged successfully",
                "doc_id": doc_id,
                "filename": file_path.name,
                "chunks_created": len(chunks)
            }
        )

    except Exception as e:
        logging.error(str(e))
        raise MedicalAssistantException(e, sys)


@app.get("/catalog/")
async def get_catalog():
    return doc_catalog.get_catalog()


@app.delete("/catalog/{doc_id}")
async def delete_catalog_doc(doc_id: str):
    try:
        store = PineconeStore()
        store.delete_document_vectors(doc_id)
        rag.retriever.bm25_retriever.remove_document_by_id(doc_id)
        meta = doc_catalog.remove_document(doc_id)
        if meta:
            return {"message": f"Successfully deleted document '{meta.get('filename')}'", "doc_id": doc_id}
        return JSONResponse(status_code=404, content={"message": "Document ID not found"})
    except Exception as e:
        logging.error(str(e))
        raise MedicalAssistantException(e, sys)


@app.get("/check-file/{filename}")
async def check_file_exists(filename: str):
    target_path = upload_dir / filename
    return target_path.exists()


# ---------------- ASK QUESTION ---------------- #

@app.post("/ask/", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        answer, top_k_chunks = rag.ask(
            question=request.question,
            session_id=request.session_id,
            file_name=request.file_name,
            category=request.category
        )

        sources = []
        for doc in top_k_chunks:
            page_num = doc.metadata.get("page")
            if page_num is not None:
                sources.append(f"page: {page_num + 1 if isinstance(page_num, int) and page_num == 0 else page_num}")
            else:
                src_name = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "document"))
                sources.append(f"file: {src_name}")

        chunks_text = [doc.page_content for doc in top_k_chunks]

        return QueryResponse(answer=answer, sources=sources, chunks_text=chunks_text)

    except Exception as e:
        raise MedicalAssistantException(e, sys)
