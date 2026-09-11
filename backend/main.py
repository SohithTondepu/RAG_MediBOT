import os
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
from dotenv import load_dotenv

from logger import logging
from exception import MedicalAssistantException

from loader.pdf_loader import DocumentLoader
from chunking.text_chunker import TextChunker
from vectordb.pinecone_store import PineconeStore
from rag.rag_pipeline import RAGPipeline
from fastapi.staticfiles import StaticFiles


load_dotenv()

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()


@app.get("/")
async def root():
    return {"message": "Medical Assistant API running"}


# ---------------- PDF Upload ---------------- #

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        upload_dir = "./upload"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = Path(upload_dir) / file.filename
        

        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logging.info(f"File saved: {file_path}")
        print("filepath in upload: ",file_path)

        # -------- PIPELINE -------- #

        document_loader=DocumentLoader(file_path=file_path)
        documents=document_loader.load_documents()
        chunker=TextChunker()
        chunks=chunker.chunk_documents(documents)
        store=PineconeStore()
        store.store_document(chunks)

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and indexed successfully",
                "chunks_created": len(chunks)
            }
        )

    except Exception as e:
        logging.error(str(e))
        raise MedicalAssistantException(e, sys)


upload_dir = Path("upload")

app.mount("/upload", StaticFiles(directory=str(upload_dir)), name="upload")
# ---------------- ASK QUESTION ---------------- #

class QueryRequest(BaseModel):
    question: str
    


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]]
    chunks_text:Optional[List[str]]


@app.post("/ask/", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    try:
        answer,top_k_chunks = rag.ask(request.question)

        sources = [f"page: {doc.metadata['page']}" for doc in top_k_chunks]
        chunks_text=[doc.page_content for doc in top_k_chunks]

        return QueryResponse(answer=answer,sources=sources,chunks_text=chunks_text)

    except Exception as e:
        raise MedicalAssistantException(e, sys)





# if __name__ == "__main__":
#     # document_loader=DocumentLoader("upload")
#     # documents=document_loader.load_documents()
#     # chunker=TextChunker()
#     # chunks=chunker.chunk_documents(documents)
#     # store=PineconeStore()
#     # store.store_document(chunks)
#     rag=RAGPipeline()
#     response=rag.ask(question="headache")
#     print("\n\n")
#     print("="*60)
#     print(response)
    



