from typing import List, Optional
from pydantic import BaseModel, Field


class IngestURLRequest(BaseModel):
    url: str = Field(..., description="Document download URL")
    category: Optional[str] = Field("general", description="Document categorization tag")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User query or clinical question")
    session_id: Optional[str] = Field(None, description="Multi-turn conversation session ID")
    file_name: Optional[str] = Field(None, description="Filter search to specific document")
    category: Optional[str] = Field(None, description="Filter search to specific category")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated markdown answer")
    sources: Optional[List[str]] = Field(default=[], description="List of source file names or page numbers")
    chunks_text: Optional[List[str]] = Field(default=[], description="Raw text of top-k retrieved chunks")


class DocumentCatalogItem(BaseModel):
    doc_id: str
    filename: str
    file_path: str
    file_hash: str
    category: str
    source_url: str
    chunks_count: int
    created_at: float
