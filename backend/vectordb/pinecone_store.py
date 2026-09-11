import os,sys,uuid
from typing import List
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from logger import logging
from exception import MedicalAssistantException
from embeddings.embedder import Embedder
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


class PineconeStore:
    def __init__(self,index_name:str="medicalassistant"):
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # pinecone client
        self.pc = Pinecone(api_key=api_key)

        self.index_name = "medicalassistant"

        # create index if not exists
        if self.index_name not in self.pc.list_indexes().names():

            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

        logging.info(f"Connected to Pinecone index: {index_name}")

    def store_document(self,chunks:List[Document]):
        try:
            ids = [str(uuid.uuid4()) for _ in chunks]
            for chunk in chunks:
                chunk.metadata["chunk_length"] = len(chunk.page_content)
            self.vector_store.add_documents(
                documents=chunks,
                ids=ids
            )
            logging.info(f"Stored {len(chunks)} chunks in Pinecone")
            pass
        except Exception as e:
            logging.error(f"error in storing document in pinecone : {e}")
            raise MedicalAssistantException(e,sys)
