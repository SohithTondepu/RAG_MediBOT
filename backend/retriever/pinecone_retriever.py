from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from logger import logging
from exception import MedicalAssistantException
import os,sys


class PineconeRetriever:

    def __init__(self):
        
        logging.info("Starting retrieval")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index = pc.Index("medicalassistant")

        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings
        )
        logging.info("intialised retriever with top k as 9 similaryity search")
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )


    def retrieve(self, query):

        try:
            logging.info("Retrieving user query from pinecone")
            docs = self.retriever.invoke(query)

            return docs
        except Exception as e:
            raise MedicalAssistantException(e,sys)
