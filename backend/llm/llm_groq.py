from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from logger import logging
from exception import MedicalAssistantException
import os,sys

def load_llm():
    try:
        logging.info("Loading LLM")
        groq_api_key=os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant",temperature=0.1,max_tokens=1024)
        logging.info("llm model loaded successfully")
        return llm
    except Exception as e:
        raise MedicalAssistantException(e,sys)


# from langchain_ollama import ChatOllama
# from logger import logging
# from exception import MedicalAssistantException
# import os,sys


# def load_llm():
#     try:
#         logging.info("Loading LLM")
#         model_name=os.getenv("OLLAMA_MODEL")
#         ollama_api_key=os.getenv("OLLAMA_API_KEY")
#         if not ollama_api_key:
#             raise ValueError("OLLAMA_API_KEY is not set")
#         if not model_name:
#             raise ValueError("OLLAMA_MODEL is not set")
#         llm = ChatOllama(
#             model=model_name,
#             temperature=0.1,
#             base_url="https://ollama.com",
#             headers={
#                 "Authorization": f"Bearer {ollama_api_key}"
#             }
#         )

#         logging.info("llm model loaded successfully")
#         return llm

#     except Exception as e:
#         raise MedicalAssistantException(e,sys)