import os,sys
from logger import logging
from exception import MedicalAssistantException
from retriever.pinecone_retriever import PineconeRetriever
from llm.llm_groq import load_llm


class RAGPipeline:
    def __init__(self):
        logging.info("Initialising RAG Pipeline")
        self.retriever=PineconeRetriever()
        self.llm=load_llm()
        self.chat_history=[]

    def _rewrite_question(self, current_question):
        if not self.chat_history:
            return current_question
        
        # Create a prompt to rewrite the question
        history_text = ""
        for q, a in self.chat_history[-2:]:  # Last 2 exchanges for context
            history_text += f"User: {q}\nAssistant: {a}\n"
        
        rewrite_prompt = f"""
            Given the conversation history and a new question, rewrite the new question 
            to be a standalone question that contains all necessary context.
            
            Conversation History:
            {history_text}
            
            New Question: {current_question}
            
            Rewritten Question (make it specific and include context from history):
        """
        
        try:
            # Use a smaller/faster model for rewriting (or same LLM)
            response = self.llm.invoke(rewrite_prompt)
            rewritten = response.content.strip()
            
            logging.info(f"Original: '{current_question}' → Rewritten: '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logging.warning(f"Question rewriting failed: {e}. Using original.")
            return current_question
    
    def ask(self,question):
        try:
            logging.info("Answering user question")

            rewritten_question = question # self._rewrite_question(question)


            top_k_chunks = self.retriever.retrieve(rewritten_question)
        


            context = "\n\n".join([doc.page_content for doc in top_k_chunks])
            
            print("chat history: ",self.chat_history)
            history_text = ""
            # if self.chat_history:
            #     recent_history=self.chat_history[-3:]
            #     for q,a in recent_history:
            #         history_text += f"User: {q}\nAssistant: {a}\n"

            prompt = f"""
                You are **MediBot**, an AI-powered medical assistant.
                
                {f"📜 **Previous Conversation**:{history_text}" if history_text else ""}
                
                🔍 **Medical Context**:
                {context}
                
                🙋‍♂️ **Current Question**:
                {question}
                
                ---
                💬 **Instructions**:
                - If there's previous conversation, use it to understand context
                - Answer ONLY using the provided medical context
                - If answer not in context, say: "I could not find this information"
                - Keep answers concise
            """
            
            response=self.llm.invoke(prompt)
            answer=response.content
            # self.chat_history.append((question,answer))
            logging.info("Answer generated successfully")

            return answer,top_k_chunks
        except Exception as e:
            raise MedicalAssistantException(e,sys)

