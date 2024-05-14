from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from IPython.display import Markdown, display
from llama_index.core import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.prompts import PromptTemplate
import os

class ChatBot:
    def __init__(self, api_key):
        self.documents = SimpleDirectoryReader("./../data").load_data()
        llm = Gemini(model_name="models/gemini-1.0-pro", api_key=api_key)
        embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embed_model)

    def send_prompt(self, prompt, temperature=0.1):
        index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context
        )
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
        )

        qa_template = PromptTemplate(template)

        query_engine = index.as_query_engine(text_qa_template=qa_template)
        response = query_engine.query(prompt)
        return f'{response}\n'