import openai
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import fitz

class ChatBot:
    def __init__(self, api_key):
        # Configure your OpenAI API key
        openai.api_key = api_key

        # Initialize BART tokenizer and model
        self.tokenizer = SentenceTransformer('all-MiniLM-L6-v2')

        # List to store document text
        self.documents = []

        # Load and process pdf paths
        pdf_paths = ["data/FAQ.pdf", "data/SwethaBattaTestDataDocForBills_AEP.pdf", "data/SwethaBattaTestDataDocForBills_AT&T.pdf"]
        
        # Process each PDF file
        for pdf_path in pdf_paths:
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                self.documents.append(text)

        # Encode and index the documents using FAISS
        d = self.tokenizer.get_sentence_embedding_dimension()  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(d)

        # Generate embeddings for each document
        for doc in self.documents:
            vectors = self.embed_text(doc).astype('float32')
            self.index.add(np.array([vectors]))

    def extract_text_from_pdf(self, pdf_path):
        try:
            document = fitz.open(pdf_path)
            text = ""
            for page_num in range(document.page_count):
                page = document.load_page(page_num)
                text += page.get_text()
            return text
        except fitz.fitz.FileNotFoundError:
            print(f"File not found: {pdf_path}")
            return ""

    def embed_text(self, text):
        return self.tokenizer.encode(text, convert_to_tensor=True).cpu().numpy()

    def query_knowledge_base(self, query):
        # Generate embeddings for the query
        query_embedding = self.embed_text(query).reshape(1, -1)  # Reshape for FAISS search
        D, I = self.index.search(query_embedding, k=1)  # Get top k documents

        # Retrieve the top documents
        retrieved_docs = [self.documents[i] for i in I[0]]

        # Combine query with retrieved documents to form the context
        context = "\n\n".join(retrieved_docs)

        # Generate a response using OpenAI's API
        response = openai.Completion.create(
            engine="gpt-4-turbo",  # Use the appropriate model name
            prompt=f"{context}\n\n{query}",
            max_tokens=150
        )

        return response.choices[0].text.strip()

    def send_prompt(self, query):
        response = self.query_knowledge_base(query)
        return response