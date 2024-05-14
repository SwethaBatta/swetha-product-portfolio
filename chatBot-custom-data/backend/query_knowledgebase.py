import openai
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import fitz

class ChatBot:
    def __init__(self, api_key):
        # Configure your OpenAI API key
        openai.api_key = api_key

        # Initialize BART tokenizer and model
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

        # Load a pre-trained model from sentence-transformers
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load and process PDF documents
        pdf_paths = ["/Users/swethabatta/Desktop/openai-chatBot/openai_chatBotApp/data/FAQ.pdf", "/Users/swethabatta/Desktop/openai-chatBot/openai_chatBotApp/data/SwethaBattaTestDataDocForBills_AEP.pdf", "/Users/swethabatta/Desktop/openai-chatBot/openai_chatBotApp/data/SwethaBattaTestDataDocForBills_AT&T.pdf"]  # Add your PDF file paths here
        
        # List to store document text
        self.documents = []

        # Process each PDF file
        for pdf_path in pdf_paths:
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                self.documents.append(text)

        # Encode and index the documents using FAISS
        d = self.embedding_model.get_sentence_embedding_dimension()  # Dimension of the embeddings
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
        return self.embedding_model.encode(text, convert_to_tensor=True).cpu().numpy()

    def query_knowledge_base(self, query):
        # Generate embeddings for the query
        query_embedding = self.embed_text(query).reshape(1, -1)  # Reshape for FAISS search
        D, I = self.index.search(query_embedding, k=1)  # Get top k documents

        # Retrieve the top documents
        retrieved_docs = [self.documents[i] for i in I[0]]

        # Combine query with retrieved documents
        inputs = self.tokenizer(retrieved_docs, return_tensors="pt", padding=True, truncation=True)

        # Generate a response
        with torch.no_grad():
            output = self.model.generate(**inputs)

        # Decode the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def send_prompt(self, query):
        response = self.query_knowledge_base(query)
        return response
