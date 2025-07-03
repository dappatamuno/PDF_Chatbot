from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def split_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

class PDFChatbot:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, text_chunks):
        self.chunks = text_chunks
        embeddings = self.model.encode(text_chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def retrieve_answer(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), top_k)
        return [self.chunks[i] for i in I[0]]
