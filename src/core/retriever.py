from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
import faiss
from chonkie import RecursiveChunker
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)



class RAGRetriever:
    def __init__(self, model_name = 'multi-qa-MiniLM-L6-cos-v1'):
        self.chunker = RecursiveChunker(chunk_size=512)
        self.embedder = SentenceTransformer(model_name)
        self.chunks = []
        self.index = None

    def prepare_index(self, text: str):
        self.chunks = self.chunker.chunk(text)  # use .chunk() method
        if isinstance(self.chunks[0], str):
            to_embed = self.chunks
        else:
            to_embed = [chunk.text for chunk in self.chunks]

        embeddings = self.embedder.encode(to_embed, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def query(self, question: str, threshold: float = 0.2, top_k: int = 4):
        if self.index is None:
            return ["Index not initialized."]

        # Embed the query
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

        # Search top max_k results
        scores, indices = self.index.search(q_vec, top_k)

        # Filter by threshold
        filtered_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                filtered_chunks.append(self.chunks[idx])

        # Fallback: return at least one chunk if none pass threshold
        if not filtered_chunks and top_k > 0:
            filtered_chunks.append(self.chunks[indices[0][0]])

        return filtered_chunks