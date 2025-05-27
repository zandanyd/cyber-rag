from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from transformers import AutoTokenizer
import faiss

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)



class RAGRetriever:
    def __init__(self, model_name = 'multi-qa-MiniLM-L6-cos-v1', tokenizer_name='bert-base-uncased'):
        self.embedder = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.index = None


    def chunk_text(self, text: str, max_tokens: int = 100, stride: int = 50) -> list:
        sentences = nltk.sent_tokenize(text)
        tokenized_sentences = [self.tokenizer.tokenize(sent) for sent in sentences]

        chunks = []

        i = 0
        while i < len(tokenized_sentences):
            current_chunk = []
            current_length = 0
            start_i = i

            # Add as many sentences as we can without exceeding max_tokens
            while i < len(tokenized_sentences) and current_length + len(tokenized_sentences[i]) <= max_tokens:
                current_chunk.append(sentences[i])
                current_length += len(tokenized_sentences[i])
                i += 1

            chunks.append(" ".join(current_chunk))

            # Move back by stride tokens (approx. by sentences)
            i = max(start_i + 1, i - stride // 10)  # estimate ~10 tokens per sentence

        return chunks

    def embed_chunks(self, chunks: list) -> np.ndarray:
        return self.embedder.encode(chunks, convert_to_numpy=True)

    def retrieve(self, chunks: list, chunk_embeddings: np.ndarray, question: str, top_k: int = 3) -> list:
        # Normalize chunk embeddings for cosine similarity
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

        # Build cosine similarity FAISS index (Inner Product on unit vectors)
        dim = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(chunk_embeddings)

        # Normalize query embedding
        query_embedding = self.embedder.encode([question], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search top-k
        scores, indices = self.index.search(query_embedding, top_k)
        return [chunks[i] for i in indices[0]]