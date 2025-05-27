from rag.retriever import RAGRetriever
from rag.generator import LLMGenerator


class RAGPipeline:
    def __init__(self,
                 embed_model: str = 'multi-qa-MiniLM-L6-cos-v1',
                 tokenizer_name: str = 'bert-base-uncased',
                 llm_model: str = 'phi3',
                 prompt_name: str = 'extract_qa',
                 top_k = 2):
        """
        Initializes the full RAG pipeline:
        - Embedding + chunking using SentenceTransformer
        - Generation using local Ollama model

        Args:
            embed_model (str): Name of the sentence-transformers model.
            tokenizer_name (str): Tokenizer used for chunk token counting.
            llm_model (str): Ollama model to use (e.g., 'mistral', 'phi3').
            prompt_name (str): Prompt template to load (e.g., 'extract_qa').
            top_k (int): Number of most relevant chunks to use as context.
        """
        self.retriever = RAGRetriever(model_name=embed_model, tokenizer_name=tokenizer_name)
        self.generator = LLMGenerator(model=llm_model, prompt_name=prompt_name)
        self.top_k = top_k


    def run(self, blog_text: str, question: str) -> str:
        """
        Executes the full pipeline: chunk → embed → retrieve → generate answer.

        Args:
            blog_text (str): Full blog content as a single string.
            question (str): The question to answer.

        Returns:
            str: The LLM-generated answer.
        """
        # Step 1: Chunk and embed the blog
        chunks = self.retriever.chunk_text(blog_text)
        embeddings = self.retriever.embed_chunks(chunks)
        top_k = self.top_k

        # Step 2: Retrieve top-k relevant chunks
        top_chunks = self.retriever.retrieve(chunks, embeddings, question, top_k)
        context = "\n".join(top_chunks)

        # Step 3: Generate final answer using the retrieved context
        answer = self.generator.generate_answer(question, context)
        return answer