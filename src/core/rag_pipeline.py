from src.core.retriever import RAGRetriever
from src.core.generator import LLMGenerator


class RAGPipeline:
    def __init__(self,
                 embed_model: str = 'multi-qa-MiniLM-L6-cos-v1',
                 tokenizer_name: str = 'bert-base-uncased',
                 llm_model: str = 'phi3',
                 prompt_name: str = 'extract_qa',
                 top_k = 4):
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
        self.retriever = RAGRetriever(model_name=embed_model)
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
        self.retriever.prepare_index(blog_text)
        retrieved_chunks = self.retriever.query(question, top_k=self.top_k)
        context = "\n".join(
            chunk.text if hasattr(chunk, "text") else chunk for chunk in retrieved_chunks
        )

        answer = self.generator.generate_answer(question, context)
        return answer