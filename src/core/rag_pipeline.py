from src.core.retriever import RAGRetriever
from src.core.generator import LLMGenerator
from src.questions.predefined_questions import load_predefined_questions


class RAGPipeline:
    def __init__(self,
                 embed_model: str = 'multi-qa-MiniLM-L6-cos-v1',
                 tokenizer_name: str = 'bert-base-uncased',
                 llm_model: str = 'phi3',
                 prompt_name: str = 'extract_qa',
                 top_k = 4):

        self.retriever = RAGRetriever(model_name=embed_model)
        self.generator = LLMGenerator(prompt_name=prompt_name)
        self.top_k = top_k

        q_data = load_predefined_questions()
        self.questions = q_data["analyst_questions"]
        self.queries = q_data["analyst_queries"]
        if len(self.questions) != len(self.queries):
            raise ValueError("Mismatch between number of questions and queries.")


    def run_all(self, blog_text: str) -> list[dict]:
        self.retriever.prepare_index(blog_text)
        results = []

        for i, (question, query) in enumerate(zip(self.questions, self.queries), 1):
            retrieved_chunks = self.retriever.query(query, top_k=self.top_k)
            context = "\n".join(
                chunk.text if hasattr(chunk, "text") else chunk for chunk in retrieved_chunks
            )
            answer = self.generator.generate_answer(question, context)
            results.append({
                "question_id": i,
                "question": question,
                "retrieval_query": query,
                "retrieved_context": context,
                "rag_answer": answer
            })

        return results