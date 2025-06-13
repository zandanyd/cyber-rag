import json
from dotenv import load_dotenv

from src.llm_api import get_chat_llm_client

load_dotenv()


def extract_examples_from_jsonl(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            article = data.get("outputs", {}).get("article_textual_content")
            qna_list = data.get("outputs", {}).get("qna", [])
            if article and qna_list:
                for pair in qna_list:
                    question = pair.get("question")
                    answer = pair.get("answer")
                    if question and answer:
                        examples.append({
                            "content": article.strip(),
                            "question": question.strip(),
                            "ground_truth_answer": answer.strip()
                        })
    return examples


def run_rag_on_examples(examples, pipeline):
    results = []

    for ex in examples:
        article = ex["article"]
        question = ex["question"]

        try:
            rag_answer = pipeline.run(article, question).strip()
        except Exception as e:
            rag_answer = f"Error: {str(e)}"

        results.append({
            "question": question,
            "article": article,
            "ground_truth_answer": ex["ground_truth_answer"],
            "rag_answer": rag_answer,
        })

    return results


def evaluate_answer_with_watsonx(client, question: str, ground_truth: str, rag_answer: str) -> str:
    """Uses Watsonx to compare the RAG answer with the ground truth answer."""
    prompt = f"""You are an expert evaluator. 
    Given a question, a ground truth answer, and a generated answer, your task is to rate how accurate the generated answer is.
    
    Question: {question}
    
    Ground Truth Answer: {ground_truth}
    
    RAG Answer: {rag_answer}
    
    Please reply with one of the following: "Correct", "Partially Correct", or "Incorrect", and a short explanation."""

    try:
        response = client.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def run_full_rag_evaluation(examples, rag_pipeline, model_name="meta-llama/llama-3-3-70b-instruct"):
    watsonx_client = get_chat_llm_client(model_name=model_name)

    results = []
    for ex in examples:
        article = ex["content"]
        question = ex["question"]
        ground_truth = ex["ground_truth_answer"]

        try:
            rag_answer = rag_pipeline.run(article, question).strip()
        except Exception as e:
            rag_answer = f"Error: {str(e)}"

        evaluation = evaluate_answer_with_watsonx(
            client=watsonx_client,
            question=question,
            ground_truth=ground_truth,
            rag_answer=rag_answer
        )

        results.append({
            "question": question,
            "article": article,
            "ground_truth_answer": ground_truth,
            "rag_answer": rag_answer,
            "watsonx_evaluation": evaluation
        })

    return results



if __name__ == "__main__":
    # Step 1: Get the LLM client
    model_name = "meta-llama/llama-3-3-70b-instruct"
    client = get_chat_llm_client(model_name=model_name)

    # Step 2: Provide a test question + answers
    question = "What is the capital of France?"
    ground_truth = "The capital of France is Paris."
    rag_answer = "Paris is the capital city of France."

    # Step 3: Call the evaluator
    print("Sending test request to Watsonx...")
    result = evaluate_answer_with_watsonx(client, question, ground_truth, rag_answer)

    # Step 4: Output the response
    print("\n--- Watsonx Evaluation Result ---")
    print(result)