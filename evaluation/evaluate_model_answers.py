import json
import re
from dotenv import load_dotenv
from src.llm_api import get_chat_llm_client
from src.core.generator import LLMGenerator

load_dotenv()

def extract_examples_from_jsonl(path):
    """Extracts article content, questions, answers, and article URL from dataset."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            article = data.get("outputs", {}).get("article_textual_content")
            article_url = data.get("inputs", {}).get("url") or ""
            qna_list = data.get("outputs", {}).get("qna", [])
            if article and qna_list:
                for pair in qna_list:
                    question = pair.get("question")
                    answer = pair.get("answer")
                    if question and answer:
                        examples.append({
                            "article_url": article_url,
                            "content": article.strip(),
                            "question": question.strip(),
                            "ground_truth_answer": answer.strip()
                        })
    return examples


def run_with_rag_and_evaluate(examples, pipeline, client):
    """Runs RAG pipeline using run_all and evaluates using WatsonX, assuming same question order."""
    results = []
    article_cache = {}

    # Group by article
    from collections import defaultdict
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex["content"]].append(ex)

    for article_idx, (article, ex_list) in enumerate(grouped.items(), 1):
        print(f"\n=== Processing Article {article_idx} with {len(ex_list)} questions ===")

        try:
            rag_outputs = pipeline.run_all(article)
        except Exception as e:
            print(f" Error running RAG on article {article_idx}: {e}")
            continue

        if len(rag_outputs) != len(ex_list):
            print(f"Mismatch in question count: RAG returned {len(rag_outputs)}, expected {len(ex_list)}")
            continue

        for i, ex in enumerate(ex_list):
            question = ex["question"]
            ground_truth = ex["ground_truth_answer"]
            rag_answer = rag_outputs[i]["rag_answer"]
            article_url = ex["article_url"]

            eval_result = evaluate_answer_with_watsonx(client, question, ground_truth, rag_answer)

            print(f"\nEvaluation {len(results)+1}")
            print(f"Q: {question}")
            print(f"Eval: {eval_result['evaluation']} | Grade: {eval_result['grade']}")

            results.append({
                "article_url": article_url,
                "question": question,
                "ground_truth_answer": ground_truth,
                "rag_answer": rag_answer,
                **eval_result
            })

    return results


def evaluate_answer_with_watsonx(client, question: str, ground_truth: str, rag_answer: str):
    """Uses WatsonX to evaluation RAG answer against ground truth."""
    prompt = f"""
You are an expert evaluator.
Given a question, a ground truth answer, and a generated answer, Your task is to evaluation whether the generated answer (RAG Answer) correctly and fully answers the user's question,
regardless of whether it matches the phrasing or exact content of the Ground Truth Answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG Answer: {rag_answer}

Guidelines:
- Focus only on whether the RAG Answer provides a correct, complete, and relevant response to the Question.
- Do NOT penalize for wording differences or different phrasing from the Ground Truth.
- If the RAG Answer fully and accurately answers the Question, give it a perfect score (1.0).
- If the RAG Answer partially answers the Question, give it a partial score.
- If the RAG Answer is wrong, irrelevant, or doesn’t answer the Question, give it a low score (0).

Please reply in the following JSON format:
{{
  "evaluation": "Correct" | "Partially Correct" | "Incorrect",
  "grade": number between 0.0 and 1.0,
  "explanation": "short explanation here"
}}
"""
    if rag_answer == ground_truth:
        return {
            "evaluation": "Correct",
            "grade": 1.0,
            "explanation": "perfect match"
        }
    try:
        response = client.invoke(prompt)
        text = response.content.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {
            "evaluation": "Invalid",
            "grade": 0.0,
            "explanation": "Could not parse JSON: " + text
        }
    except Exception as e:
        return {
            "evaluation": "Error",
            "grade": 0.0,
            "explanation": f"Exception: {str(e)}"
        }


def save_final_results(results, output_path="answers_with_rag.json"):
    """Saves all evaluation results to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n Results saved to {output_path}")


def evaluate_with_rag():
    raw_examples = extract_examples_from_jsonl("blogs_with_questions_and_answers.jsonl")

    from src.core.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()

    client = get_chat_llm_client(model_name="meta-llama/llama-3-3-70b-instruct")

    all_results = run_with_rag_and_evaluate(raw_examples, pipeline, client)

    save_final_results(all_results)



def run_without_rag_and_evaluate(examples, llm_generator, eval_client):
    results = []
    for idx, ex in enumerate(examples, 1):
        article = ex["content"]
        question = ex["question"]
        ground_truth = ex["ground_truth_answer"]
        article_url = ex["article_url"]

        try:
            model_answer = llm_generator.generate_answer(question, article)
        except Exception as e:
            print(f"Error generating answer for question {idx}: {e}")
            model_answer = "Error generating answer."

        eval_result = evaluate_answer_with_watsonx(eval_client, question, ground_truth, model_answer)

        print(f"\nEvaluation {idx}")
        print(f"Q: {question}")
        print(f"Eval: {eval_result['evaluation']} | Grade: {eval_result['grade']}")

        results.append({
            "article_url": article_url,
            "question": question,
            "ground_truth_answer": ground_truth,
            "model_answer": model_answer,
            **eval_result
        })

    return results

def evaluate_without_rag():
    raw_examples = extract_examples_from_jsonl("blogs_with_questions_and_answers.jsonl")

    llm_generator = LLMGenerator(prompt_name="extract_qa")  # Uses model from .env
    eval_client = get_chat_llm_client(model_name="meta-llama/llama-3-3-70b-instruct")

    results = run_without_rag_and_evaluate(raw_examples, llm_generator, eval_client)
    save_final_results(results, output_path="answers_without_rag.json")


if __name__ == "__main__":
    evaluate_with_rag()
    # evaluate_without_rag()