import json
import csv
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
    i = 0
    for idx, ex in enumerate(examples):
        article = ex["content"]
        question = ex["question"]
        try:
            rag_answer = pipeline.run(article, question).strip()
        except Exception as e:
            rag_answer = f"Error: {str(e)}"
        i += 1
        if i == 20:
            break
        print(f"\n--- Example {idx + 1} ---")
        print(f"Question: {question}")
        print(f"Ground Truth: {ex['ground_truth_answer']}")
        print(f"RAG Answer: {rag_answer}")

        results.append({
            "article": article,
            "question": question,
            "ground_truth_answer": ex["ground_truth_answer"],
            "rag_answer": rag_answer
        })
    return results


def save_answers_only(results, json_path="answers.json", csv_path="answers.csv"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth_answer", "rag_answer"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "question": row["question"],
                "ground_truth_answer": row["ground_truth_answer"],
                "rag_answer": row["rag_answer"]
            })

import re

def evaluate_answer_with_watsonx(client, question: str, ground_truth: str, rag_answer: str):
    """Uses Watsonx to compare the RAG answer with the ground truth answer."""
    prompt = f"""
You are an expert evaluator.
Given a question, a ground truth answer, and a generated answer, your task is to rate how accurate the generated answer is.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG Answer: {rag_answer}

Please reply in the following JSON format:
{{
  "evaluation": "Correct" | "Partially Correct" | "Incorrect",
  "grade": number between 0.0 and 1.0,
  "explanation": "short explanation here"
}}
"""

    try:
        response = client.invoke(prompt)
        text = response.content.strip()

        # Try to extract JSON block manually
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_block = match.group(0)
            return json.loads(json_block)

        # fallback: return as explanation only
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


def run_watsonx_evaluation_on_saved_answers(
    json_path="answers.json",
    output_path="answers_with_eval.json",
    model_name="meta-llama/llama-3-3-70b-instruct"
):
    client = get_chat_llm_client(model_name=model_name)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluated = []
    for idx, item in enumerate(results):
        eval_result = evaluate_answer_with_watsonx(
            client,
            item["question"],
            item["ground_truth_answer"],
            item["rag_answer"]
        )

        print(f"\n--- Evaluation {idx + 1} ---")
        print(f"Question: {item['question']}")
        print(f"Evaluation: {eval_result['evaluation']}")
        print(f"grade: {eval_result['grade']}")
        print(f"Explanation: {eval_result['explanation']}")

        item.update(eval_result)
        evaluated.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluated, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All evaluations saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Step 1: Load examples
    path = "dataset.jsonl"
    examples = extract_examples_from_jsonl(path)

    # Step 2: Run RAG once
    from src.core.rag_pipeline import RAGPipeline  # replace with your actual import
    pipeline = RAGPipeline()
    rag_results = run_rag_on_examples(examples, pipeline)

    # Step 3: Save RAG output to JSON and CSV
    save_answers_only(rag_results)

    # Step 4: Run WatsonX evaluation (later or now)
    run_watsonx_evaluation_on_saved_answers()
