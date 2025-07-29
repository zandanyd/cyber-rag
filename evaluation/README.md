# 🧪 RAG Evaluation

This folder contains all files used to evaluate the effectiveness of the RAG system.

## Files

- `blogs_with_questions_and_answers.jsonl`: The dataset containing blog URLs, questions, and ground-truth answers.
- `answers_with_rag.json`: Output and evaluation of answers using RAG.
- `answers_without_rag.json`: Output and evaluation of answers without RAG.
- `evaluate_model_answers.py`: Script to compare model answers to ground truth using a large LLM (LLaMA 3 70B).
- `analyze_rag_evaluation.ipynb`: Jupyter notebook for comparing performance metrics and generating plots.

## Comparison of RAG vs No-RAG Performance
This experiment evaluates the effectiveness of incorporating Retrieval-Augmented Generation (RAG) in answering predefined cybersecurity questions from blog posts.

**Experiment Setup:**

1. With RAG: The model answers questions based on relevant passages retrieved from the blog (retrieval-augmented).

2. Without RAG: The model answers using the full blog content directly, without retrieval or context filtering.

3. Both setups were executed using the mistralai/mistral-small-3-1-24b-instruct-2503 model, a compact 7.3 billion parameter LLM.

4. The dataset includes 35 blogs and 350 predefined questions, ensuring a consistent and fair evaluation across both methods.

5. Answers were graded using meta-llama/llama-3-70b-instruct (70B parameters), assigning a grade from 0 (incorrect) to 1 (correct) with partial values indicating partly correct answers.

6. The evaluation emphasizes answer correctness (not exact match to ground truth), using a WatsonX-based grading system.

This comparison demonstrates whether small, efficient models like Mistral can benefit from RAG to improve performance and reduce the dependency on large, expensive LLMs.
