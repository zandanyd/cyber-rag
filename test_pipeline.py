from rag.rag_pipeline import RAGPipeline
from questions.predefined_questions import load_predefined_questions

# Load questions
questions = load_predefined_questions()

# Sample blog content (replace with real content or load from file)
blog_text = """
The malware installs a malicious Chrome extension that captures browser session cookies and forwards them to an external C2 server.
It also contains a credential stealer targeting banking sites, and sends the stolen data to hxxp://bad.example.com/steal.
"""

# Initialize pipeline
pipeline = RAGPipeline(llm_model='phi3', prompt_name='extract_qa')

# Run through all analyst questions
for question in questions["analyst_questions"]:
    print(f"\n🔹 Question: {question}")
    answer = pipeline.run(blog_text, question)
    print(f"✅ Answer: {answer}")
