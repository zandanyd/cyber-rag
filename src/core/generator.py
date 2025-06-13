import ollama
import os

class LLMGenerator:
    def __init__(self, model='mistral', prompt_name = "extract_qa"):
        """
        Initialize the generator with the selected Ollama model.
        """
        self.model = model
        self.prompt = self.get_prompt(prompt_name)


    def build_prompt(self, question: str, context: str) -> str:
        """
        Creates a structured prompt for the LLM using the retrieved context and a fixed question.
        """
        return self.prompt.format(context=context.strip(), question=question.strip())


    def generate_answer(self, question: str, context: str) -> str:
        """
        Sends the structured prompt to the Ollama model and returns the generated response.
        """
        prompt = self.build_prompt(question, context)

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a cybersecurity assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()


    def get_prompt(self, name: str) -> str:

        path = os.path.join("src/prompts", f"{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()
