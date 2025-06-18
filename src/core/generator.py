import os
import ollama
from dotenv import load_dotenv
from src.llm_api import get_chat_llm_client  # Your WatsonX API wrapper

load_dotenv()

class LLMGenerator:
    def __init__(self, prompt_name="extract_qa"):
        """
        Initialize the generator with the selected LLM provider and model from .env
        """
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.model = os.getenv("LLM_MODEL", "mistral")
        self.prompt = self.get_prompt(prompt_name)

        if self.provider == "watsonx":
            self.client = get_chat_llm_client(model_name=self.model)

    def build_prompt(self, question: str, context: str) -> str:
        return self.prompt.format(context=context.strip(), question=question.strip())

    def generate_answer(self, question: str, context: str) -> str:
        prompt = self.build_prompt(question, context)

        if self.provider == "watsonx":
            return self._generate_with_watsonx(prompt)
        else:
            return self._generate_with_ollama(prompt)

    def _generate_with_ollama(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a cybersecurity assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()

    def _generate_with_watsonx(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.content.strip()

    def get_prompt(self, name: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.abspath(os.path.join(current_dir, ".."))
        prompt_path = os.path.join(src_dir, "prompts", f"{name}.txt")

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
