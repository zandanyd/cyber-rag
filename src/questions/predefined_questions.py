import json
import os


def load_predefined_questions(path=None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "questions.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)