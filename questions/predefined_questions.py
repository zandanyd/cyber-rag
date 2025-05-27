import json
import os

def load_predefined_questions(path="questions/questions.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)