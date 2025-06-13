import sys
import os
import logging
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add `src/` to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Imports from src
from core.rag_pipeline import RAGPipeline
from parser.html_parser import HTMLParser
from questions.predefined_questions import load_predefined_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyber-rag")




def _display_results(console: Console, results: dict, url: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        Panel(f"[bold white]🔗 Blog URL:[/] {url}", subtitle=f"🕒 Run at {timestamp}", style="cyan")
    )

    for q, a in results.items():
        console.print(Panel(f"[bold cyan]Q:[/] {q}\n\n[bold green]A:[/] {a}", border_style="bright_black"))

# ──────────────────────────────────────────────────────────────
@click.command()
@click.option("--url", help="URL of the blog post to run the RAG pipeline on.")
def extract(url: Optional[str]):
    console = Console()

    if not url:
        url = input("Enter a blog URL: ")

    try:
        # Step 1: Get content
        parser = HTMLParser(url, use_ocr=False)
        content = parser.get_textual_content()

        # Step 2: Load questions and run RAG
        questions = load_predefined_questions()
        pipeline = RAGPipeline(llm_model="phi3", prompt_name="extract_qa")

        results = {}
        for question in questions["analyst_questions"]:
            answer = pipeline.run(content, question)
            results[question] = answer

        # Step 3: Display
        _display_results(console, results, url)

    except Exception as e:
        logger.error(f"[bold red]Error:[/] {str(e)}")

# Entry point
if __name__ == "__main__":
    extract()
