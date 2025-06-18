import sys
import os
import logging
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from core.rag_pipeline import RAGPipeline
from parser.html_parser import HTMLParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyber-rag")


# ──────────────────────────────────────────────────────────────


def _display_results(console: Console, results: list[dict], url: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        Panel(f"[bold white]🔗 Blog URL:[/] {url}", subtitle=f"🕒 Run at {timestamp}", style="cyan")
    )

    for item in results:
        question = item.get("question", "Unknown question")
        answer = item.get("rag_answer", "No answer")
        console.print(Panel(f"[bold cyan]Q:[/] {question}\n\n[bold green]A:[/] {answer}", border_style="bright_black"))

# ──────────────────────────────────────────────────────────────
@click.command()
@click.option("--url", help="URL of the blog post to run the RAG pipeline on.")
def extract(url: Optional[str]):
    console = Console()

    if not url:
        url = input("Enter a blog URL: ")

    try:
        parser = HTMLParser(url, use_ocr=False)
        content = parser.get_textual_content()

        pipeline = RAGPipeline(prompt_name="extract_qa")

        answers = pipeline.run_all(content)

        _display_results(console, answers, url)

    except Exception as e:
        logger.error(f"[bold red]Error:[/] {str(e)}")




if __name__ == "__main__":
    extract()
