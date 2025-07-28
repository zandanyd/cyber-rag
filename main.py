import sys
import os
import logging
from datetime import datetime
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich import box
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.core.rag_pipeline import RAGPipeline
from src.parser.html_parser import HTMLParser

# ──────────────────────────────────────────────────────────────
# Logging setup
console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cyber-rag")
# ──────────────────────────────────────────────────────────────


def _banner():
    banner_text = """
    [bold blue]
     ██████╗██╗   ██╗██████╗ ███████╗██████╗     ██████╗  █████╗  ██████╗ 
    ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██╔══██╗██╔════╝ 
    ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝    ██████╔╝███████║██║  ███╗
    ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗    ██╔══██╗██╔══██║██║   ██║
    ╚██████╗   ██║   ██████╔╝███████╗██║  ██║    ██║  ██║██║  ██║╚██████╔╝
     ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
    [/bold blue]
    [italic white]   Extracting insights from cybersecurity blogs using RAG 🔍[/italic white]
        """
    console.print(banner_text)

# ──────────────────────────────────────────────────────────────

def _display_results(console: Console, results: list[dict], url: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        Panel(
            f"[bold white] Blog URL:[/] [cyan]{url}[/]",
            subtitle=f"[dim] Run at {timestamp}[/]",
            style="magenta",
            box=box.ROUNDED
        )
    )

    for item in results:
        question = item.get("question", "Unknown question")
        answer = item.get("rag_answer", "No answer")
        console.print(Panel(f"[bold cyan]Q:[/] {question}\n\n[bold green]A:[/] {answer}", border_style="bright_black"))

# ──────────────────────────────────────────────────────────────
@click.command()
@click.option("--url", help="URL of the blog post to run the RAG pipeline on.")
def extract(url: Optional[str]):
    _banner()
    if not url:
        console.print("[yellow]No URL provided. Please enter one manually:[/]")
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
