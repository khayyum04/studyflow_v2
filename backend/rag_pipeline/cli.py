from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import click
from dotenv import load_dotenv

# These are noisy but harmless — env/dependency notices, not bugs in this codebase.
# Filtered before any command's lazy imports pull in the packages that emit them.
warnings.filterwarnings("ignore", message=".*Python version 3.9 past its end of life.*")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# Load ANTHROPIC_API_KEY from the .env file in the project root before importing anything that uses it
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from ..textbook_extraction.batch import run_extraction
from ..textbook_extraction.config import TEXTBOOKS
from ..textbook_extraction.textbook import TextbookSection, load_sections, parse_section_arg


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--textbook", type=click.Choice(list(TEXTBOOKS.keys())), default=None,
              help="Restrict to one textbook.")
@click.option("--section", "section_arg", default=None,
              help="Single section, e.g. sejarah_t4/bab04/sec_4_2")
@click.option("--dry-run", is_flag=True, help="Show what would be extracted without calling the API.")
@click.option("--force", is_flag=True, help="Re-extract sections that already have .md files.")
def extract(textbook: str | None, section_arg: str | None, dry_run: bool, force: bool) -> None:
    """Extract textbook pages to section-level markdown files via the Batch API."""
    # Figure out which sections to work on based on the flags provided
    if section_arg:
        try:
            sections = [parse_section_arg(section_arg)]
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        textbook_ids = [textbook] if textbook else None
        sections = load_sections(textbook_ids)

    if dry_run:
        _print_dry_run(sections, force)
        return

    run_extraction(sections, force=force)


@cli.command()
@click.option("--textbook", type=click.Choice(list(TEXTBOOKS.keys())), default=None,
              help="Restrict to one textbook.")
def chunk(textbook: str | None) -> None:
    """Chunk extracted section markdown files into retrieval-ready chunks."""
    from .chunking import run_chunking
    run_chunking([textbook] if textbook else None)


@cli.command()
def embed() -> None:
    """Embed chunks and (re)build the ChromaDB collection."""
    from .embedding import run_embedding
    run_embedding()


@cli.command("query")
@click.argument("text")
@click.option("--k", default=5, help="Number of results to return.")
def query_cmd(text: str, k: int) -> None:
    """Query the embedding collection and print the top-k matching chunks."""
    from .retrieval import Retriever
    retriever = Retriever()
    for hit in retriever.retrieve(text, k=k):
        meta = hit["metadata"]
        click.echo(
            f"{hit['chunk_id']:<40}  score={hit['score']:.4f}  "
            f"{meta['chapter_title']} > {meta['section_title']}"
        )


@cli.command()
@click.argument("text")
@click.option("--k", default=5, help="Number of chunks to retrieve as context.")
def ask(text: str, k: int) -> None:
    """Ask a question and get a RAG-generated answer with its source."""
    from .generation import answer_question
    answer = answer_question(text, k=k)
    click.echo(answer.text)
    if answer.source:
        s = answer.source
        click.echo(f"\nSource: {s.chapter_title} > {s.section_title} (m.s. {s.page})")


@cli.command()
@click.option("--n", default=100, help="Number of eval questions to sample.")
@click.option("--seed", default=0, help="Random seed for sampling (keep matching --seed on `eval`).")
def baseline(n: int, seed: int) -> None:
    """Run the Gemini + Google Search grounded no-context baseline eval and save results."""
    from .eval_set_mcq.eval_baseline import run_baseline
    run_baseline(n=n, seed=seed)


@cli.command("eval")
@click.option("--n", default=100, help="Number of eval questions to sample.")
@click.option("--seed", default=0, help="Random seed for sampling (keep matching --seed on `baseline`).")
@click.option("--k", default=5, help="Number of chunks to retrieve as context per question.")
def eval_cmd(n: int, seed: int, k: int) -> None:
    """Run the RAG eval (Gemini for generation) and save results."""
    from .eval_set_mcq.eval_rag import run_eval
    run_eval(n=n, seed=seed, k=k)


@cli.command()
def analyze() -> None:
    """Compare baseline vs RAG eval results."""
    from .eval_set_mcq.eval_analyze import run_analyze
    run_analyze()


def _print_dry_run(sections: list[TextbookSection], force: bool) -> None:
    # Shows which sections would be submitted without actually hitting the API
    pending = [s for s in sections if force or not s.is_extracted()]
    done = len(sections) - len(pending)

    click.echo(f"\nPending sections ({len(pending)} of {len(sections)} total):\n")
    for s in pending:
        n_pages = s.page_end - s.page_start + 1
        rel_out = s.output_path.relative_to(s.output_path.parents[3])
        click.echo(f"  {s.custom_id:<40}  pages {s.page_start}-{s.page_end} ({n_pages}p)  →  {rel_out}")

    if done:
        click.echo(f"\n{done} already extracted (skipped). Use --force to re-extract.")
    click.echo()


if __name__ == "__main__":
    cli()
