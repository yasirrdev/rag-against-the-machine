"""Expose the command-line interface through Python Fire."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import NoReturn

import fire
from pydantic import ValidationError

from src.batch import evaluate_recall, retrieve_dataset, retrieve_question
from src.indexing import IndexInputError, build_index, load_index, save_index
from src.models import (
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.retrieval import BM25


def index(
    max_chunk_size: int = 2000,
    corpus_path: str = "data/raw",
    index_path: str = "data/processed/index.json",
) -> None:
    """Build a searchable index of the corpus."""
    root = Path.cwd()
    try:
        corpus = (root / corpus_path).resolve()
        inverted_index = build_index(corpus, root, max_chunk_size)
        save_index(inverted_index, Path(index_path))
    except (IndexInputError, OSError, UnicodeError, SyntaxError) as error:
        raise SystemExit(f"Indexing failed: {error}") from None
    count = len(inverted_index.chunks)
    print(f"Indexed {count} chunks. Saved to {index_path}")


def search(
    query: str,
    k: int = 10,
    index_path: str = "data/processed/index.json",
    output_path: str = "data/output/search_results/single/search.json",
) -> None:
    """Print and save up to k ranked sources for one question as JSON."""
    if not isinstance(query, str):
        raise SystemExit("Search failed: query must be text")
    if type(k) is not int or k < 0:
        raise SystemExit("Search failed: k must be a non-negative integer")
    if not isinstance(index_path, str) or not isinstance(output_path, str):
        raise SystemExit("Search failed: paths must be text")

    try:
        indexed = load_index(Path(index_path))
    except (OSError, UnicodeError, JSONDecodeError, ValidationError) as error:
        raise SystemExit(f"Search failed: {error}") from None

    question = UnansweredQuestion(question=query)
    result = retrieve_question(BM25(indexed), question, k)
    payload = StudentSearchResults(search_results=[result], k=k)
    serialized = payload.model_dump_json(indent=2)
    destination = Path(output_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Search failed: {error}") from None
    print(serialized)


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 10,
    index_path: str = "data/processed/index.json",
) -> None:
    """Retrieve sources for each question in a dataset."""
    if type(k) is not int or k < 0:
        raise SystemExit("Search failed: k must be a non-negative integer")
    if not all(isinstance(p, str) and p for p in (
        dataset_path, save_directory, index_path
    )):
        raise SystemExit("Search failed: paths must be non-empty text")
    source = Path(dataset_path)
    destination = Path(save_directory) / source.name
    try:
        if destination.resolve() in {source.resolve(),
                                     Path(index_path).resolve()}:
            raise SystemExit("Search failed: output would replace an input")
        dataset = RagDataset.model_validate_json(source.read_bytes())
        ranker = BM25(load_index(Path(index_path)))
        results = retrieve_dataset(ranker, dataset, k)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            results.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, UnicodeError, JSONDecodeError, ValidationError) as error:
        raise SystemExit(f"Search failed: {error}") from None
    print(f"Saved student_search_results to {destination}")


def answer(query: str, k: int = 10) -> NoReturn:
    """Generate an answer grounded in retrieved sources."""
    raise SystemExit("Answer generation is scheduled for phase 4.")


def answer_dataset(
    student_search_results_path: str, save_directory: str
) -> NoReturn:
    """Generate answers using existing batch search results."""
    raise SystemExit("Batch answer generation is scheduled for phase 4.")


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Measure recall against a reference dataset."""
    if not all(isinstance(p, str) and p for p in (
        student_search_results_path, dataset_path
    )):
        raise SystemExit("Evaluation failed: paths must be non-empty text")
    try:
        results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_bytes()
        )
        reference = RagDataset.model_validate_json(
            Path(dataset_path).read_bytes()
        )
        if results.k < 0:
            raise ValueError("k must be a non-negative integer")
        recall, misses = evaluate_recall(results, reference)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from None
    print(json.dumps({
        **recall,
        "missed_at_5": [{
            "question_id": row.question_id,
            "question": row.question,
            "missing_sources": [
                source.model_dump() for source in row.retrieved_sources
            ],
        } for row in misses],
    }, indent=2))


def main() -> None:
    """Register the command names required by the subject."""
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })


if __name__ == "__main__":
    main()
