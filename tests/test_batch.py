"""Verify the dataset CLI and the local recall calculation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.__main__ import search_dataset
from src.batch import evaluate_recall
from src.indexing import InvertedIndex, save_index
from src.models import RagDataset, StudentSearchResults


def test_empty_dataset() -> None:
    """Write a valid empty result with the input filename and requested k."""
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        dataset = root / "questions.json"
        dataset.write_text('{"rag_questions": []}', encoding="utf-8")
        index = root / "index.json"
        save_index(InvertedIndex(chunks=[], postings={}), index)
        output = root / "out"
        search_dataset(str(dataset), str(output), 5, str(index))
        results = json.loads((output / dataset.name).read_bytes())
        assert results == {"search_results": [], "k": 5}


@pytest.mark.parametrize("k", [-1, 1.5, True, "foo"])
def test_invalid_k(k: int) -> None:
    """Reject invalid k before opening a dataset or index."""
    with pytest.raises(SystemExit, match="non-negative integer"):
        search_dataset("missing.json", "out", k)


def test_bad_inputs() -> None:
    """Report missing inputs and malformed JSON without a traceback."""
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        dataset = root / "questions.json"
        output = root / "out"
        index = root / "index.json"
        with pytest.raises(SystemExit, match="questions.json"):
            search_dataset(str(dataset), str(output), 5, str(index))
        dataset.write_text("{", encoding="utf-8")
        with pytest.raises(SystemExit, match="Invalid JSON"):
            search_dataset(str(dataset), str(output), 5, str(index))
        dataset.write_text('{"rag_questions": []}', encoding="utf-8")
        with pytest.raises(SystemExit, match="index.json"):
            search_dataset(str(dataset), str(output), 5, str(index))


def test_recall_at_cutoffs() -> None:
    """Match by path and IoU and report reference sources missed at five."""
    source = {
        "file_path": "raw/example.py",
        "first_character_index": 0,
        "last_character_index": 100,
    }
    question = {"question_id": "q1", "question": "Example?"}
    reference = RagDataset.model_validate({"rag_questions": [{
        **question, "answer": "Example.",
        "sources": [source, {**source, "file_path": "raw/missing.py"}],
    }]})
    results = StudentSearchResults.model_validate({
        "k": 5, "search_results": [{
            **question, "retrieved_sources": [
                {**source, "last_character_index": 4},
                {**source, "last_character_index": 5},
            ],
        }],
    })
    recall, misses = evaluate_recall(results, reference)
    assert recall == {
        "recall@1": 0, "recall@3": 0.5, "recall@5": 0.5, "recall@10": 0.5,
    }
    assert [m.question_id for m in misses] == ["q1"]
    assert [s.file_path for s in misses[0].retrieved_sources] == [
        "raw/missing.py"
    ]
