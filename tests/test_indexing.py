"""Regression tests for the index and its input contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from src.indexing import InvertedIndex, build_index, load_index, save_index
from src.models import RagDataset


def test_index_roundtrip() -> None:
    """Keep exact metadata and term frequencies through a JSON roundtrip."""
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        corpus = root / "raw"
        corpus.mkdir()
        (corpus / "notes.txt").write_bytes(b"foo foo\nbar bar")
        index = build_index(corpus, root, max_chunk_size=8)
        assert [
            (c.file_path, c.first_character_index, c.last_character_index,
             c.token_count)
            for c in index.chunks
        ] == [("raw/notes.txt", 0, 8, 2), ("raw/notes.txt", 8, 15, 2)]
        assert index.postings == {"foo": [(0, 2)], "bar": [(1, 2)]}
        path = root / "processed" / "index.json"
        save_index(index, path)
        assert json.loads(path.read_text(encoding="utf-8"))["postings"] == {
            "foo": [[0, 2]], "bar": [[1, 2]],
        }
        assert load_index(path) == index


@pytest.mark.parametrize("posting", [(0, -1), (0, 0), (-1, 1), (1, 1)])
def test_invalid_posting(posting: tuple[int, int]) -> None:
    """Reject nonpositive frequencies and out-of-range chunk IDs."""
    with pytest.raises(ValidationError):
        InvertedIndex.model_validate({
            "chunks": [{
                "file_path": "raw/example.txt",
                "first_character_index": 0,
                "last_character_index": 5,
                "token_count": 1,
            }],
            "postings": {"example": [posting]},
        })


def test_malformed_reference() -> None:
    """Keep a malformed answer from falling back to an unanswered question."""
    with pytest.raises(ValidationError, match="file_path"):
        RagDataset.model_validate({
            "rag_questions": [{
                "question": "What is the answer?",
                "answer": "Example.",
                "sources": [{}],
            }],
        })
