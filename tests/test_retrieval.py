"""Tests for BM25 scores, ranking, and persisted retrieval."""

from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.indexing import IndexedChunk, InvertedIndex, load_index, save_index
from src.retrieval import BM25


@pytest.fixture
def index() -> InvertedIndex:
    """Build three fragments with known lengths and term frequencies."""
    return InvertedIndex(
        chunks=[
            IndexedChunk(
                file_path="raw/example.py",
                first_character_index=start,
                last_character_index=start + 10,
                token_count=length,
            )
            for start, length in [(0, 4), (10, 8), (20, 6)]
        ],
        postings={"cache": [(0, 2), (1, 2)], "block": [(1, 1)]},
    )


def test_score_and_length_normalization(index: InvertedIndex) -> None:
    """Check BM25 against a hand-computed score for average length six."""
    hits = BM25(index).search("cache")
    assert [source.first_character_index for source, _ in hits] == [0, 10]
    assert hits[0][1] == pytest.approx(log(1.6) * 4.4 / 2.9)
    assert hits[1][1] == pytest.approx(log(1.6) * 4.4 / 3.5)


def test_rank_multiple_terms(index: InvertedIndex) -> None:
    """Favor the fragment matching both terms and enforce top k."""
    [(source, score)] = BM25(index).search("cache block", k=1)
    assert source.file_path == "raw/example.py"
    assert (source.first_character_index, source.last_character_index) == (
        10, 20
    )
    assert score == pytest.approx(
        log(1.6) * 4.4 / 3.5 + log(8 / 3) * 2.2 / 2.5
    )


def test_empty_results(index: InvertedIndex) -> None:
    """Return an empty list for empty queries, misses and k=0."""
    bm25 = BM25(index)
    assert bm25.search(" \t") == []
    assert bm25.search("xyz") == []
    assert bm25.search("cache", k=0) == []
    assert BM25(InvertedIndex(chunks=[], postings={})).search("cache") == []
    with pytest.raises(ValueError, match="non-negative integer"):
        bm25.search("cache", k=-1)


def test_persisted_ranking(index: InvertedIndex) -> None:
    """Produce identical results after loading the index from JSON."""
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "index.json"
        save_index(index, path)
        assert BM25(load_index(path)).search("cache block") == (
            BM25(index).search("cache block")
        )
