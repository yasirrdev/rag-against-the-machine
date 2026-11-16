"""Rank indexed fragments with BM25."""

from collections import defaultdict
from heapq import nsmallest
from math import log1p

from src.indexing import InvertedIndex, tokenize
from src.models import MinimalSource

_K1 = 1.2
_B = 0.75


class BM25:
    """Prepare corpus statistics once and score only matching fragments."""

    def __init__(self, index: InvertedIndex) -> None:
        """Precompute the document-length normalization for every fragment."""
        self.index = index
        self.count = len(index.chunks)
        total_length = sum(chunk.token_count for chunk in index.chunks)
        average = total_length / self.count if self.count else 0.0
        self.normalization = [
            _K1 * (1 - _B + _B * chunk.token_count / average)
            for chunk in index.chunks
        ] if average else []

    def search(
        self, query: str, k: int = 10
    ) -> list[tuple[MinimalSource, float]]:
        """Return up to k ranked sources and their scores."""
        if type(k) is not int or k < 0:
            raise ValueError("k must be a non-negative integer")
        if not k or not self.normalization:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(tokenize(query)):
            postings = self.index.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = log1p((self.count - df + 0.5) / (df + 0.5))
            for chunk_id, frequency in postings:
                denominator = frequency + self.normalization[chunk_id]
                scores[chunk_id] += (
                    idf * frequency * (_K1 + 1) / denominator
                )

        ranked = nsmallest(
            k, scores.items(), key=lambda hit: (-hit[1], hit[0])
        )
        results: list[tuple[MinimalSource, float]] = []
        for chunk_id, score in ranked:
            chunk = self.index.chunks[chunk_id]
            source = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            results.append((source, score))
        return results
