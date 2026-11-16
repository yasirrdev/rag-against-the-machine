"""Build and persist the inverted index used by BM25."""

import json
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from src.chunking import chunk_file
from src.models import MinimalSource

_WORDS = re.compile(r"[^\W_]+(?:_[^\W_]+)*")
_IDENTIFIER_SEPARATOR = re.compile(
    r"_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_EXTENSIONS = {".py", ".md", ".markdown", ".txt"}
ChunkId = Annotated[int, Field(ge=0)]
TermFrequency = Annotated[int, Field(gt=0)]


class IndexInputError(ValueError):
    """Report invalid arguments to the index builder."""


def tokenize(text: str) -> list[str]:
    """Extract words and searchable components of code identifiers."""
    tokens: list[str] = []
    for word in _WORDS.findall(text):
        tokens.append(word.casefold())
        components = _IDENTIFIER_SEPARATOR.split(word)
        if len(components) > 1:
            tokens.extend(component.casefold() for component in components)
    return tokens


class IndexedChunk(MinimalSource):
    """Identify a chunk and record its length in indexed terms."""

    token_count: int


class InvertedIndex(BaseModel):
    """Store chunk metadata and term postings as (chunk ID, frequency)."""

    chunks: list[IndexedChunk]
    postings: dict[str, list[tuple[ChunkId, TermFrequency]]]

    @model_validator(mode="after")
    def validate_chunk_references(self) -> "InvertedIndex":
        """Ensure every posting points to an existing chunk."""
        chunk_count = len(self.chunks)
        for term, entries in self.postings.items():
            for chunk_id, _ in entries:
                if chunk_id >= chunk_count:
                    raise ValueError(f"Unknown chunk {chunk_id} for {term!r}")
        return self


def build_index(
    corpus_root: Path, project_root: Path, max_chunk_size: int = 2000
) -> InvertedIndex:
    """Build postings, using each chunk's list position as its stable ID."""
    if type(max_chunk_size) is not int or not 1 <= max_chunk_size <= 2000:
        raise IndexInputError(
            "max_chunk_size must be an integer from 1 to 2000"
        )
    if not corpus_root.is_dir():
        raise IndexInputError(f"Corpus directory not found: {corpus_root}")
    if not corpus_root.is_relative_to(project_root):
        raise IndexInputError("Corpus directory must be inside the project")

    files = sorted(
        path for path in corpus_root.rglob("*")
        if path.suffix.lower() in _EXTENSIONS and path.is_file()
    )
    chunks: list[IndexedChunk] = []
    postings: dict[str, list[tuple[int, int]]] = {}
    for path in tqdm(files, desc="Indexing", unit="file"):
        for chunk in chunk_file(path, project_root, max_chunk_size):
            tokens = tokenize(chunk.text)
            chunk_id = len(chunks)
            chunks.append(IndexedChunk(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
                token_count=len(tokens),
            ))
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((chunk_id, frequency))
    return InvertedIndex(chunks=chunks, postings=postings)


def save_index(index: InvertedIndex, path: Path) -> None:
    """Publish a complete JSON index via an atomic file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        dir=path.parent, encoding="utf-8", mode="w", delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(index.model_dump_json())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_index(path: Path) -> InvertedIndex:
    """Load and validate an index saved by the indexing command."""
    with path.open(encoding="utf-8") as stream:
        return InvertedIndex.model_validate(json.load(stream))
