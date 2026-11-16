"""Split corpus files into contiguous, source-addressable chunks."""

import ast
import re
import tokenize
from io import BytesIO
from pathlib import Path

from src.models import MinimalSource


class Chunk(MinimalSource):
    """A source range together with its exact text."""

    text: str


def _python_boundaries(text: str) -> list[int]:
    """Find the starts of top-level functions and classes."""
    try:
        module = ast.parse(text.removeprefix("\ufeff"))
    except (SyntaxError, ValueError):
        return []

    line_starts = [0]
    line_starts.extend(m.end() for m in re.finditer(r"\r\n|\r|\n", text))
    boundaries: list[int] = []
    for node in module.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            lines = [node.lineno, *(d.lineno for d in node.decorator_list)]
            line = min(lines)
            boundaries.append(line_starts[line - 1])
    return boundaries


def _markdown_boundaries(text: str) -> list[int]:
    """Find ATX headings outside fenced code blocks."""
    boundaries: list[int] = []
    fence = ""
    for line in re.finditer(r"[^\r\n]+", text):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line[0])
        if marker:
            run, tail = marker.groups()
            if not fence:
                fence = run
            elif run.startswith(fence) and not tail.strip():
                fence = ""
        elif not fence and re.match(r" {0,3}#{1,6}(?:[ \t]|$)", line[0]):
            boundaries.append(line.start())
    return boundaries


def chunk_file(
    file_path: Path, project_root: Path, max_chunk_size: int = 2000
) -> list[Chunk]:
    """Read a corpus file and return exact, project-relative text ranges.

    Offsets are zero-based character indices with an exclusive end.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    relative_path = file_path.relative_to(project_root).as_posix()
    raw = file_path.read_bytes()
    suffix = file_path.suffix.lower()
    encoding = "utf-8"
    if suffix == ".py":
        encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
        if encoding == "utf-8-sig":
            encoding = "utf-8"
    text = raw.decode(encoding)

    if suffix == ".py":
        boundaries = _python_boundaries(text)
    elif suffix in {".md", ".markdown"}:
        boundaries = _markdown_boundaries(text)
    else:
        boundaries = [
            m.end()
            for m in re.finditer(
                r"(?:\r\n|\n|\r(?!\n))[ \t]*(?:\r\n|\n|\r(?!\n))",
                text,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    for end in [*boundaries, len(text)]:
        while start < end:
            stop = min(start + max_chunk_size, end)
            if stop < end:
                newline = max(
                    text.rfind("\n", start, stop),
                    text.rfind("\r", start, stop),
                )
                if newline >= start:
                    stop = newline + 1
            chunks.append(Chunk(
                file_path=relative_path,
                first_character_index=start,
                last_character_index=stop,
                text=text[start:stop],
            ))
            start = stop
    return chunks
