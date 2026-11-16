"""Tests for chunk boundaries, offsets, and complete file coverage."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.chunking import Chunk, chunk_file


class ChunkingTests(unittest.TestCase):
    """Exercise the file-level chunking API."""

    def split(
        self, name: str, text: str, limit: int = 2000
    ) -> list[Chunk]:
        """Write a fixture and check that the chunks reproduce it exactly."""
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            relative = f"data/raw/vllm-0.10.1/{name}"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(text.encode("utf-8"))
            chunks = chunk_file(path, root, limit)

        start = 0
        for chunk in chunks:
            stop = chunk.last_character_index
            self.assertEqual(chunk.file_path, relative)
            self.assertEqual(chunk.first_character_index, start)
            self.assertGreater(stop, start)
            self.assertLessEqual(stop - start, limit)
            self.assertEqual(chunk.text, text[start:stop])
            start = stop
        self.assertEqual(start, len(text))
        return chunks

    def test_empty_files(self) -> None:
        """Do not emit an empty fragment for an empty file."""
        for name in ("empty.py", "empty.md", "empty.txt"):
            with self.subTest(name=name):
                self.assertEqual(self.split(name, ""), [])

    def test_python_declarations(self) -> None:
        """Keep decorators with functions and count Unicode characters."""
        header = "# café 🌍\r\nimport os\r\n\r\n"
        function = "@decorate\r\nasync def run():\r\n    pass\r\n\r\n"
        klass = "class Engine:\r\n    def run(self):\r\n        pass\r\n"
        chunks = self.split("engine.py", header + function + klass)
        self.assertEqual([c.text for c in chunks], [header, function, klass])

    def test_markdown_sections(self) -> None:
        """Ignore headings inside fences and preserve blank lines."""
        intro = "Introduction\n\n"
        first = "# First\n\n```python\n# comment\n```\n\n"
        second = "## Second\ntext\n"
        chunks = self.split("guide.md", intro + first + second)
        self.assertEqual([c.text for c in chunks], [intro, first, second])

    def test_plain_text_paragraphs(self) -> None:
        """Distinguish a single CRLF from a blank line."""
        first = "one\r\ntwo\r\n\r\n"
        chunks = self.split("notes.txt", first + "three")
        self.assertEqual([c.text for c in chunks], [first, "three"])

    def test_oversized_blocks(self) -> None:
        """Split oversized Python and Markdown blocks without losing text."""
        for name, prefix in (("long.py", "# "), ("long.md", "# Heading\n")):
            with self.subTest(name=name):
                chunks = self.split(name, prefix + "é" * 4500)
                self.assertGreaterEqual(len(chunks), 3)
                self.assertEqual(max(len(c.text) for c in chunks), 2000)

    def test_custom_limit_and_invalid_python(self) -> None:
        """Keep invalid Python source while respecting the custom limit."""
        text = "def broken(\n" + "x" * 30
        chunks = self.split("broken.py", text, 5)
        self.assertGreater(len(chunks), 1)

    def test_invalid_limit(self) -> None:
        """Reject a limit that would prevent making progress."""
        with self.assertRaisesRegex(ValueError, "must be positive"):
            chunk_file(Path("file.py"), Path("."), 0)
