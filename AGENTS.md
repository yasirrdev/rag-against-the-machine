# RAG Against the Machine — Codex Instructions

## Goal
Build a RAG system for the vLLM 0.10.1 codebase, following the official 42 subject.

## Rules
- The official subject is the source of truth.
- Read the Moulinette README to understand evaluation requirements.
- Use Python 3.10+, uv, Pydantic, Python Fire and tqdm.
- Implement BM25 or TF-IDF retrieval.
- Support Python and Markdown chunking.
- Use Qwen/Qwen3-0.6B for answer generation.
- Respect the 2000-character default chunk limit.
- Prioritize Recall@5 and performance requirements.
- Keep the architecture simple and modular.
- Use type hints, docstrings, flake8 and mypy.
- Do not implement bonuses before mandatory requirements pass.

## Workflow
- Communicate in Spanish.
- Work in small, testable phases.
- Explain important technical decisions.
- Ask before major architectural changes.
- Run relevant tests after implementation.
- Do not modify unrelated files.
- Never claim tests passed without executing them.
- Ensure I can explain the implementation during my 42 evaluation.
