*This project has been created as part of the 42 curriculum by ybel-maa.*

# RAG Against the Machine

## Description

This project builds a retrieval-augmented generation system for the vLLM 0.10.1 codebase. The Fire CLI can build and save the inverted index. Retrieval and generation are pending.

## Instructions

Use Python 3.10 or later and run `uv sync` from the repository root. Place the supplied corpus in `data/raw/vllm-0.10.1/` and the reference datasets in their corresponding `data/datasets/AnsweredQuestions/` and `data/datasets/UnansweredQuestions/` directories.

## Example usage

```sh
uv run python -m src --help
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --help
make lint
```

## System architecture

The planned pipeline indexes the corpus under `data/processed/`, retrieves ranked file ranges for each question, and passes the corresponding text to Qwen/Qwen3-0.6B. Batch results go under `data/output/search_results/` and generated answers under `data/output/search_results_and_answer/`, each scoped by dataset.

## Chunking strategy

The indexer splits Python around top-level definitions and Markdown around sections. It subdivides long blocks and preserves character offsets, with a default maximum of 2,000 characters per chunk.

## Retrieval method

The persisted inverted index stores term frequencies and chunk lengths for BM25. It also retains the original file path and character range for each chunk. BM25 scoring will be added with retrieval.

## Performance analysis

Benchmarks are pending. The targets are at least 80% Recall@5 for documentation and 50% for code, with indexing in at most five minutes and retrieval of 200 questions in at most 90 seconds.

## Design decisions

The CLI, Pydantic contracts, retrieval, and generation have separate responsibilities. Single-question and batch commands will share the same retrieval implementation.

## Challenges faced

The supplied corpus was nested one level deeper than the reference paths. The repository layout was corrected so that those paths resolve directly.

## Resources

- [vLLM 0.10.1 source](https://github.com/vllm-project/vllm/tree/v0.10.1)
- [Pydantic documentation](https://docs.pydantic.dev/)
- The supplied subject and Moulinette notes are in `docs/`.

AI assisted with reviewing the subject, planning the pipeline, scaffolding the data models and CLI, and checking the corpus layout against the reference datasets.
