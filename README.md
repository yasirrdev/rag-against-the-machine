# RAG Against the Machine

This project has been created as part of the 42 curriculum by yasirrdev.

## Description

**RAG Against the Machine** is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the [vLLM](https://github.com/vllm-project/vllm) codebase.

The project combines information retrieval techniques with a Large Language Model (LLM) to retrieve relevant code and documentation snippets and generate answers grounded in the source material.

The system uses **Qwen3-0.6B** for answer generation and implements lexical retrieval techniques to identify relevant information within the codebase.

### Main objectives

- Index the vLLM codebase.
- Implement chunking strategies for Python code and Markdown documentation.
- Retrieve relevant source code and documentation.
- Generate context-aware answers using Qwen3-0.6B.
- Evaluate retrieval quality using Recall@k metrics.

## Technologies

- Python 3.10+
- Qwen3-0.6B
- Pydantic
- Python Fire
- uv
- BM25 / TF-IDF
- tqdm

## Project Structure

The project follows a modular architecture separating indexing, chunking, retrieval, generation and evaluation components.

## Instructions

Installation and usage instructions will be documented as the implementation progresses.

## Resources

- [42 School](https://42.fr/)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

## Acknowledgements

Developed as part of the 42 curriculum.

## AI Usage

AI tools may be used as development assistants for research, explanations, debugging and code review. All implementation decisions and submitted code must be understood and validated by the student.
