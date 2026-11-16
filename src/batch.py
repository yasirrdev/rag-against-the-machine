"""Run dataset searches and calculate local source recall."""

from tqdm import tqdm

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.retrieval import BM25

RECALL_CUTOFFS = (1, 3, 5, 10)


def retrieve_question(
    ranker: BM25, question: UnansweredQuestion, k: int
) -> MinimalSearchResults:
    """Return ranked sources with the original question identity."""
    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=[
            source for source, _ in ranker.search(question.question, k)
        ],
    )


def retrieve_dataset(
    ranker: BM25, dataset: RagDataset, k: int
) -> StudentSearchResults:
    """Reuse one prepared ranker for all questions in input order."""
    return StudentSearchResults(
        search_results=[
            retrieve_question(ranker, question, k)
            for question in tqdm(dataset.rag_questions, unit="question")
        ],
        k=k,
    )


def _matches(expected: MinimalSource, retrieved: MinimalSource) -> bool:
    """Check literal path equality and character-span IoU of at least 0.05."""
    if expected.file_path != retrieved.file_path:
        return False
    overlap = min(
        expected.last_character_index, retrieved.last_character_index
    ) - max(expected.first_character_index, retrieved.first_character_index)
    union = max(
        expected.last_character_index, retrieved.last_character_index
    ) - min(expected.first_character_index, retrieved.first_character_index)
    return overlap > 0 and overlap * 20 >= union


def evaluate_recall(
    results: StudentSearchResults, reference: RagDataset
) -> tuple[dict[str, float], list[MinimalSearchResults]]:
    """Average source recall by question and report sources missed at five."""
    retrieved = {
        row.question_id: row.retrieved_sources
        for row in results.search_results
    }
    totals = dict.fromkeys(RECALL_CUTOFFS, 0.0)
    misses: list[MinimalSearchResults] = []
    count = 0
    for question in reference.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            raise ValueError(
                "Evaluation requires answered questions with sources"
            )
        count += 1
        candidates = retrieved.get(question.question_id, [])[:results.k]
        missing = []
        for source in question.sources:
            rank = next(
                (i for i, hit in enumerate(candidates, 1)
                 if _matches(source, hit)), None
            )
            for cutoff in RECALL_CUTOFFS:
                if rank is not None and rank <= cutoff:
                    totals[cutoff] += 1 / len(question.sources)
            if rank is None or rank > 5:
                missing.append(source)
        if missing:
            misses.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=missing,
            ))
    recall = {
        f"recall@{k}": value / count if count else 0.0
        for k, value in totals.items()
    }
    return recall, misses
