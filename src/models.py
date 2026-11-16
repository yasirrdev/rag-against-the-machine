"""Pydantic models for the datasets and the pipeline's JSON output."""

from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class MinimalSource(BaseModel):
    """Identify a character range within a corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question and its stable identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a reference answer and its supporting sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent the questions in an input dataset."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @field_validator("rag_questions", mode="before")
    @classmethod
    def validate_answered_questions(cls, value: object) -> object:
        """Require a complete answer record when answer fields are present."""
        if not isinstance(value, list):
            return value
        return [
            AnsweredQuestion.model_validate(item)
            if isinstance(item, dict)
            and {"answer", "sources"}.intersection(item)
            else item
            for item in value
        ]


class MinimalSearchResults(BaseModel):
    """Store the ranked sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a generated answer alongside its retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Store ranked search results and the requested number of sources."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store generated answers and the requested number of sources."""

    search_results: list[MinimalAnswer]
    k: int
