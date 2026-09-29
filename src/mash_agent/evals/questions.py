"""The fixed evaluation question set."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_QUESTIONS_PATH = Path("evals/questions.yaml")


class Question(BaseModel):
    id: str
    category: str
    question: str
    coverage_terms: list[str] = Field(default_factory=list)
    expect_no_claims: bool = False


class QuestionSet(BaseModel):
    questions: list[Question]

    @model_validator(mode="after")
    def _unique_ids(self) -> "QuestionSet":
        ids = [q.id for q in self.questions]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate question ids: {dupes}")
        if not ids:
            raise ValueError("question set is empty")
        return self


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    return QuestionSet.model_validate(yaml.safe_load(path.read_text())).questions


def select(questions: list[Question], ids: list[str] | None, limit: int | None) -> list[Question]:
    """Filter by id (unknown ids are an error, not silently ignored), then cap the count."""
    if ids:
        known = {q.id for q in questions}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"unknown question id(s): {unknown}; known: {sorted(known)}")
        questions = [q for q in questions if q.id in ids]
    return questions[:limit] if limit else questions
