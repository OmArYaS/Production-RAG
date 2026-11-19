"""FastAPI request/response schemas.

Response models reuse the pure domain models (``DocumentRecord``,
``QueryLogEntry``, ``AnswerResponse``). Request models are new.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """POST /api/query request body."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return v.strip()
