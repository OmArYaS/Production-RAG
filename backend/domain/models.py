"""Pure Pydantic domain models.

No framework or vendor imports. These model the core business entities of the
Hybrid Document & Knowledge Assistant.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    """A registered document with its indexing metadata."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source_name: str
    file_path: Optional[str] = None
    content_type: str
    size_bytes: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    status: str = "indexed"  # indexed | failed | deleted

    model_config = ConfigDict(frozen=False)


class ChunkRecord(BaseModel):
    """A text chunk of a document, enriched with retrieval metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    source_name: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A chunk returned by hybrid retrieval, with its fusion score."""

    chunk_id: str
    doc_id: str
    source_name: str
    text: str
    score: float
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_index: Optional[int] = None


class Citation(BaseModel):
    """A citation referencing an exact source chunk."""

    index: int  # 1-based citation number used inline as [N]
    chunk_id: str
    doc_id: str
    source_name: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_index: Optional[int] = None
    snippet: str
    score: Optional[float] = None


class AnswerResponse(BaseModel):
    """The synthesized answer to a user question."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    abstained: bool = False
    model: str
    latency_ms: float
    hits_count: int = 0


class QueryLogEntry(BaseModel):
    """A persisted Q&A history record."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    abstained: bool = False
    model: str
    latency_ms: float
    hits_count: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
