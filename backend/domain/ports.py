"""Abstract ports (interfaces) for the domain layer.

Zero framework or vendor imports. Adapters in ``infrastructure/`` implement
these contracts. Swapping an adapter only requires changing the adapter and
the DI wiring — never ``domain/`` or ``application/``.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import ChunkRecord, DocumentRecord, QueryLogEntry


class EmbeddingProvider(ABC):
    """Dedicated embedding provider (NEVER the chat LLM)."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns one vector per text."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""

    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class VectorIndex(ABC):
    """Dense vector index for semantic similarity search."""

    @abstractmethod
    def add(
        self, items: List[Tuple[str, str, List[float]]]
    ) -> None:
        """Add vectors as (chunk_id, doc_id, vector) tuples."""

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        """Return [(chunk_id, score)] sorted by relevance desc."""

    @abstractmethod
    def delete(self, doc_id: str) -> int:
        """Remove all vectors belonging to a document. Return count removed."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the whole index."""

    @abstractmethod
    def dimension(self) -> int:
        """Expected vector dimension."""


class KeywordIndex(ABC):
    """Sparse keyword index (e.g. BM25) for lexical search."""

    @abstractmethod
    def add(self, chunks: List[ChunkRecord]) -> None:
        """Index chunks for keyword search."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Return [(chunk_id, score)] sorted by relevance desc."""

    @abstractmethod
    def delete(self, doc_id: str) -> int:
        """Remove all index entries belonging to a document."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the whole index."""


class LLMProvider(ABC):
    """Model-agnostic chat LLM provider (local Ollama by default)."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion. Returns raw model text."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""


class DocumentRepository(ABC):
    """Persistent document + chunk metadata store."""

    @abstractmethod
    def save_document(self, doc: DocumentRecord) -> None:
        ...

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        ...

    @abstractmethod
    def list_documents(self) -> List[DocumentRecord]:
        ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Return True if a document was deleted."""

    @abstractmethod
    def save_chunk(self, chunk: ChunkRecord) -> None:
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        ...

    @abstractmethod
    def get_chunks_by_document(self, doc_id: str) -> List[ChunkRecord]:
        ...

    @abstractmethod
    def delete_chunks_by_document(self, doc_id: str) -> int:
        """Return count of chunks removed."""


class QueryLogRepository(ABC):
    """Persistent Q&A history store."""

    @abstractmethod
    def save_entry(self, entry: QueryLogEntry) -> None:
        ...

    @abstractmethod
    def list_entries(self, limit: int = 100) -> List[QueryLogEntry]:
        ...
