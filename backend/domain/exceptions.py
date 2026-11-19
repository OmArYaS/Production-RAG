"""Domain exceptions.

All exceptions raised by domain/application layers. Presentation layer maps
these to HTTP responses. Never raised by infrastructure adapters directly
(wrap vendor errors in these).
"""

from typing import Optional


class DomainError(Exception):
    """Base domain error."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentNotFoundError(DomainError):
    """A requested document does not exist."""

    def __init__(self, doc_id: str):
        super().__init__(f"Document not found: {doc_id}", status_code=404)
        self.doc_id = doc_id


class UnsupportedFileFormatError(DomainError):
    """The uploaded file format is not supported."""

    def __init__(self, filename: str, supported: Optional[list] = None):
        supported = supported or [".txt", ".md", ".pdf"]
        msg = (
            f"Unsupported file format: {filename}. "
            f"Supported formats: {', '.join(supported)}"
        )
        super().__init__(msg, status_code=400)
        self.filename = filename


class EmptyDocumentError(DomainError):
    """A document contains no extractable text."""

    def __init__(self, filename: str, reason: str = "no extractable text"):
        super().__init__(
            f"Empty document '{filename}': {reason}. "
            "Nothing to index.",
            status_code=422,
        )
        self.filename = filename


class DuplicateDocumentError(DomainError):
    """A document with the same source name already exists."""

    def __init__(self, source_name: str, doc_id: str):
        super().__init__(
            f"Document already exists: {source_name} (doc_id={doc_id}). "
            "Re-uploading replaces the existing document.",
            status_code=409,
        )
        self.source_name = source_name
        self.doc_id = doc_id


class NoIndexedDocumentsError(DomainError):
    """The knowledge base is empty; no retrieval possible."""

    def __init__(self):
        super().__init__(
            "No indexed documents. Upload documents before asking questions.",
            status_code=422,
        )


class InsufficientContextError(DomainError):
    """Retrieved chunks do not contain enough evidence to answer."""

    def __init__(self, question: str):
        super().__init__(
            "The information requested is not found in the provided documents.",
            status_code=422,
        )
        self.question = question


class LLMProviderError(DomainError):
    """The chat LLM provider failed or returned an unusable response."""

    def __init__(self, message: str):
        super().__init__(f"LLM provider error: {message}", status_code=502)


class EmbeddingDimensionError(DomainError):
    """Embedding dimension mismatch between query and index."""

    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"Embedding dimension mismatch: index expects {expected}, "
            f"got {actual}. Re-index documents with the current embedding model.",
            status_code=409,
        )
        self.expected = expected
        self.actual = actual


class VectorIndexError(DomainError):
    """Vector index operation failed."""

    pass
