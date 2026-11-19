"""Document deletion use case.

Depends ONLY on ``domain/`` ports.
"""

from __future__ import annotations

from domain.ports import (
    DocumentRepository,
    KeywordIndex,
    VectorIndex,
)


class DeleteDocumentUseCase:
    """Delete a document and clean up its chunks from all indexes.

    Returns True if deleted, False if the document did not exist.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
    ) -> None:
        self._doc_repo = document_repo
        self._vector_index = vector_index
        self._keyword_index = keyword_index

    def execute(self, doc_id: str) -> bool:
        if not self._doc_repo.get_document(doc_id):
            return False
        self._keyword_index.delete(doc_id)
        self._vector_index.delete(doc_id)
        self._doc_repo.delete_chunks_by_document(doc_id)
        self._doc_repo.delete_document(doc_id)
        return True
