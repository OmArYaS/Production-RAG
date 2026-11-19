"""SQLite DocumentRepository adapter."""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from domain.models import ChunkRecord, DocumentRecord
from domain.ports import DocumentRepository

from . import db


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, url: str) -> None:
        self._url = url
        db.init_document_schema(url)

    def _conn(self) -> sqlite3.Connection:
        return db.connect(self._url)

    # ------------------------------------------------------------------
    # DocumentRecord helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_to_row(doc: DocumentRecord):
        return (
            doc.doc_id,
            doc.source_name,
            doc.file_path,
            doc.content_type,
            doc.size_bytes,
            doc.chunk_count,
            doc.status,
            doc.created_at.isoformat(),
            doc.updated_at.isoformat(),
        )

    @staticmethod
    def _doc_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            source_name=row["source_name"],
            file_path=row["file_path"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"] or 0,
            chunk_count=row["chunk_count"] or 0,
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # ChunkRecord helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_to_row(chunk: ChunkRecord):
        return (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.source_name,
            chunk.chunk_index,
            chunk.text,
            chunk.page_number,
            chunk.section_heading,
            json.dumps(chunk.metadata or {}),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
        metadata = row["metadata"] or "{}"
        try:
            parsed = json.loads(metadata)
        except (ValueError, TypeError):
            parsed = {}
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_name=row["source_name"],
            chunk_index=row["chunk_index"] or 0,
            text=row["text"],
            page_number=row["page_number"],
            section_heading=row["section_heading"],
            metadata=parsed,
        )

    # ------------------------------------------------------------------
    # DocumentRepository contract
    # ------------------------------------------------------------------
    def save_document(self, doc: DocumentRecord) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, source_name, file_path, content_type, size_bytes,
                 chunk_count, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._doc_to_row(doc),
            )
            conn.commit()
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            )
            row = cur.fetchone()
            return self._doc_from_row(row) if row else None
        finally:
            conn.close()

    def list_documents(self) -> List[DocumentRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            )
            return [self._doc_from_row(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def save_chunk(self, chunk: ChunkRecord) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, source_name, chunk_index, text, page_number,
                 section_heading, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._chunk_to_row(chunk),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            row = cur.fetchone()
            return self._chunk_from_row(row) if row else None
        finally:
            conn.close()

    def get_chunks_by_document(self, doc_id: str) -> List[ChunkRecord]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            )
            return [self._chunk_from_row(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def delete_chunks_by_document(self, doc_id: str) -> int:
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
