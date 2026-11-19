"""SQLite QueryLogRepository adapter."""

from __future__ import annotations

import json
import sqlite3
from typing import List

from domain.models import Citation, QueryLogEntry
from domain.ports import QueryLogRepository

from . import db
from .document_repo import SQLiteDocumentRepository


class SQLiteQueryLogRepository(QueryLogRepository):
    def __init__(self, url: str) -> None:
        self._url = url
        db.init_query_log_schema(url)

    def _conn(self) -> sqlite3.Connection:
        return db.connect(self._url)

    @staticmethod
    def _entry_to_row(entry: QueryLogEntry):
        return (
            entry.id,
            entry.question,
            entry.answer,
            json.dumps(
                [c.model_dump() for c in entry.citations]
            ),
            1 if entry.abstained else 0,
            entry.model,
            entry.latency_ms,
            entry.hits_count,
            entry.created_at.isoformat(),
        )

    @staticmethod
    def _entry_from_row(row: sqlite3.Row) -> QueryLogEntry:
        try:
            citations = [
                Citation(**c) for c in json.loads(row["citations"] or "[]")
            ]
        except (ValueError, TypeError):
            citations = []
        return QueryLogEntry(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            citations=citations,
            abstained=bool(row["abstained"]),
            model=row["model"],
            latency_ms=row["latency_ms"] or 0.0,
            hits_count=row["hits_count"] or 0,
            created_at=row["created_at"],
        )

    def save_entry(self, entry: QueryLogEntry) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_logs
                (id, question, answer, citations, abstained, model,
                 latency_ms, hits_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._entry_to_row(entry),
            )
            conn.commit()
        finally:
            conn.close()

    def list_entries(self, limit: int = 100) -> List[QueryLogEntry]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [self._entry_from_row(r) for r in cur.fetchall()]
        finally:
            conn.close()


def build_sqlite_repos(url: str):
    """Factory: build both repos bound to the same SQLite database."""
    db.init_document_schema(url)
    db.init_query_log_schema(url)
    return (
        SQLiteDocumentRepository(url),
        SQLiteQueryLogRepository(url),
    )
