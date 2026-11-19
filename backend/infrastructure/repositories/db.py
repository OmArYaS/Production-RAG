"""Shared SQLite helper for repositories."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from infrastructure.config import Settings


def _parse_db_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


def ensure_db(url: str) -> str:
    path = _parse_db_url(url)
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    return path


def connect(url: str) -> sqlite3.Connection:
    path = ensure_db(url)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_document_schema(url: str) -> None:
    conn = connect(url)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                file_path TEXT,
                content_type TEXT,
                size_bytes INTEGER,
                chunk_count INTEGER,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_name TEXT,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                page_number INTEGER,
                section_heading TEXT,
                metadata TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)"
        )
        conn.commit()
    finally:
        conn.close()


def init_query_log_schema(url: str) -> None:
    conn = connect(url)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT,
                abstained INTEGER,
                model TEXT,
                latency_ms REAL,
                hits_count INTEGER,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
