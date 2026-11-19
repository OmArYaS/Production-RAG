"""Q&A history retrieval use case.

Depends ONLY on ``domain/`` ports.
"""

from __future__ import annotations

from typing import List

from domain.models import QueryLogEntry
from domain.ports import QueryLogRepository


class ListHistoryUseCase:
    """Retrieve Q&A history entries, most recent first."""

    def __init__(self, query_log_repo: QueryLogRepository) -> None:
        self._query_log_repo = query_log_repo

    def execute(self, limit: int = 100) -> List[QueryLogEntry]:
        entries = self._query_log_repo.list_entries(limit=limit)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries
