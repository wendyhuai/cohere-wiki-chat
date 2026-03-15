"""In-memory store for query history."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class Record:
    id: int
    query: str
    response: str
    created_at: datetime


class HistoryStore:
    """Thread-safe, in-memory store for chat query/response pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[Record] = []
        self._next_id: int = 1

    def save(self, query: str, response: str) -> int:
        with self._lock:
            record = Record(
                id=self._next_id,
                query=query,
                response=response,
                created_at=datetime.now(timezone.utc),
            )
            self._records.append(record)
            self._next_id += 1
            return record.id

    def all(self) -> List[Record]:
        with self._lock:
            return list(self._records)
