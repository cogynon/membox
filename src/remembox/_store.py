"""Storage backend protocol. Implement this to plug in any database."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Protocol, runtime_checkable

from remembox.models import Episode, Fact


@runtime_checkable
class EpisodicStoreProtocol(Protocol):
    """Interface for episodic memory backends.

    Implement these methods to support a new storage backend
    (e.g., PostgreSQL, Redis). The default SQLite implementation
    is in episodic.py.
    """

    def record(self, episode: Episode) -> None:
        """Persist a single episode."""
        ...

    def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID. Returns None if not found."""
        ...

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the N most recent episodes, newest first."""
        ...

    def search(self, keyword: str, limit: int = 10) -> list[Episode]:
        """Full-text keyword search across episode content."""
        ...

    def count(self) -> int:
        """Total number of episodes in the store."""
        ...

    # ── Maintenance methods used by forgetting.py and consolidation.py ──

    def iter_all(self, batch_size: int = 500) -> Iterator[Episode]:
        """Iterate over all episodes in storage-efficient batches."""
        ...

    def delete(self, episode_ids: list[str]) -> int:
        """Delete episodes by ID. Returns number deleted."""
        ...

    def mark_consolidated(self, episode_ids: list[str]) -> int:
        """Mark episodes as consolidated. Returns number updated."""
        ...

    def mark_archived(self, episode_ids: list[str]) -> int:
        """Mark episodes as archived (soft-delete). Returns number updated."""
        ...

    def unconsolidated(self, limit: int = 100) -> list[Episode]:
        """Return episodes that have not yet been consolidated."""
        ...

    def by_time_range(self, start: datetime, end: datetime) -> list[Episode]:
        """Return episodes within a time window."""
        ...

    def increment_access(self, episode_id: str) -> None:
        """Bump the access counter for an episode."""
        ...

    def by_thread(self, thread_id: str, limit: int = 1000) -> list[Episode]:
        """Return episodes belonging to a thread."""
        ...

    def by_parent(self, parent_id: str, limit: int = 1000) -> list[Episode]:
        """Return child episodes of a parent episode."""
        ...

    def threads(self, limit: int = 100) -> list[str]:
        """Return distinct thread IDs."""
        ...

    def update(self, episode: Episode) -> None:
        """Replace an existing episode in place."""
        ...

    def annotate(self, episode_id: str, annotation: dict) -> Episode | None:
        """Attach a structured annotation to an episode."""
        ...


@runtime_checkable
class SemanticStoreProtocol(Protocol):
    """Interface for semantic fact backends."""

    def put(self, fact: Fact) -> None:
        """Insert or update a fact."""
        ...

    def get(self, fact_id: str) -> Fact | None:
        """Retrieve a single fact by ID."""
        ...

    def find(self, subject: str,
             predicate: str | None = None,
             at_time: datetime | None = None) -> list[Fact]:
        """Find active facts matching subject (and optionally predicate/time)."""
        ...

    def search(self, keyword: str, limit: int = 10) -> list[Fact]:
        """Search facts by keyword across all fields."""
        ...

    def count(self) -> int:
        """Total number of active facts."""
        ...

    # ── Maintenance methods used by consolidation.py ──

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None,
              valid_from: datetime | None = None,
              valid_until: datetime | None = None,
              recurrence: str | None = None) -> tuple[Fact, str]:
        """Learn a fact with conflict resolution.

        Returns (fact, action) where action is one of
        'new' | 'reinforced' | 'contradicted'.
        """
        ...

    def edit_fact(self, fact_id: str,
                  obj: str | None = None,
                  predicate: str | None = None,
                  confidence: float | None = None,
                  source_episode_ids: list[str] | None = None,
                  is_active: bool | None = None) -> Fact | None:
        """Edit a fact in place."""
        ...

    def correct_fact(self, fact_id: str,
                     new_object: str | None = None,
                     new_predicate: str | None = None,
                     new_confidence: float | None = None) -> tuple[Fact, str]:
        """Correct a fact, deactivating the old version."""
        ...
