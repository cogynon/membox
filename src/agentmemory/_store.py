"""Storage backend protocol. Implement this to plug in any database."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentmemory.models import Episode, Fact


@runtime_checkable
class EpisodicStoreProtocol(Protocol):
    """Interface for episodic memory backends.

    Implement these 5 methods to support a new storage backend
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


@runtime_checkable
class SemanticStoreProtocol(Protocol):
    """Interface for semantic fact backends."""

    def put(self, fact: Fact) -> None:
        """Insert or update a fact."""
        ...

    def get(self, fact_id: str) -> Fact | None:
        """Retrieve a single fact by ID."""
        ...

    def find(self, subject: str, predicate: str | None = None) -> list[Fact]:
        """Find active facts matching subject (and optionally predicate)."""
        ...

    def search(self, keyword: str, limit: int = 10) -> list[Fact]:
        """Search facts by keyword across all fields."""
        ...

    def count(self) -> int:
        """Total number of active facts."""
        ...
