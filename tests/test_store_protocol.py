"""Tests for storage backend protocol completeness."""

from datetime import datetime, timedelta
from typing import Iterator

import pytest

from remembox._store import EpisodicStoreProtocol, SemanticStoreProtocol
from remembox.config import MemoryConfig
from remembox.consolidation import consolidate
from remembox.forgetting import forget
from remembox.models import Episode, Fact


class FakeEpisodicStore(EpisodicStoreProtocol):
    """Minimal in-memory episodic store implementing the protocol."""

    def __init__(self, episodes: list[Episode] | None = None):
        self._episodes = {ep.id: ep for ep in (episodes or [])}
        self._deleted: list[str] = []
        self._consolidated: list[str] = []

    def record(self, episode: Episode) -> None:
        self._episodes[episode.id] = episode

    def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    def recent(self, n: int = 10) -> list[Episode]:
        return sorted(self._episodes.values(), key=lambda e: e.timestamp, reverse=True)[:n]

    def search(self, keyword: str, limit: int = 10) -> list[Episode]:
        return [ep for ep in self._episodes.values() if keyword.lower() in ep.content.lower()][:limit]

    def count(self) -> int:
        return len(self._episodes)

    def iter_all(self, batch_size: int = 500) -> Iterator[Episode]:
        yield from self._episodes.values()

    def delete(self, episode_ids: list[str]) -> int:
        for eid in episode_ids:
            self._episodes.pop(eid, None)
            self._deleted.append(eid)
        return len(episode_ids)

    def mark_consolidated(self, episode_ids: list[str]) -> int:
        for eid in episode_ids:
            ep = self._episodes.get(eid)
            if ep:
                ep.consolidated = True
                self._consolidated.append(eid)
        return len(episode_ids)

    def unconsolidated(self, limit: int = 100) -> list[Episode]:
        return [ep for ep in self._episodes.values() if not ep.consolidated][:limit]

    def by_time_range(self, start: datetime, end: datetime) -> list[Episode]:
        return [ep for ep in self._episodes.values() if start <= ep.timestamp <= end]

    def increment_access(self, episode_id: str) -> None:
        ep = self._episodes.get(episode_id)
        if ep:
            ep.access_count += 1


class FakeSemanticStore(SemanticStoreProtocol):
    """Minimal in-memory semantic store implementing the protocol."""

    def __init__(self):
        self._facts: list[Fact] = []

    def put(self, fact: Fact) -> None:
        self._facts.append(fact)

    def get(self, fact_id: str) -> Fact | None:
        for f in self._facts:
            if f.id == fact_id:
                return f
        return None

    def find(self, subject: str, predicate: str | None = None) -> list[Fact]:
        return [
            f for f in self._facts
            if f.subject == subject and f.is_active and (predicate is None or f.predicate == predicate)
        ]

    def search(self, keyword: str, limit: int = 10) -> list[Fact]:
        return [
            f for f in self._facts
            if keyword.lower() in (f.subject + f.predicate + f.object).lower()
        ][:limit]

    def count(self) -> int:
        return len([f for f in self._facts if f.is_active])

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None) -> tuple[Fact, str]:
        fact = Fact(subject=subject, predicate=predicate, object=obj, confidence=confidence)
        self._facts.append(fact)
        return fact, "new"


class TestProtocolCompleteness:
    """Verify maintenance modules only call protocol-declared methods."""

    def test_forget_runs_on_protocol_store(self):
        now = datetime(2026, 3, 25, 12, 0, 0)
        old_trivial = Episode(
            content="old trivial",
            importance=0.1,
            timestamp=now - timedelta(days=30),
        )
        critical = Episode(
            content="critical event",
            importance=0.95,
            timestamp=now - timedelta(days=30),
        )
        store = FakeEpisodicStore([old_trivial, critical])

        result = forget(store, now=now)

        assert result["deleted"] >= 1
        assert result["kept"] >= 1
        assert old_trivial.id in store._deleted
        assert critical.id not in store._deleted

    def test_consolidate_runs_on_protocol_stores(self):
        now = datetime(2026, 3, 25, 12, 0, 0)
        ep = Episode(
            content="I prefer black coffee",
            timestamp=now - timedelta(hours=2),
        )
        episodic = FakeEpisodicStore([ep])
        semantic = FakeSemanticStore()

        config = MemoryConfig(consolidation_min_age_hours=1.0)
        result = consolidate(episodic, semantic, config=config, now=now)

        assert result["episodes_processed"] == 1
        assert result["facts_extracted"] >= 1
        assert ep.consolidated is True
        assert semantic.count() >= 1

    def test_runtime_checkable_protocol(self):
        """The protocol should be runtime_checkable for isinstance checks."""
        store = FakeEpisodicStore()
        assert isinstance(store, EpisodicStoreProtocol)
