"""Tests for episodic store: CRUD, search, persistence, batch ops, scale."""

import time
from datetime import datetime, timedelta

import pytest

from membox.models import Episode
from membox.episodic import EpisodicStore


# ── Basic CRUD ──────────────────────────────────────────────────────

class TestEpisodicCRUD:

    def test_record_and_get(self):
        store = EpisodicStore(":memory:")
        ep = Episode(content="hello world", importance=0.7)
        store.record(ep)
        retrieved = store.get(ep.id)
        assert retrieved is not None
        assert retrieved.content == "hello world"
        assert retrieved.importance == 0.7

    def test_get_nonexistent(self):
        store = EpisodicStore(":memory:")
        assert store.get("nonexistent") is None

    def test_count(self):
        store = EpisodicStore(":memory:")
        assert store.count() == 0
        store.record(Episode(content="a"))
        store.record(Episode(content="b"))
        assert store.count() == 2

    def test_delete(self):
        store = EpisodicStore(":memory:")
        ep = Episode(content="to be deleted")
        store.record(ep)
        assert store.count() == 1
        deleted = store.delete([ep.id])
        assert deleted == 1
        assert store.count() == 0

    def test_delete_nonexistent(self):
        store = EpisodicStore(":memory:")
        deleted = store.delete(["nope"])
        assert deleted == 0

    def test_upsert_on_record(self):
        """Recording an episode with the same ID should update, not duplicate."""
        store = EpisodicStore(":memory:")
        ep = Episode(content="original", id="fixed-id")
        store.record(ep)
        ep_updated = Episode(content="updated", id="fixed-id", importance=0.9)
        store.record(ep_updated)
        assert store.count() == 1
        retrieved = store.get("fixed-id")
        assert retrieved.content == "updated"
        assert retrieved.importance == 0.9


# ── Query methods ───────────────────────────────────────────────────

class TestEpisodicQueries:

    def test_recent(self):
        store = EpisodicStore(":memory:")
        now = datetime.now()
        for i in range(5):
            store.record(Episode(
                content=f"event {i}",
                timestamp=now - timedelta(hours=5 - i),
            ))
        recent = store.recent(3)
        assert len(recent) == 3
        assert recent[0].content == "event 4"  # Most recent
        assert recent[2].content == "event 2"

    def test_search(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="User ordered black coffee"))
        store.record(Episode(content="User went for a run"))
        store.record(Episode(content="User drank green tea"))
        results = store.search("coffee")
        assert len(results) == 1
        assert "coffee" in results[0].content

    def test_search_case_insensitive(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="User loves COFFEE"))
        results = store.search("coffee")
        assert len(results) == 1

    def test_by_importance(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="trivial", importance=0.1))
        store.record(Episode(content="moderate", importance=0.5))
        store.record(Episode(content="critical", importance=0.95))
        results = store.by_importance(0.7)
        assert len(results) == 1
        assert results[0].content == "critical"

    def test_unconsolidated(self):
        store = EpisodicStore(":memory:")
        ep1 = Episode(content="not consolidated")
        ep2 = Episode(content="already consolidated", consolidated=True)
        store.record(ep1)
        store.record(ep2)
        results = store.unconsolidated()
        assert len(results) == 1
        assert results[0].content == "not consolidated"

    def test_by_time_range(self):
        store = EpisodicStore(":memory:")
        now = datetime.now()
        store.record(Episode(content="yesterday", timestamp=now - timedelta(days=1)))
        store.record(Episode(content="today", timestamp=now))
        store.record(Episode(content="last week", timestamp=now - timedelta(days=7)))
        results = store.by_time_range(
            now - timedelta(days=2),
            now + timedelta(hours=1),
        )
        assert len(results) == 2
        assert results[0].content == "yesterday"
        assert results[1].content == "today"


# ── Batch operations ────────────────────────────────────────────────

class TestEpisodicBatch:

    def test_batch_insert(self):
        store = EpisodicStore(":memory:")
        episodes = [Episode(content=f"batch {i}") for i in range(100)]
        store.record_batch(episodes)
        assert store.count() == 100

    def test_batch_empty(self):
        store = EpisodicStore(":memory:")
        store.record_batch([])  # Should not raise
        assert store.count() == 0

    def test_mark_consolidated(self):
        store = EpisodicStore(":memory:")
        eps = [Episode(content=f"ep {i}") for i in range(5)]
        store.record_batch(eps)
        updated = store.mark_consolidated([eps[0].id, eps[1].id])
        assert updated == 2
        uncons = store.unconsolidated()
        assert len(uncons) == 3


# ── Update operations ──────────────────────────────────────────────

class TestEpisodicUpdates:

    def test_increment_access(self):
        store = EpisodicStore(":memory:")
        ep = Episode(content="accessed")
        store.record(ep)
        store.increment_access(ep.id)
        store.increment_access(ep.id)
        retrieved = store.get(ep.id)
        assert retrieved.access_count == 2

    def test_delete_before(self):
        store = EpisodicStore(":memory:")
        now = datetime.now()
        store.record(Episode(content="old", timestamp=now - timedelta(days=30)))
        store.record(Episode(content="recent", timestamp=now))
        deleted = store.delete_before(now - timedelta(days=7))
        assert deleted == 1
        assert store.count() == 1
        assert store.recent(1)[0].content == "recent"


# ── Stats ───────────────────────────────────────────────────────────

class TestEpisodicStats:

    def test_stats(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="a", importance=0.2))
        store.record(Episode(content="b", importance=0.8, consolidated=True))
        s = store.stats()
        assert s["total"] == 2
        assert s["avg_importance"] == 0.5
        assert s["consolidated"] == 1
        assert s["unconsolidated"] == 1

    def test_stats_empty(self):
        store = EpisodicStore(":memory:")
        s = store.stats()
        assert s["total"] == 0


# ── Persistence ─────────────────────────────────────────────────────

class TestEpisodicPersistence:

    def test_survives_reopen(self, tmp_db):
        """Data persists across store instances (same file)."""
        store1 = EpisodicStore(tmp_db)
        store1.record(Episode(content="persistent data", importance=0.9))
        store1.close()

        store2 = EpisodicStore(tmp_db)
        assert store2.count() == 1
        ep = store2.recent(1)[0]
        assert ep.content == "persistent data"
        assert ep.importance == 0.9
        store2.close()


# ── Iteration ───────────────────────────────────────────────────────

class TestEpisodicIteration:

    def test_iter_all(self):
        store = EpisodicStore(":memory:")
        store.record_batch([Episode(content=f"ep {i}") for i in range(50)])
        all_eps = list(store.iter_all(batch_size=20))
        assert len(all_eps) == 50

    def test_iter_empty(self):
        store = EpisodicStore(":memory:")
        assert list(store.iter_all()) == []


# ── Context Manager ─────────────────────────────────────────────────

class TestEpisodicContextManager:

    def test_with_statement(self):
        with EpisodicStore(":memory:") as store:
            store.record(Episode(content="in context"))
            assert store.count() == 1


# ── Scale ───────────────────────────────────────────────────────────

class TestEpisodicScale:

    def test_insert_100k(self):
        """100K inserts should complete in under 10 seconds."""
        store = EpisodicStore(":memory:")
        n = 100_000
        episodes = [
            Episode(content=f"event {i}", importance=(i % 10) / 10)
            for i in range(n)
        ]
        t0 = time.perf_counter()
        store.record_batch(episodes)
        elapsed = time.perf_counter() - t0
        assert store.count() == n
        assert elapsed < 10.0, f"100K inserts took {elapsed:.1f}s (limit: 10s)"

    def test_query_latency_at_100k(self):
        """Queries on 100K rows should be < 50ms at p99."""
        store = EpisodicStore(":memory:")
        now = datetime.now()
        episodes = [
            Episode(
                content=f"event {i} about topic-{i % 50}",
                importance=(i % 10) / 10,
                timestamp=now - timedelta(hours=i),
            )
            for i in range(100_000)
        ]
        store.record_batch(episodes)

        # Benchmark: recent()
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            store.recent(20)
            latencies.append(time.perf_counter() - t0)
        p99 = sorted(latencies)[98]
        assert p99 < 0.05, f"recent() p99={p99*1000:.1f}ms (limit: 50ms)"

        # Benchmark: search()
        latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            store.search(f"topic-{i % 50}", limit=10)
            latencies.append(time.perf_counter() - t0)
        p99 = sorted(latencies)[98]
        assert p99 < 0.1, f"search() p99={p99*1000:.1f}ms (limit: 100ms)"

        # Benchmark: by_importance()
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            store.by_importance(0.7, limit=10)
            latencies.append(time.perf_counter() - t0)
        p99 = sorted(latencies)[98]
        assert p99 < 0.05, f"by_importance() p99={p99*1000:.1f}ms (limit: 50ms)"
