"""Integration & E2E tests for Membox facade."""

import time
from datetime import datetime, timedelta

import pytest

from membox import Membox, MemoryConfig, Episode, Fact


NOW = datetime(2026, 3, 25, 12, 0, 0)


# ── Basic API ──────────────────────────────────────────────────────

class TestMemboxBasic:

    def test_three_line_usage(self):
        """The 3-line promise: init, record, recall."""
        m = Membox(":memory:")
        m.record("User loves hiking in the Himalayas", importance=0.8)
        results = m.recall("hiking", k=1, now=NOW)
        assert len(results) >= 1
        assert "hiking" in results[0].episode.content

    def test_record_returns_episode(self):
        m = Membox(":memory:")
        ep = m.record("hello", importance=0.7, emotion="happy")
        assert isinstance(ep, Episode)
        assert ep.importance == 0.7
        assert ep.emotion == "happy"

    def test_recall_empty(self):
        m = Membox(":memory:")
        results = m.recall("anything", now=NOW)
        assert results == []

    def test_recent(self):
        m = Membox(":memory:")
        m.record("a", timestamp=NOW - timedelta(hours=2))
        m.record("b", timestamp=NOW - timedelta(hours=1))
        m.record("c", timestamp=NOW)
        recent = m.recent(2)
        assert len(recent) == 2
        assert recent[0].content == "c"

    def test_search(self):
        m = Membox(":memory:")
        m.record("User ordered black coffee")
        m.record("User went for a run")
        results = m.search("coffee")
        assert len(results) == 1


# ── Semantic: learn + about ────────────────────────────────────────

class TestMemboxSemantic:

    def test_learn_and_about(self):
        m = Membox(":memory:")
        fact, action = m.learn("user", "prefers", "coffee", confidence=0.9)
        assert action == "new"
        facts = m.about("user")
        assert len(facts) == 1
        assert facts[0].object == "coffee"

    def test_reinforce(self):
        m = Membox(":memory:")
        m.learn("user", "prefers", "coffee", confidence=0.5)
        fact, action = m.learn("user", "prefers", "coffee")
        assert action == "reinforced"
        assert fact.confidence > 0.5

    def test_contradict(self):
        m = Membox(":memory:")
        m.learn("user", "lives_in", "Delhi")
        fact, action = m.learn("user", "lives_in", "Mumbai")
        assert action == "contradicted"
        facts = m.about("user")
        assert len(facts) == 1
        assert facts[0].object == "Mumbai"

    def test_find_fact(self):
        m = Membox(":memory:")
        m.learn("user", "name", "Pranav")
        m.learn("user", "prefers", "coffee")
        facts = m.find_fact("user", "name")
        assert len(facts) == 1
        assert facts[0].object == "Pranav"


# ── Context builder ────────────────────────────────────────────────

class TestMemboxContext:

    def test_context_includes_facts(self):
        m = Membox(":memory:")
        m.learn("user", "prefers", "coffee", confidence=0.9)
        ctx = m.context("what does the user like?", now=NOW)
        assert "coffee" in ctx
        assert "User Profile" in ctx

    def test_context_includes_memories(self):
        m = Membox(":memory:")
        m.record("User went hiking in the Himalayas", importance=0.8,
                 timestamp=NOW - timedelta(hours=2))
        ctx = m.context("hiking", now=NOW)
        assert "hiking" in ctx.lower()

    def test_context_respects_token_budget(self):
        m = Membox(":memory:")
        for i in range(50):
            m.record(f"Episode {i} about a very long topic " * 10,
                     timestamp=NOW - timedelta(hours=i))
        ctx = m.context("topic", max_tokens=200, now=NOW)
        # Rough check: shouldn't exceed budget by much
        assert len(ctx) // 4 < 300  # Some overhead is OK

    def test_context_empty(self):
        m = Membox(":memory:")
        ctx = m.context("anything", now=NOW)
        assert ctx == ""


# ── Consolidation ──────────────────────────────────────────────────

class TestMemboxConsolidate:

    def test_consolidate(self):
        m = Membox(":memory:")
        m.record("User said: I prefer green tea",
                 timestamp=NOW - timedelta(hours=2))
        result = m.consolidate(now=NOW)
        assert result["episodes_processed"] >= 1
        assert result["facts_extracted"] >= 1

    def test_consolidate_empty(self):
        m = Membox(":memory:")
        result = m.consolidate(now=NOW)
        assert result["episodes_processed"] == 0


# ── Forgetting ─────────────────────────────────────────────────────

class TestMemboxForget:

    def test_forget_removes_old_trivial(self):
        m = Membox(":memory:")
        m.record("old trivial", importance=0.1,
                 timestamp=NOW - timedelta(days=30))
        m.record("important", importance=0.95,
                 timestamp=NOW - timedelta(days=30))
        result = m.forget(now=NOW)
        assert result["deleted"] >= 1

    def test_forget_keeps_important(self):
        m = Membox(":memory:")
        m.record("critical event", importance=0.95,
                 timestamp=NOW - timedelta(days=5))
        result = m.forget(now=NOW)
        assert result["deleted"] == 0


# ── Stats ──────────────────────────────────────────────────────────

class TestMemboxStats:

    def test_stats_structure(self):
        m = Membox(":memory:")
        m.record("test")
        m.learn("user", "name", "X")
        s = m.stats()
        assert "episodes" in s
        assert "facts" in s
        assert s["episodes"]["total"] == 1
        assert s["facts"]["active"] == 1


# ── Config presets ─────────────────────────────────────────────────

class TestMemboxConfig:

    def test_fast_preset(self):
        m = Membox(":memory:", config=MemoryConfig.fast())
        m.record("test", importance=0.1,
                 timestamp=NOW - timedelta(days=5))
        result = m.forget(now=NOW)
        assert result["deleted"] >= 1

    def test_deep_preset(self):
        m = Membox(":memory:", config=MemoryConfig.deep())
        m.record("test", importance=0.1,
                 timestamp=NOW - timedelta(days=5))
        result = m.forget(now=NOW)
        assert result["deleted"] == 0  # Deep retains longer


# ── Context manager ────────────────────────────────────────────────

class TestMemboxLifecycle:

    def test_context_manager(self):
        with Membox(":memory:") as m:
            m.record("inside")
            assert m.recall("inside", k=1, now=NOW)

    def test_repr(self):
        m = Membox(":memory:")
        m.record("x")
        assert "episodes=1" in repr(m)

    def test_persistence(self, tmp_db):
        m1 = Membox(tmp_db)
        m1.record("persistent data", importance=0.9)
        m1.learn("user", "name", "Test")
        m1.close()

        m2 = Membox(tmp_db)
        assert m2.recent(1)[0].content == "persistent data"
        assert m2.about("user")[0].object == "Test"
        m2.close()


# ── E2E full loop ──────────────────────────────────────────────────

class TestMemboxE2E:

    def test_full_lifecycle(self):
        """End-to-end: record → learn → recall → context → consolidate → forget → stats."""
        m = Membox(":memory:")

        # Record episodes
        m.record("User said: I prefer black coffee", importance=0.6,
                 timestamp=NOW - timedelta(hours=5))
        m.record("User got promoted to Director", importance=1.0,
                 timestamp=NOW - timedelta(hours=3))
        m.record("User ordered lunch", importance=0.2,
                 timestamp=NOW - timedelta(hours=2))

        # Learn facts
        m.learn("user", "name", "Pranav", confidence=0.95)
        m.learn("user", "works_at", "Google", confidence=0.9)

        # Recall
        results = m.recall("promotion", k=2, now=NOW)
        assert len(results) >= 1

        # Context
        ctx = m.context("what happened today?", now=NOW)
        assert "Pranav" in ctx or "promoted" in ctx.lower() or "User Profile" in ctx

        # Consolidate
        consol = m.consolidate(now=NOW)
        assert consol["episodes_processed"] >= 1

        # Stats
        s = m.stats()
        assert s["episodes"]["total"] >= 3
        assert s["facts"]["active"] >= 2

        # Forget (nothing old enough with default config)
        result = m.forget(now=NOW)
        assert result["total_evaluated"] >= 3


# ── Scale test ─────────────────────────────────────────────────────

class TestMemboxScale:

    def test_recall_latency_at_scale(self):
        """1000 episodes, recall should complete in < 100ms."""
        m = Membox(":memory:")
        for i in range(1000):
            m.record(f"event {i} about topic-{i % 20}",
                     importance=(i % 10) / 10,
                     timestamp=NOW - timedelta(hours=i))

        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            m.recall(f"topic-{i % 20}", k=5, now=NOW)
            latencies.append(time.perf_counter() - t0)
        p99 = sorted(latencies)[48]
        assert p99 < 0.1, f"recall p99={p99*1000:.1f}ms (limit: 100ms)"

    def test_context_latency(self):
        """Context generation should be fast."""
        m = Membox(":memory:")
        for i in range(100):
            m.record(f"event {i}", timestamp=NOW - timedelta(hours=i))
        m.learn("user", "name", "Test")

        t0 = time.perf_counter()
        m.context("test query", now=NOW)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"context took {elapsed*1000:.1f}ms"
