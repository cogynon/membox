"""Tests for semantic store: learn, reinforce, contradict, search, persistence."""

from agentmemory.models import Fact
from agentmemory.config import MemoryConfig
from agentmemory.semantic import SemanticStore


# ── Basic learn ─────────────────────────────────────────────────────

class TestSemanticLearn:

    def test_learn_new(self):
        store = SemanticStore(":memory:")
        fact, action = store.learn("user", "prefers", "coffee", confidence=0.8)
        assert action == "new"
        assert fact.subject == "user"
        assert fact.object == "coffee"
        assert fact.confidence == 0.8
        assert fact.is_active is True

    def test_learn_with_source(self):
        store = SemanticStore(":memory:")
        fact, _ = store.learn("user", "likes", "tea", source_episode_id="ep123")
        assert "ep123" in fact.source_episode_ids


# ── Reinforcement ──────────────────────────────────────────────────

class TestSemanticReinforce:

    def test_reinforce_boosts_confidence(self):
        store = SemanticStore(":memory:")
        fact1, a1 = store.learn("user", "prefers", "coffee", confidence=0.5)
        fact2, a2 = store.learn("user", "prefers", "coffee", confidence=0.5)
        assert a1 == "new"
        assert a2 == "reinforced"
        assert fact2.confidence > 0.5  # Boosted

    def test_reinforce_accumulates_sources(self):
        store = SemanticStore(":memory:")
        store.learn("user", "likes", "X", source_episode_id="ep1")
        fact, _ = store.learn("user", "likes", "X", source_episode_id="ep2")
        assert "ep1" in fact.source_episode_ids
        assert "ep2" in fact.source_episode_ids

    def test_reinforce_respects_config(self):
        cfg = MemoryConfig(reinforce_boost_rate=0.5)  # Very aggressive boost
        store = SemanticStore(":memory:", config=cfg)
        store.learn("user", "likes", "X", confidence=0.5)
        fact, _ = store.learn("user", "likes", "X")
        # 0.5 + (1.0 - 0.5) * 0.5 = 0.75
        assert abs(fact.confidence - 0.75) < 0.01

    def test_confidence_never_exceeds_1(self):
        store = SemanticStore(":memory:")
        store.learn("user", "likes", "X", confidence=0.99)
        for _ in range(20):
            fact, _ = store.learn("user", "likes", "X")
        assert fact.confidence <= 1.0


# ── Contradiction ──────────────────────────────────────────────────

class TestSemanticContradict:

    def test_contradict_creates_new(self):
        store = SemanticStore(":memory:")
        f1, a1 = store.learn("user", "lives_in", "Delhi")
        f2, a2 = store.learn("user", "lives_in", "Mumbai")
        assert a1 == "new"
        assert a2 == "contradicted"
        assert f2.object == "Mumbai"

    def test_contradict_deactivates_old(self):
        store = SemanticStore(":memory:")
        f1, _ = store.learn("user", "lives_in", "Delhi")
        f2, _ = store.learn("user", "lives_in", "Mumbai")
        old = store.get(f1.id)
        assert old.is_active is False

    def test_only_active_returned(self):
        store = SemanticStore(":memory:")
        store.learn("user", "lives_in", "Delhi")
        store.learn("user", "lives_in", "Mumbai")
        facts = store.about("user")
        assert len(facts) == 1
        assert facts[0].object == "Mumbai"


# ── Queries ─────────────────────────────────────────────────────────

class TestSemanticQueries:

    def test_about(self):
        store = SemanticStore(":memory:")
        store.learn("user", "name", "Pranav", confidence=0.95)
        store.learn("user", "prefers", "coffee", confidence=0.8)
        facts = store.about("user")
        assert len(facts) == 2
        assert facts[0].confidence >= facts[1].confidence  # Sorted desc

    def test_find_with_predicate(self):
        store = SemanticStore(":memory:")
        store.learn("user", "name", "Pranav")
        store.learn("user", "prefers", "coffee")
        facts = store.find("user", "name")
        assert len(facts) == 1
        assert facts[0].object == "Pranav"

    def test_search(self):
        store = SemanticStore(":memory:")
        store.learn("user", "prefers", "black coffee")
        store.learn("user", "lives_in", "Mumbai")
        results = store.search("coffee")
        assert len(results) == 1
        assert results[0].object == "black coffee"

    def test_all_active(self):
        store = SemanticStore(":memory:")
        store.learn("user", "a", "1")
        store.learn("user", "b", "2")
        store.learn("user", "a", "X")  # Contradicts first
        active = store.all_active()
        assert len(active) == 2  # "a→X" and "b→2"

    def test_get_by_id(self):
        store = SemanticStore(":memory:")
        fact, _ = store.learn("user", "name", "Test")
        retrieved = store.get(fact.id)
        assert retrieved.object == "Test"

    def test_get_nonexistent(self):
        store = SemanticStore(":memory:")
        assert store.get("nope") is None


# ── Delete / Deactivate ────────────────────────────────────────────

class TestSemanticDelete:

    def test_deactivate(self):
        store = SemanticStore(":memory:")
        fact, _ = store.learn("user", "name", "X")
        result = store.deactivate(fact.id)
        assert result is True
        assert store.count() == 0

    def test_hard_delete(self):
        store = SemanticStore(":memory:")
        f1, _ = store.learn("user", "a", "1")
        f2, _ = store.learn("user", "b", "2")
        deleted = store.delete([f1.id])
        assert deleted == 1
        assert store.get(f1.id) is None


# ── Stats ───────────────────────────────────────────────────────────

class TestSemanticStats:

    def test_stats(self):
        store = SemanticStore(":memory:")
        store.learn("user", "a", "1", confidence=0.6)
        store.learn("user", "b", "2", confidence=0.8)
        store.learn("user", "a", "X")  # Contradicts: supersedes "a→1"
        s = store.stats()
        assert s["active"] == 2
        assert s["superseded"] == 1
        assert s["total"] == 3

    def test_stats_empty(self):
        store = SemanticStore(":memory:")
        s = store.stats()
        assert s["total"] == 0
        assert s["active"] == 0


# ── Persistence ─────────────────────────────────────────────────────

class TestSemanticPersistence:

    def test_survives_reopen(self, tmp_db):
        store1 = SemanticStore(tmp_db)
        store1.learn("user", "name", "Pranav", confidence=0.95)
        store1.close()

        store2 = SemanticStore(tmp_db)
        facts = store2.about("user")
        assert len(facts) == 1
        assert facts[0].object == "Pranav"
        store2.close()


# ── Context manager ─────────────────────────────────────────────────

class TestSemanticContextManager:

    def test_with_statement(self):
        with SemanticStore(":memory:") as store:
            store.learn("user", "x", "y")
            assert store.count() == 1
