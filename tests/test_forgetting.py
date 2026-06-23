"""Tests for forgetting and consolidation."""

from datetime import datetime, timedelta

from membox.models import Episode
from membox.config import MemoryConfig
from membox.episodic import EpisodicStore
from membox.semantic import SemanticStore
from membox.forgetting import retention_score, evaluate_episode, forget
from membox.consolidation import RuleBasedConsolidator, consolidate


NOW = datetime(2026, 3, 25, 12, 0, 0)


# ── Retention scoring ──────────────────────────────────────────────

class TestRetentionScore:

    def test_recent_high_importance_scores_high(self):
        ep = Episode(content="X", timestamp=NOW, importance=0.9)
        score = retention_score(ep, NOW, MemoryConfig())
        assert score > 0.5

    def test_old_low_importance_scores_low(self):
        ep = Episode(content="X", timestamp=NOW - timedelta(days=30), importance=0.1)
        score = retention_score(ep, NOW, MemoryConfig())
        assert score < 0.2

    def test_access_boosts_score(self):
        ep_no_access = Episode(content="X", timestamp=NOW - timedelta(days=5), importance=0.3)
        ep_accessed = Episode(content="X", timestamp=NOW - timedelta(days=5),
                              importance=0.3, access_count=5)
        s1 = retention_score(ep_no_access, NOW, MemoryConfig())
        s2 = retention_score(ep_accessed, NOW, MemoryConfig())
        assert s2 > s1

    def test_consolidated_penalty(self):
        ep_fresh = Episode(content="X", timestamp=NOW, importance=0.5)
        ep_cons = Episode(content="X", timestamp=NOW, importance=0.5, consolidated=True)
        s1 = retention_score(ep_fresh, NOW, MemoryConfig())
        s2 = retention_score(ep_cons, NOW, MemoryConfig())
        assert s1 > s2

    def test_clamped_0_to_1(self):
        ep = Episode(content="X", timestamp=NOW - timedelta(days=365), importance=0.0)
        score = retention_score(ep, NOW, MemoryConfig())
        assert 0.0 <= score <= 1.0


# ── Evaluate episode ───────────────────────────────────────────────

class TestEvaluateEpisode:

    def test_old_low_importance_gets_deleted(self):
        ep = Episode(content="X", timestamp=NOW - timedelta(days=10), importance=0.1)
        action = evaluate_episode(ep, NOW, MemoryConfig())
        assert action.action == "delete"

    def test_high_importance_kept(self):
        ep = Episode(content="X", timestamp=NOW - timedelta(days=10), importance=0.95)
        action = evaluate_episode(ep, NOW, MemoryConfig())
        assert action.action == "keep"

    def test_recent_low_importance_kept(self):
        ep = Episode(content="X", timestamp=NOW - timedelta(days=1), importance=0.1)
        action = evaluate_episode(ep, NOW, MemoryConfig())
        assert action.action == "keep"  # Only 1 day old, below 7-day threshold


# ── Forget ──────────────────────────────────────────────────────────

class TestForget:

    def test_deletes_old_trivial(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="old trivial",
                             timestamp=NOW - timedelta(days=30), importance=0.1))
        store.record(Episode(content="important",
                             timestamp=NOW - timedelta(days=30), importance=0.95))
        result = forget(store, config=MemoryConfig(), now=NOW)
        assert result["deleted"] >= 1
        assert store.count() >= 1  # Important one survives

    def test_keeps_recent(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="recent", timestamp=NOW, importance=0.1))
        result = forget(store, config=MemoryConfig(), now=NOW)
        assert result["deleted"] == 0
        assert store.count() == 1

    def test_empty_store(self):
        store = EpisodicStore(":memory:")
        result = forget(store, now=NOW)
        assert result["total_evaluated"] == 0

    def test_fast_config_more_aggressive(self):
        store = EpisodicStore(":memory:")
        store.record(Episode(content="X",
                             timestamp=NOW - timedelta(days=5), importance=0.2))
        result_fast = forget(store, config=MemoryConfig.fast(), now=NOW)
        assert result_fast["deleted"] >= 1


# ── Rule-based consolidator ────────────────────────────────────────

class TestRuleBasedConsolidator:

    def test_extracts_preference(self):
        c = RuleBasedConsolidator()
        facts = c.extract(["User said: I prefer black coffee"])
        assert len(facts) >= 1
        assert any(f["predicate"] == "prefers" for f in facts)

    def test_extracts_location(self):
        c = RuleBasedConsolidator()
        facts = c.extract(["I live in Mumbai"])
        assert any(f["predicate"] == "lives_in" and f["object"] == "Mumbai" for f in facts)

    def test_extracts_work(self):
        c = RuleBasedConsolidator()
        facts = c.extract(["I work at Google"])
        assert any(f["predicate"] == "works_at" and f["object"] == "Google" for f in facts)

    def test_no_extraction_from_irrelevant(self):
        c = RuleBasedConsolidator()
        facts = c.extract(["The weather is nice today"])
        assert len(facts) == 0

    def test_handles_sentence_boundaries(self):
        c = RuleBasedConsolidator()
        facts = c.extract(["I live in Delhi and work at Google."])
        locations = [f for f in facts if f["predicate"] == "lives_in"]
        assert len(locations) >= 1
        assert locations[0]["object"] == "Delhi"  # Not "Delhi and work at Google"


# ── Consolidate pipeline ───────────────────────────────────────────

class TestConsolidate:

    def test_extracts_and_learns(self):
        episodic = EpisodicStore(":memory:")
        semantic = SemanticStore(":memory:")
        episodic.record(Episode(
            content="User said: I prefer green tea",
            timestamp=NOW - timedelta(hours=2), importance=0.5,
        ))
        result = consolidate(episodic, semantic, now=NOW)
        assert result["episodes_processed"] >= 1
        assert result["facts_extracted"] >= 1
        assert semantic.count() >= 1

    def test_marks_as_consolidated(self):
        episodic = EpisodicStore(":memory:")
        semantic = SemanticStore(":memory:")
        episodic.record(Episode(
            content="I live in Bangalore",
            timestamp=NOW - timedelta(hours=2),
        ))
        consolidate(episodic, semantic, now=NOW)
        assert len(episodic.unconsolidated()) == 0

    def test_skips_too_recent(self):
        episodic = EpisodicStore(":memory:")
        semantic = SemanticStore(":memory:")
        episodic.record(Episode(
            content="I prefer tea",
            timestamp=NOW - timedelta(minutes=5),  # Less than min_age
        ))
        result = consolidate(episodic, semantic, now=NOW)
        assert result["episodes_processed"] == 0

    def test_empty_store(self):
        episodic = EpisodicStore(":memory:")
        semantic = SemanticStore(":memory:")
        result = consolidate(episodic, semantic, now=NOW)
        assert result["episodes_processed"] == 0
