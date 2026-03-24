"""Tests for retrieval engine: scoring, ranking, edge cases."""

import math
from datetime import datetime, timedelta

from agentmemory.models import Episode, RetrievalResult
from agentmemory.config import MemoryConfig
from agentmemory.episodic import EpisodicStore
from agentmemory.retrieval import (
    recency_score, relevance_score, score_episode, recall,
)


NOW = datetime(2026, 3, 25, 12, 0, 0)


# ── Scoring functions ──────────────────────────────────────────────

class TestRecencyScore:

    def test_now_is_1(self):
        ep = Episode(content="x", timestamp=NOW)
        assert recency_score(ep, NOW) == 1.0

    def test_decays_over_time(self):
        ep = Episode(content="x", timestamp=NOW - timedelta(hours=24))
        score = recency_score(ep, NOW)
        assert 0 < score < 1

    def test_older_is_lower(self):
        recent = Episode(content="x", timestamp=NOW - timedelta(hours=1))
        old = Episode(content="x", timestamp=NOW - timedelta(days=7))
        assert recency_score(recent, NOW) > recency_score(old, NOW)

    def test_decay_rate_affects_speed(self):
        ep = Episode(content="x", timestamp=NOW - timedelta(hours=24))
        slow = recency_score(ep, NOW, decay_rate=0.01)
        fast = recency_score(ep, NOW, decay_rate=0.1)
        assert slow > fast  # Slow decay retains more

    def test_future_timestamp_clamps(self):
        """Episode in the future should still have score ~1."""
        ep = Episode(content="x", timestamp=NOW + timedelta(hours=5))
        assert recency_score(ep, NOW) == 1.0


class TestRelevanceScore:

    def test_exact_match(self):
        assert relevance_score("coffee", "coffee") == 1.0

    def test_partial_match(self):
        score = relevance_score("black coffee", "I ordered black tea")
        assert 0 < score < 1

    def test_no_match(self):
        assert relevance_score("weather", "I ordered coffee") == 0.0

    def test_empty_query(self):
        assert relevance_score("", "some content") == 0.0

    def test_empty_content(self):
        assert relevance_score("query", "") == 0.0

    def test_case_insensitive(self):
        assert relevance_score("Coffee", "COFFEE is great") > 0


# ── Combined scoring ──────────────────────────────────────────────

class TestScoreEpisode:

    def test_returns_retrieval_result(self):
        ep = Episode(content="User likes coffee", timestamp=NOW, importance=0.7)
        config = MemoryConfig()
        result = score_episode(ep, "coffee", NOW, config)
        assert isinstance(result, RetrievalResult)
        assert result.episode is ep
        assert result.score > 0

    def test_components_match(self):
        ep = Episode(content="User likes coffee", timestamp=NOW, importance=0.7)
        config = MemoryConfig()
        result = score_episode(ep, "coffee", NOW, config)
        # Verify the combined score matches the weighted sum
        expected = (config.w_recency * result.recency +
                    config.w_relevance * result.relevance +
                    config.w_importance * result.importance)
        assert abs(result.score - expected) < 0.001

    def test_high_importance_boosts_score(self):
        low = Episode(content="X", timestamp=NOW, importance=0.1)
        high = Episode(content="X", timestamp=NOW, importance=0.9)
        config = MemoryConfig()
        s_low = score_episode(low, "X", NOW, config)
        s_high = score_episode(high, "X", NOW, config)
        assert s_high.score > s_low.score


# ── Full recall ────────────────────────────────────────────────────

class TestRecall:

    def _seed_store(self) -> EpisodicStore:
        store = EpisodicStore(":memory:")
        store.record(Episode(
            content="User ordered black coffee from Blue Tokai",
            timestamp=NOW - timedelta(hours=2), importance=0.3,
        ))
        store.record(Episode(
            content="User got promoted to Director of Engineering",
            timestamp=NOW - timedelta(days=1), importance=0.95,
        ))
        store.record(Episode(
            content="User checked weather. 28C, sunny.",
            timestamp=NOW - timedelta(hours=5), importance=0.1,
        ))
        store.record(Episode(
            content="Critical server outage. 2 hours downtime.",
            timestamp=NOW - timedelta(days=3), importance=0.9,
        ))
        store.record(Episode(
            content="User went for a morning run. 5km.",
            timestamp=NOW - timedelta(hours=1), importance=0.3,
        ))
        return store

    def test_returns_k_results(self):
        store = self._seed_store()
        results = recall(store, "coffee", k=3, now=NOW)
        assert len(results) <= 3

    def test_relevant_ranked_first(self):
        store = self._seed_store()
        results = recall(store, "coffee", k=5, now=NOW)
        assert "coffee" in results[0].episode.content.lower()

    def test_scores_descending(self):
        store = self._seed_store()
        results = recall(store, "server outage", k=5, now=NOW)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_increments_access_count(self):
        store = self._seed_store()
        results = recall(store, "coffee", k=1, now=NOW)
        ep = store.get(results[0].episode.id)
        assert ep.access_count >= 1

    def test_custom_config(self):
        store = self._seed_store()
        # Weight importance very heavily
        config = MemoryConfig(w_recency=0.0, w_relevance=0.0, w_importance=1.0)
        results = recall(store, "anything", k=1, config=config, now=NOW)
        # Should return the highest-importance episode
        assert results[0].episode.importance >= 0.9

    def test_empty_store(self):
        store = EpisodicStore(":memory:")
        results = recall(store, "hello", k=5, now=NOW)
        assert results == []

    def test_empty_query(self):
        store = self._seed_store()
        results = recall(store, "", k=3, now=NOW)
        assert len(results) <= 3  # Should not crash
