"""Tests for retrieval engine: scoring, ranking, edge cases."""

import math
from datetime import datetime, timedelta

import pytest

from membox.models import Episode, RetrievalResult
from membox.config import MemoryConfig
from membox.episodic import EpisodicStore
from membox.retrieval import (
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

    def test_punctuation_does_not_break_tokens(self):
        """Regression: trailing punctuation must not create a distinct token.
        'coffee order' vs 'ordered a black coffee.' should still match on 'coffee'."""
        assert relevance_score("coffee order", "User ordered a black coffee.") == 0.5
        assert relevance_score("coffee!", "i love coffee") == 1.0
        assert relevance_score("(coffee)", "coffee, please") == 1.0


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
        # Verify the combined score is the weighted sum normalized by total weight.
        total_weight = config.w_recency + config.w_relevance + config.w_importance
        expected = (config.w_recency * result.recency +
                    config.w_relevance * result.relevance +
                    config.w_importance * result.importance) / total_weight
        assert abs(result.score - expected) < 0.001

    def test_score_is_normalized_for_custom_weights(self):
        ep = Episode(content="perfect match on every axis", timestamp=NOW, importance=1.0)
        # Weights that sum to more than 1 — score should still be 1.0 for a perfect match.
        config = MemoryConfig(w_recency=2.0, w_relevance=3.0, w_importance=5.0)
        result = score_episode(ep, "perfect match on every axis", NOW, config)
        assert result.score == pytest.approx(1.0, abs=0.001)

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

    def test_min_score_filters_weak_matches(self):
        store = self._seed_store()
        # A high threshold should filter out low-relevance results
        results = recall(store, "quantum physics", k=5, min_score=0.3, now=NOW)
        assert all(r.score >= 0.3 for r in results)

    def test_min_score_can_return_empty(self):
        store = self._seed_store()
        # A very high threshold should produce no results
        results = recall(store, "coffee", k=5, min_score=0.99, now=NOW)
        assert results == []

    def test_finds_match_on_non_first_query_word(self):
        """Regression: recall must keyword-search EVERY query token, not just the
        first word. A relevant old episode beyond the recent pool should be
        reachable even when its matching word is not the first query word."""
        store = EpisodicStore(":memory:")
        # Flood the recent pool so the target is pushed out of recent(100).
        for i in range(120):
            store.record(Episode(
                content=f"filler event number {i}",
                timestamp=NOW - timedelta(hours=i + 1), importance=0.1,
            ))
        store.record(Episode(
            content="I keep my passport in the study desk drawer.",
            timestamp=NOW - timedelta(days=200), importance=0.9,
        ))
        # 'passport' is the LAST word of the query, and old/out-of-pool.
        results = recall(store, "where is my passport", k=5, now=NOW)
        assert any("passport" in r.episode.content for r in results)

    def test_min_score_via_membox(self):
        from membox import Membox
        memory = Membox(":memory:")
        memory.record("User likes coffee", importance=0.5)
        memory.record("User enjoys hiking", importance=0.5)

        results = memory.recall("coffee", k=5, min_score=0.5)
        assert len(results) <= 1  # hiking match should likely be filtered
