"""Unit tests for data models and config."""

import json
from datetime import datetime, timedelta

from membox.models import Episode, Fact, RetrievalResult
from membox.config import MemoryConfig


# ── Episode ─────────────────────────────────────────────────────────

class TestEpisode:

    def test_defaults(self):
        ep = Episode(content="hello")
        assert ep.content == "hello"
        assert ep.importance == 0.5
        assert ep.emotion is None
        assert ep.source == "conversation"
        assert ep.context == {}
        assert ep.consolidated is False
        assert ep.access_count == 0
        assert len(ep.id) == 16

    def test_custom_fields(self):
        ts = datetime(2026, 1, 1, 12, 0)
        ep = Episode(
            content="important event",
            timestamp=ts,
            importance=0.95,
            emotion="ecstatic",
            source="email",
            context={"key": "value"},
        )
        assert ep.importance == 0.95
        assert ep.emotion == "ecstatic"
        assert ep.timestamp == ts
        assert ep.context["key"] == "value"

    def test_roundtrip_dict(self):
        """to_dict → from_dict should produce an identical Episode."""
        ep = Episode(
            content="test roundtrip",
            importance=0.8,
            emotion="happy",
            context={"nested": [1, 2]},
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)

        assert restored.id == ep.id
        assert restored.content == ep.content
        assert restored.importance == ep.importance
        assert restored.emotion == ep.emotion
        assert restored.context == ep.context
        assert restored.consolidated == ep.consolidated

    def test_to_dict_json_serializable(self):
        """to_dict output must be fully JSON-serializable."""
        ep = Episode(content="test", context={"x": [1, 2, 3]})
        d = ep.to_dict()
        serialized = json.dumps(d)  # should not raise
        assert isinstance(serialized, str)

    def test_unique_ids(self):
        """Each episode gets a unique ID."""
        ids = {Episode(content="x").id for _ in range(100)}
        assert len(ids) == 100


# ── Fact ────────────────────────────────────────────────────────────

class TestFact:

    def test_defaults(self):
        f = Fact(subject="user", predicate="likes", object="coffee")
        assert f.confidence == 0.5
        assert f.is_active is True
        assert f.source_episode_ids == []

    def test_roundtrip_dict(self):
        f = Fact(
            subject="user", predicate="lives_in", object="Delhi",
            confidence=0.85, source_episode_ids=["ep1", "ep2"],
        )
        restored = Fact.from_dict(f.to_dict())
        assert restored.subject == f.subject
        assert restored.object == f.object
        assert restored.confidence == f.confidence
        assert restored.source_episode_ids == ["ep1", "ep2"]
        assert restored.is_active is True

    def test_repr(self):
        f = Fact(subject="user", predicate="prefers", object="tea", confidence=0.9)
        assert "user" in repr(f)
        assert "90%" in repr(f)


# ── RetrievalResult ────────────────────────────────────────────────

class TestRetrievalResult:

    def test_creation(self):
        ep = Episode(content="test episode")
        r = RetrievalResult(
            episode=ep, score=0.75,
            recency=0.9, relevance=0.6, importance=0.5,
        )
        assert r.score == 0.75
        assert r.episode.content == "test episode"

    def test_repr(self):
        ep = Episode(content="a long content string for testing repr")
        r = RetrievalResult(episode=ep, score=0.5)
        assert "0.500" in repr(r)


# ── MemoryConfig ───────────────────────────────────────────────────

class TestMemoryConfig:

    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.w_recency == 0.3
        assert cfg.w_relevance == 0.4
        assert cfg.w_importance == 0.3
        assert cfg.decay_rate == 0.02
        assert len(cfg.forgetting_tiers) == 5

    def test_fast_preset(self):
        cfg = MemoryConfig.fast()
        assert cfg.decay_rate == 0.1
        assert cfg.max_context_tokens == 1000

    def test_deep_preset(self):
        cfg = MemoryConfig.deep()
        assert cfg.decay_rate == 0.005
        assert cfg.max_context_tokens == 4000

    def test_custom_override(self):
        cfg = MemoryConfig(decay_rate=0.5, w_recency=0.8)
        assert cfg.decay_rate == 0.5
        assert cfg.w_recency == 0.8
        # Other defaults preserved
        assert cfg.w_relevance == 0.4

    def test_weights_sum(self):
        """Default retrieval weights should sum to ~1.0."""
        cfg = MemoryConfig()
        total = cfg.w_recency + cfg.w_relevance + cfg.w_importance
        assert abs(total - 1.0) < 0.01
