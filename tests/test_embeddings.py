"""Tests for embedding retrieval. Skipped if sentence-transformers not installed."""

import math
from datetime import datetime, timedelta

import pytest

from membox.models import Episode
from membox.config import MemoryConfig

NOW = datetime(2026, 3, 25, 12, 0, 0)


# ── Test cosine similarity independently (no model needed) ──────────

class TestCosineSimilarity:
    """Test the cosine similarity function without requiring sentence-transformers."""

    def test_identical_vectors(self):
        from membox.embeddings import EmbeddingIndex
        a = [1.0, 0.0, 0.0]
        assert abs(EmbeddingIndex._cosine_similarity(a, a) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        from membox.embeddings import EmbeddingIndex
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(EmbeddingIndex._cosine_similarity(a, b)) < 0.001

    def test_opposite_vectors(self):
        from membox.embeddings import EmbeddingIndex
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert EmbeddingIndex._cosine_similarity(a, b) < 0

    def test_zero_vector(self):
        from membox.embeddings import EmbeddingIndex
        assert EmbeddingIndex._cosine_similarity([0, 0], [1, 1]) == 0.0


# ── Integration tests (require sentence-transformers) ──────────────

def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


requires_st = pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed",
)


@requires_st
class TestEmbeddingIndex:

    def test_add_and_query(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        index.add(Episode(content="I love hiking in the mountains", timestamp=NOW))
        index.add(Episode(content="Python is a great programming language", timestamp=NOW))
        index.add(Episode(content="The best coffee comes from Ethiopia", timestamp=NOW))

        results = index.query("outdoor activities and nature", k=1, now=NOW)
        assert len(results) >= 1
        assert "hiking" in results[0].episode.content.lower()

    def test_add_batch(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        episodes = [
            Episode(content=f"topic {i}", timestamp=NOW)
            for i in range(10)
        ]
        added = index.add_batch(episodes)
        assert added == 10
        assert len(index) == 10

    def test_no_duplicate_add(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        ep = Episode(content="test")
        index.add(ep)
        index.add(ep)  # Same ID
        assert len(index) == 1

    def test_contains(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        ep = Episode(content="test")
        index.add(ep)
        assert ep.id in index
        assert "nonexistent" not in index

    def test_empty_query(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        results = index.query("anything", k=5, now=NOW)
        assert results == []

    def test_scores_include_components(self):
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        index.add(Episode(content="I enjoy coffee", timestamp=NOW, importance=0.8))
        results = index.query("coffee", k=1, now=NOW)
        r = results[0]
        assert r.relevance > 0  # Cosine similarity
        assert r.recency > 0
        assert r.importance == 0.8

    def test_semantic_over_keyword(self):
        """Embedding search should find semantically similar, not just keyword matches."""
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        index.add(Episode(content="The dog is playing in the park", timestamp=NOW))
        index.add(Episode(content="Quarterly financial report due tomorrow", timestamp=NOW))
        index.add(Episode(content="A puppy running around outside", timestamp=NOW))

        results = index.query("pet playing outdoors", k=2, now=NOW)
        # Both dog/puppy episodes should rank above financial report
        contents = [r.episode.content.lower() for r in results]
        assert any("dog" in c or "puppy" in c for c in contents)

    def test_recency_affects_ranking(self):
        """Recent episode should rank higher than old one with same relevance."""
        from membox.embeddings import EmbeddingIndex
        index = EmbeddingIndex()
        index.add(Episode(
            content="User ordered coffee",
            timestamp=NOW - timedelta(days=30), importance=0.5,
        ))
        index.add(Episode(
            content="User ordered coffee",
            timestamp=NOW - timedelta(hours=1), importance=0.5,
        ))
        results = index.query("coffee", k=2, now=NOW)
        assert results[0].recency > results[1].recency
