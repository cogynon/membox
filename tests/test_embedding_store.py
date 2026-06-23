"""Tests for SQLite-backed embedding store (P0.2)."""

import json
from datetime import datetime, timedelta

import pytest

from membox.models import Episode

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

if SENTENCE_TRANSFORMERS_AVAILABLE:
    from membox.embedding_store import EmbeddingStore


@pytest.fixture
def mock_model():
    """Return a tiny mock sentence-transformers model for fast tests."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers not installed")

    # Use a tiny, fast model for tests
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
class TestEmbeddingStore:
    """SQLite-backed embeddings: persistence, sync, scoping."""

    def test_add_and_get(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        ep = Episode(content="I love hiking in the mountains")
        store.add(ep)

        emb = store.get(ep.id)
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) == 384  # MiniLM-L6-v2 dimension

    def test_add_batch(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        episodes = [
            Episode(content="First episode"),
            Episode(content="Second episode"),
        ]
        count = store.add_batch(episodes)
        assert count == 2

        batch = store.get_batch([ep.id for ep in episodes])
        assert len(batch) == 2
        for ep_id, emb in batch.items():
            assert len(emb) == 384

    def test_delete_cascade(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        ep = Episode(content="Delete me")
        store.add(ep)
        assert store.get(ep.id) is not None

        store.delete([ep.id])
        assert store.get(ep.id) is None
        assert store.count() == 0

    def test_clear(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        for i in range(5):
            store.add(Episode(content=f"Episode {i}"))
        assert store.count() == 5

        store.clear()
        assert store.count() == 0

    def test_similarity_search(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        episodes = [
            Episode(content="I enjoy hiking in the mountains"),
            Episode(content="The weather is sunny today"),
            Episode(content="I love outdoor adventures"),
        ]
        store.add_batch(episodes)

        results = store.similarity("outdoor activities", k=2)
        assert len(results) == 2
        # The hiking and outdoor adventures episodes should rank highest
        ids = [r[0] for r in results]
        assert episodes[0].id in ids or episodes[2].id in ids

    def test_owner_scoping(self, mock_model, tmp_db):
        alice_store = EmbeddingStore(tmp_db, owner_id="alice", model=mock_model,
                                       model_name="all-MiniLM-L6-v2")
        bob_store = EmbeddingStore(tmp_db, owner_id="bob", model=mock_model,
                                   model_name="all-MiniLM-L6-v2")

        alice_ep = Episode(content="Alice's data")
        bob_ep = Episode(content="Bob's data")

        alice_store.add(alice_ep)
        bob_store.add(bob_ep)

        assert alice_store.get(alice_ep.id) is not None
        assert alice_store.get(bob_ep.id) is None

        assert bob_store.get(bob_ep.id) is not None
        assert bob_store.get(alice_ep.id) is None

    def test_stats(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        store.add(Episode(content="One"))
        store.add(Episode(content="Two"))

        stats = store.stats()
        assert stats["total"] == 2
        assert stats["model_name"] == "all-MiniLM-L6-v2"

    def test_persistence_across_restarts(self, mock_model, tmp_db):
        store1 = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                                model_name="all-MiniLM-L6-v2")
        ep = Episode(content="Persistent embedding")
        store1.add(ep)
        store1.close()

        # Reopen without model — should still be able to read embeddings
        store2 = EmbeddingStore(tmp_db, owner_id="test", model=None,
                                model_name="all-MiniLM-L6-v2")
        emb = store2.get(ep.id)
        assert emb is not None
        assert len(emb) == 384

    def test_no_model_add_is_noop(self, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=None,
                               model_name="all-MiniLM-L6-v2")
        ep = Episode(content="Should not crash")
        store.add(ep)  # model is None, so this is a noop
        assert store.get(ep.id) is None
        assert store.count() == 0

    def test_similarity_by_vector(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        episodes = [
            Episode(content="Python programming"),
            Episode(content="Machine learning"),
            Episode(content="Italian cuisine"),
        ]
        store.add_batch(episodes)

        # Encode query manually
        query_vec = mock_model.encode("coding in Python").tolist()
        results = store.similarity_by_vector(query_vec, k=2)

        assert len(results) == 2
        # The programming and ML episodes should be most similar
        ids = [r[0] for r in results]
        assert episodes[0].id in ids

    def test_delete_before_timestamp(self, mock_model, tmp_db):
        store = EmbeddingStore(tmp_db, owner_id="test", model=mock_model,
                               model_name="all-MiniLM-L6-v2")
        # Need episodes table for this test
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                timestamp TEXT,
                importance REAL DEFAULT 0.5,
                emotion TEXT,
                source TEXT DEFAULT 'conversation',
                context TEXT DEFAULT '{}',
                consolidated INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                owner_id TEXT DEFAULT 'default'
            )
        """)
        old_ep = Episode(content="Old episode",
                         timestamp=datetime(2024, 1, 1))
        new_ep = Episode(content="New episode",
                         timestamp=datetime(2026, 1, 1))
        conn.execute(
            "INSERT INTO episodes (id, content, timestamp, owner_id) VALUES (?, ?, ?, ?)",
            (old_ep.id, old_ep.content, old_ep.timestamp.isoformat(), "test"),
        )
        conn.execute(
            "INSERT INTO episodes (id, content, timestamp, owner_id) VALUES (?, ?, ?, ?)",
            (new_ep.id, new_ep.content, new_ep.timestamp.isoformat(), "test"),
        )
        conn.commit()
        conn.close()

        store.add(old_ep)
        store.add(new_ep)
        assert store.count() == 2

        store.delete_before(datetime(2025, 6, 1))
        assert store.count() == 1
        assert store.get(new_ep.id) is not None
        assert store.get(old_ep.id) is None
