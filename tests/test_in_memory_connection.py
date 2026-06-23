"""Tests for in-memory SQLite connection sharing across stores."""

import sqlite3

import pytest

from membox import Membox, MemoryConfig, Episode
from membox.connection import create_connection
from membox.episodic import EpisodicStore
from membox.semantic import SemanticStore


class TestInMemoryConnectionSharing:
    """Verify :memory: stores share one connection and see each other's tables."""

    def test_memory_stores_share_connection(self):
        """Episodic and semantic stores should use the same connection."""
        shared = sqlite3.connect(":memory:", check_same_thread=False)
        shared.row_factory = sqlite3.Row
        shared.execute("PRAGMA journal_mode=WAL")

        ep = EpisodicStore(":memory:", owner_id="test", connection=shared)
        sem = SemanticStore(":memory:", owner_id="test", connection=shared)

        # Both should answer to the same connection object
        assert ep._conn is shared
        assert sem._conn is shared

    def test_memory_membox_creates_single_connection(self):
        """Membox with :memory: should wire all stores to one DB."""
        memory = Membox(":memory:", owner_id="alice")
        memory.record("Alice's event", importance=0.7)
        memory.learn("user", "name", "Alice", confidence=0.9)

        # Both stores should see the data
        assert memory._episodic.count() == 1
        assert memory._semantic.count() == 1
        assert memory._episodic._conn is memory._semantic._conn

    def test_memory_deletion_cascade_no_missing_table(self):
        """EmbeddingStore cross-store delete_before must not crash in :memory:."""
        memory = Membox(":memory:")
        ep = memory.record("event to delete")
        # Manually trigger the cross-store join path used by delete_before
        # via EmbeddingStore directly (it joins with episodes table)
        from datetime import datetime, timedelta
        from membox.embedding_store import EmbeddingStore

        emb = EmbeddingStore(
            ":memory:",
            owner_id="default",
            model=None,
            model_name="test",
            connection=memory._episodic._conn,
        )
        # No embeddings for this episode since model is None, but tables exist
        assert emb.count() == 0
        # This executes: DELETE FROM embeddings WHERE episode_id IN (SELECT id FROM episodes ...)
        # It would crash if embeddings store couldn't see episodes table.
        deleted = emb.delete_before(datetime.now() + timedelta(days=1))
        assert deleted >= 0


class TestFileConnectionIsolations:
    """Verify on-disk databases keep separate connections (concurrency)."""

    def test_file_db_uses_distinct_connections(self, tmp_db):
        ep = EpisodicStore(tmp_db, owner_id="test")
        sem = SemanticStore(tmp_db, owner_id="test")
        assert ep._conn is not sem._conn


class TestConnectionHelper:
    """Unit tests for the create_connection helper."""

    def test_create_connection_reuses_shared_memory_conn(self):
        shared = sqlite3.connect(":memory:", check_same_thread=False)
        conn = create_connection(":memory:", shared_memory_conn=shared)
        assert conn is shared

    def test_create_connection_creates_new_for_file_path(self, tmp_db):
        conn = create_connection(tmp_db)
        assert conn is not None
        conn.close()
