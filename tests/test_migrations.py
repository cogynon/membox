"""Tests for SQLite schema migrations."""

import sqlite3
import tempfile

import pytest

from membox import Membox


def _create_legacy_database(path: str) -> None:
    """Create a database with the pre-v2 schema (no owner_id columns)."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            emotion TEXT,
            source TEXT NOT NULL DEFAULT 'conversation',
            context TEXT NOT NULL DEFAULT '{}',
            consolidated INTEGER NOT NULL DEFAULT 0,
            access_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            source_episode_ids TEXT NOT NULL DEFAULT '[]',
            first_observed TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
    """)
    conn.execute(
        "INSERT INTO episodes (id, content, timestamp, importance) VALUES (?, ?, ?, ?)",
        ("ep1", "legacy episode", "2024-01-01T00:00:00", 0.7),
    )
    conn.commit()
    conn.close()


class TestMigrations:

    def test_legacy_database_migrates_and_remains_queryable(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        _create_legacy_database(path)

        # Opening Membox should run migrations transparently.
        memory = Membox(path)

        # Legacy data still accessible
        recent = memory.recent(5)
        assert len(recent) == 1
        assert recent[0].content == "legacy episode"

        # New operations work
        memory.record("new episode", importance=0.5)
        assert memory._episodic.count() == 2

        memory.close()

    def test_new_database_reports_zero_migrations(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        memory = Membox(path)
        stats = memory.stats()
        assert stats["episodes"]["total"] == 0
        memory.close()

    def test_migration_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        _create_legacy_database(path)

        # First open runs migration
        m1 = Membox(path)
        m1.close()

        # Second open should not crash and should not re-apply
        m2 = Membox(path)
        assert m2._episodic.count() == 1
        m2.close()
