"""Lightweight SQLite schema migrations.

Prevents crashes when upgrading an existing database file after package
updates add new columns or tables. Each migration is a numbered SQL
script; the database tracks the current version in a pragma table.

Usage:
    from membox.migrations import migrate
    migrate(conn)

This is called automatically by each store's __init__ after creating
its schema, so both new and legacy databases end up with the same
structure.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

CURRENT_SCHEMA_VERSION = 4

MigrationsType = list[tuple[int, str, Callable[[sqlite3.Connection], None]]]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _ensure_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS _schema_version (
            version INTEGER PRIMARY KEY
        )"""
    )
    conn.commit()


def _get_version(conn: sqlite3.Connection) -> int:
    _ensure_version_table(conn)
    row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
    return row[0] if row and row[0] is not None else 0


def _set_version(conn: sqlite3.Connection, version: int) -> None:
    _ensure_version_table(conn)
    conn.execute("DELETE FROM _schema_version")
    conn.execute("INSERT INTO _schema_version (version) VALUES (?)", (version,))
    conn.commit()


def _add_owner_id(conn: sqlite3.Connection) -> None:
    """Migration 1: add owner_id columns and indexes to all existing tables."""
    if _table_exists(conn, "episodes") and not _column_exists(conn, "episodes", "owner_id"):
        conn.execute("ALTER TABLE episodes ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_owner ON episodes(owner_id)")

    if _table_exists(conn, "facts") and not _column_exists(conn, "facts", "owner_id"):
        conn.execute("ALTER TABLE facts ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_f_owner ON facts(owner_id)")

    if _table_exists(conn, "embeddings") and not _column_exists(conn, "embeddings", "owner_id"):
        conn.execute("ALTER TABLE embeddings ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_owner ON embeddings(owner_id)")

    if _table_exists(conn, "procedures") and not _column_exists(conn, "procedures", "owner_id"):
        conn.execute("ALTER TABLE procedures ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_owner ON procedures(owner_id)")

    conn.commit()


def _add_fact_temporal_fields(conn: sqlite3.Connection) -> None:
    """Migration 2: add temporal/recurrence columns to facts table."""
    if _table_exists(conn, "facts") and not _column_exists(conn, "facts", "valid_from"):
        conn.execute("ALTER TABLE facts ADD COLUMN valid_from TEXT")
    if _table_exists(conn, "facts") and not _column_exists(conn, "facts", "valid_until"):
        conn.execute("ALTER TABLE facts ADD COLUMN valid_until TEXT")
    if _table_exists(conn, "facts") and not _column_exists(conn, "facts", "recurrence"):
        conn.execute("ALTER TABLE facts ADD COLUMN recurrence TEXT")
    if _table_exists(conn, "facts"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_f_valid ON facts(valid_from, valid_until)")
    conn.commit()


def _add_episode_thread_columns(conn: sqlite3.Connection) -> None:
    """Migration 3: add thread_id, parent_id, depth to episodes."""
    if _table_exists(conn, "episodes") and not _column_exists(conn, "episodes", "thread_id"):
        conn.execute("ALTER TABLE episodes ADD COLUMN thread_id TEXT")
    if _table_exists(conn, "episodes") and not _column_exists(conn, "episodes", "parent_id"):
        conn.execute("ALTER TABLE episodes ADD COLUMN parent_id TEXT")
    if _table_exists(conn, "episodes") and not _column_exists(conn, "episodes", "depth"):
        conn.execute("ALTER TABLE episodes ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")
    if _table_exists(conn, "episodes"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_thread ON episodes(thread_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_parent ON episodes(parent_id)")
    conn.commit()


def _add_episode_archived_column(conn: sqlite3.Connection) -> None:
    """Migration 4: add a separate ``archived`` flag to episodes.

    Previously ``forget()``'s archive action reused the ``consolidated`` flag,
    which made archived episodes indistinguishable from knowledge-extracted
    ones and caused consolidate() to skip them. ``archived`` decouples the two.
    """
    if _table_exists(conn, "episodes") and not _column_exists(conn, "episodes", "archived"):
        conn.execute("ALTER TABLE episodes ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_archived ON episodes(archived)")
    # Composite index to speed up forgetting candidate scans (importance + age).
    if _table_exists(conn, "episodes"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_imp_ts ON episodes(importance, timestamp)")
    conn.commit()


MIGRATIONS: MigrationsType = [
    (1, "Add owner_id columns to all existing tables", _add_owner_id),
    (2, "Add temporal columns to facts table", _add_fact_temporal_fields),
    (3, "Add thread hierarchy columns to episodes", _add_episode_thread_columns),
    (4, "Add archived flag to episodes", _add_episode_archived_column),
]


def migrate(conn: sqlite3.Connection) -> dict:
    """Run any pending migrations on an open SQLite connection.

    Returns {"from": old_version, "to": new_version, "applied": [ids]}.
    """
    from_version = _get_version(conn)
    applied: list[int] = []

    if from_version >= CURRENT_SCHEMA_VERSION:
        return {"from": from_version, "to": from_version, "applied": applied}

    for version, description, fn in MIGRATIONS:
        if version <= from_version:
            continue
        fn(conn)
        applied.append(version)

    new_version = max(applied) if applied else from_version
    _set_version(conn, new_version)

    return {"from": from_version, "to": new_version, "applied": applied}
