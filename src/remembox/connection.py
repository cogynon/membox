"""SQLite connection helpers.

When using an in-memory database (``:memory:``), each call to
``sqlite3.connect(":memory:")`` produces a brand-new, isolated database.
This means the episodic, semantic, and embedding stores would lose
visibility into each other's tables, breaking cross-store queries and
any consistency guarantees.

This module provides helpers to share a single connection for in-memory
databases while keeping distinct connections for on-disk databases (so
WAL-mode concurrency is preserved).
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def create_connection(
    db_path: str,
    shared_memory_conn: Optional[sqlite3.Connection] = None,
) -> sqlite3.Connection:
    """Create or reuse a SQLite connection.

    Args:
        db_path: SQLite database path. If ``:memory:``, `shared_memory_conn`
            is returned when provided; otherwise a new in-memory connection
            is created.
        shared_memory_conn: A connection to share when db_path is ``:memory:``.

    Returns:
        A sqlite3.Connection configured with WAL mode, row factory, and
        reasonable defaults.
    """
    if db_path == ":memory:" and shared_memory_conn is not None:
        return shared_memory_conn

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply standard pragmas to a fresh connection.

    Safe to call repeatedly (WAL mode is idempotent once enabled).
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
