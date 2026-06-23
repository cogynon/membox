"""SQLite-backed episodic memory store.

Thread-safe, WAL-mode, indexed for fast queries at 1M+ rows.
Implements EpisodicStoreProtocol from _store.py.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Iterator

from remembox.connection import create_connection
from remembox.migrations import migrate
from remembox.models import Episode
from remembox.tokens import escape_like

# ── Schema ──────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT    PRIMARY KEY,
    content     TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    importance  REAL    NOT NULL DEFAULT 0.5,
    emotion     TEXT,
    source      TEXT    NOT NULL DEFAULT 'conversation',
    context     TEXT    NOT NULL DEFAULT '{}',
    consolidated INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0,
    owner_id    TEXT    NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_ep_ts   ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ep_imp  ON episodes(importance DESC);
CREATE INDEX IF NOT EXISTS idx_ep_src  ON episodes(source);
CREATE INDEX IF NOT EXISTS idx_ep_cons ON episodes(consolidated);
"""


class EpisodicStore:
    """SQLite-backed persistent episodic memory.

    Designed for production:
    - WAL mode: concurrent reads during writes
    - Prepared statements: no SQL injection, faster queries
    - Indexes on timestamp, importance, source, consolidated
    - Batch insert: 10x faster for bulk operations
    - Thread-safe: check_same_thread=False

    Usage:
        store = EpisodicStore("agent.db")
        store.record(Episode(content="user said hello"))
        recent = store.recent(5)
    """

    def __init__(self, db_path: str = ":memory:",
                 owner_id: str = "default",
                 connection: sqlite3.Connection | None = None) -> None:
        self._owner_id = owner_id
        # If an external connection is passed, we don't own it and must not close it.
        self._owns_connection = connection is None
        self._conn = create_connection(db_path, shared_memory_conn=connection)
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        migrate(self._conn)

    # ── Write ───────────────────────────────────────────────────────

    def record(self, episode: Episode) -> None:
        """Persist a single episode."""
        episode.owner_id = self._owner_id
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, content, timestamp, importance, emotion, source,
                context, consolidated, access_count, owner_id,
                thread_id, parent_id, depth, archived)
               VALUES (:id, :content, :timestamp, :importance, :emotion,
                       :source, :context, :consolidated, :access_count, :owner_id,
                       :thread_id, :parent_id, :depth, :archived)""",
            episode.to_dict(),
        )
        self._conn.commit()

    def record_batch(self, episodes: list[Episode]) -> None:
        """Batch-insert episodes. Much faster than one-by-one.

        Uses a single transaction for atomicity and speed.
        Benchmarked: ~50K inserts/sec on typical hardware.
        """
        if not episodes:
            return
        for ep in episodes:
            ep.owner_id = self._owner_id
        self._conn.executemany(
            """INSERT OR REPLACE INTO episodes
               (id, content, timestamp, importance, emotion, source,
                context, consolidated, access_count, owner_id,
                thread_id, parent_id, depth, archived)
               VALUES (:id, :content, :timestamp, :importance, :emotion,
                       :source, :context, :consolidated, :access_count, :owner_id,
                       :thread_id, :parent_id, :depth, :archived)""",
            [ep.to_dict() for ep in episodes],
        )
        self._conn.commit()

    # ── Read ────────────────────────────────────────────────────────

    def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ? AND owner_id = ?",
            (episode_id, self._owner_id),
        ).fetchone()
        return Episode.from_dict(row) if row else None

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the N most recent episodes, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? ORDER BY timestamp DESC LIMIT ?",
            (self._owner_id, n),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def search(self, keyword: str, limit: int = 10) -> list[Episode]:
        """Keyword search across episode content. Case-insensitive.

        LIKE wildcards in ``keyword`` (``%``/``_``) are escaped so they match
        literally rather than acting as wildcards.
        """
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND content LIKE ? ESCAPE '\\' "
            "ORDER BY timestamp DESC LIMIT ?",
            (self._owner_id, f"%{escape_like(keyword)}%", limit),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def by_importance(self, min_importance: float = 0.7, limit: int = 10) -> list[Episode]:
        """Get episodes above an importance threshold."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND importance >= ? ORDER BY importance DESC LIMIT ?",
            (self._owner_id, min_importance, limit),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def unconsolidated(self, limit: int = 100) -> list[Episode]:
        """Get episodes that haven't been consolidated yet."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND consolidated = 0 ORDER BY timestamp ASC LIMIT ?",
            (self._owner_id, limit),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def by_thread(self, thread_id: str, limit: int = 1000) -> list[Episode]:
        """All episodes in a thread, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND thread_id = ? ORDER BY timestamp ASC LIMIT ?",
            (self._owner_id, thread_id, limit),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def by_parent(self, episode_id: str, limit: int = 1000) -> list[Episode]:
        """Direct child episodes of an episode, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND parent_id = ? ORDER BY timestamp ASC LIMIT ?",
            (self._owner_id, episode_id, limit),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    def threads(self, limit: int = 100) -> list[str]:
        """Distinct thread IDs for this owner, newest activity first."""
        rows = self._conn.execute(
            """SELECT thread_id FROM episodes
               WHERE owner_id = ? AND thread_id IS NOT NULL
               GROUP BY thread_id ORDER BY MAX(timestamp) DESC LIMIT ?""",
            (self._owner_id, limit),
        ).fetchall()
        return [r["thread_id"] for r in rows]

    def by_time_range(self, start: datetime, end: datetime) -> list[Episode]:
        """Get episodes within a time range (inclusive)."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE owner_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
            (self._owner_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [Episode.from_dict(r) for r in rows]

    # ── Update ──────────────────────────────────────────────────────

    def mark_consolidated(self, episode_ids: list[str]) -> int:
        """Mark episodes as consolidated. Returns number updated."""
        if not episode_ids:
            return 0
        placeholders = ",".join("?" for _ in episode_ids)
        cursor = self._conn.execute(
            f"UPDATE episodes SET consolidated = 1 WHERE owner_id = ? AND id IN ({placeholders})",
            (self._owner_id, *episode_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    def mark_archived(self, episode_ids: list[str]) -> int:
        """Mark episodes as archived (soft-delete). Returns number updated.

        Distinct from ``mark_consolidated``: archiving is a forgetting action,
        while consolidation records that knowledge was extracted. Keeping them
        separate means archived episodes are not mistaken for consolidated ones.
        """
        if not episode_ids:
            return 0
        placeholders = ",".join("?" for _ in episode_ids)
        cursor = self._conn.execute(
            f"UPDATE episodes SET archived = 1 WHERE owner_id = ? AND id IN ({placeholders})",
            (self._owner_id, *episode_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    def increment_access(self, episode_id: str) -> None:
        """Bump access_count for a retrieved episode."""
        self._conn.execute(
            "UPDATE episodes SET access_count = access_count + 1 WHERE owner_id = ? AND id = ?",
            (self._owner_id, episode_id),
        )
        self._conn.commit()

    def update(self, episode: Episode) -> None:
        """Update an existing episode in place (INSERT OR REPLACE by ID)."""
        episode.owner_id = self._owner_id
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, content, timestamp, importance, emotion, source,
                context, consolidated, access_count, owner_id,
                thread_id, parent_id, depth, archived)
               VALUES (:id, :content, :timestamp, :importance, :emotion,
                       :source, :context, :consolidated, :access_count, :owner_id,
                       :thread_id, :parent_id, :depth, :archived)""",
            episode.to_dict(),
        )
        self._conn.commit()

    def annotate(self, episode_id: str, annotation: dict) -> Episode | None:
        """Append a structured annotation to an episode's context.

        The annotation is stored under ``context['__annotations__']`` and
        preserves a timestamped audit trail of corrections or flags.
        """
        existing = self.get(episode_id)
        if existing is None:
            return None
        annotations = existing.context.setdefault("__annotations__", [])
        if not isinstance(annotations, list):
            annotations = []
        annotations.append(annotation)
        existing.context["__annotations__"] = annotations
        self.update(existing)
        return existing

    # ── Delete ──────────────────────────────────────────────────────

    def delete(self, episode_ids: list[str]) -> int:
        """Delete episodes by ID. Returns number deleted."""
        if not episode_ids:
            return 0
        placeholders = ",".join("?" for _ in episode_ids)
        cursor = self._conn.execute(
            f"DELETE FROM episodes WHERE owner_id = ? AND id IN ({placeholders})",
            (self._owner_id, *episode_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_before(self, cutoff: datetime) -> int:
        """Delete all episodes before a timestamp. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM episodes WHERE owner_id = ? AND timestamp < ?",
            (self._owner_id, cutoff.isoformat()),
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of episodes."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE owner_id = ?", (self._owner_id,)
        ).fetchone()[0]

    def stats(self) -> dict:
        """Aggregate statistics about the episodic store."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(importance) as avg_importance,
                COALESCE(SUM(CASE WHEN consolidated = 1 THEN 1 ELSE 0 END), 0) as consolidated,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM episodes
            WHERE owner_id = ?
        """, (self._owner_id,)).fetchone()
        return {
            "total": row["total"],
            "avg_importance": round(row["avg_importance"] or 0, 3),
            "consolidated": row["consolidated"],
            "unconsolidated": row["total"] - row["consolidated"],
            "earliest": row["earliest"],
            "latest": row["latest"],
        }

    # ── Iteration ───────────────────────────────────────────────────

    def iter_all(self, batch_size: int = 500) -> Iterator[Episode]:
        """Memory-efficient iteration over all episodes.

        Uses keyset pagination on ``(timestamp, id)`` rather than LIMIT/OFFSET.
        This is O(N) overall instead of O(N²), and is stable under concurrent
        deletes (rows are neither skipped nor duplicated when earlier rows
        disappear mid-iteration).
        """
        last_ts: str | None = None
        last_id: str | None = None
        while True:
            if last_ts is None:
                rows = self._conn.execute(
                    "SELECT * FROM episodes WHERE owner_id = ? "
                    "ORDER BY timestamp ASC, id ASC LIMIT ?",
                    (self._owner_id, batch_size),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM episodes WHERE owner_id = ? "
                    "AND (timestamp > ? OR (timestamp = ? AND id > ?)) "
                    "ORDER BY timestamp ASC, id ASC LIMIT ?",
                    (self._owner_id, last_ts, last_ts, last_id, batch_size),
                ).fetchall()
            if not rows:
                break
            for row in rows:
                yield Episode.from_dict(row)
            last_ts = rows[-1]["timestamp"]
            last_id = rows[-1]["id"]
            if len(rows) < batch_size:
                break

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
