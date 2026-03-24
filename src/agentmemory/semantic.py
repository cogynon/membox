"""SQLite-backed semantic fact store with conflict resolution.

Implements SemanticStoreProtocol. Supports reinforcement (repeated facts
boost confidence) and contradiction (new facts supersede old ones).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from agentmemory.config import MemoryConfig
from agentmemory.models import Fact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id                TEXT PRIMARY KEY,
    subject           TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    object            TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 0.5,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    first_observed    TEXT NOT NULL,
    last_updated      TEXT NOT NULL,
    is_active         INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_f_sp     ON facts(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_f_subj   ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_f_active ON facts(is_active);
"""


class SemanticStore:
    """SQLite-backed semantic memory with conflict resolution.

    Facts are (subject, predicate, object) triples. When a fact is learned:

    - **New**: No existing fact → insert with given confidence.
    - **Reinforce**: Same (subject, predicate, object) → boost confidence.
    - **Contradict**: Same (subject, predicate) but different object →
      deactivate old fact, insert new one.

    Usage:
        store = SemanticStore("agent.db")
        fact, action = store.learn("user", "prefers", "coffee", confidence=0.8)
        facts = store.about("user")
    """

    def __init__(self, db_path: str = ":memory:",
                 config: MemoryConfig | None = None) -> None:
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._config = config or MemoryConfig()

    # ── Write ───────────────────────────────────────────────────────

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None) -> tuple[Fact, str]:
        """Learn a fact. Returns (fact, action) where action is one of:
        'new', 'reinforced', 'contradicted'.
        """
        existing = self._conn.execute(
            "SELECT * FROM facts WHERE subject=? AND predicate=? AND is_active=1",
            (subject, predicate),
        ).fetchone()

        if not existing:
            return self._insert_new(subject, predicate, obj, confidence,
                                    source_episode_id), "new"

        existing_fact = Fact.from_dict(existing)

        if existing_fact.object == obj:
            return self._reinforce(existing_fact, source_episode_id), "reinforced"
        else:
            return self._contradict(existing_fact, subject, predicate, obj,
                                    confidence, source_episode_id), "contradicted"

    def put(self, fact: Fact) -> None:
        """Insert or replace a fact directly (bypass conflict resolution)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO facts
               (id, subject, predicate, object, confidence,
                source_episode_ids, first_observed, last_updated, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact.id, fact.subject, fact.predicate, fact.object,
             fact.confidence, json.dumps(fact.source_episode_ids),
             fact.first_observed.isoformat(), fact.last_updated.isoformat(),
             int(fact.is_active)),
        )
        self._conn.commit()

    # ── Read ────────────────────────────────────────────────────────

    def get(self, fact_id: str) -> Fact | None:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return Fact.from_dict(row) if row else None

    def find(self, subject: str,
             predicate: str | None = None) -> list[Fact]:
        """Find active facts for a subject, optionally filtered by predicate."""
        if predicate:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE subject=? AND predicate=? AND is_active=1 "
                "ORDER BY confidence DESC",
                (subject, predicate),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE subject=? AND is_active=1 "
                "ORDER BY confidence DESC",
                (subject,),
            ).fetchall()
        return [Fact.from_dict(r) for r in rows]

    def about(self, subject: str) -> list[Fact]:
        """Alias for find(subject) — reads more naturally."""
        return self.find(subject)

    def search(self, keyword: str, limit: int = 10) -> list[Fact]:
        """Keyword search across all fact fields."""
        rows = self._conn.execute(
            """SELECT * FROM facts
               WHERE is_active=1 AND (
                   subject LIKE ? OR predicate LIKE ? OR object LIKE ?
               ) ORDER BY confidence DESC LIMIT ?""",
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
        return [Fact.from_dict(r) for r in rows]

    def all_active(self) -> list[Fact]:
        """Return all active facts."""
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE is_active=1 ORDER BY confidence DESC"
        ).fetchall()
        return [Fact.from_dict(r) for r in rows]

    # ── Delete / Deactivate ─────────────────────────────────────────

    def deactivate(self, fact_id: str) -> bool:
        """Mark a fact as inactive (soft delete). Returns True if found."""
        cursor = self._conn.execute(
            "UPDATE facts SET is_active=0 WHERE id=?", (fact_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, fact_ids: list[str]) -> int:
        """Hard-delete facts. Returns count deleted."""
        if not fact_ids:
            return 0
        placeholders = ",".join("?" for _ in fact_ids)
        cursor = self._conn.execute(
            f"DELETE FROM facts WHERE id IN ({placeholders})", fact_ids
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE is_active=1"
        ).fetchone()[0]

    def stats(self) -> dict:
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN is_active=1 THEN 1 ELSE 0 END), 0) as active,
                COALESCE(SUM(CASE WHEN is_active=0 THEN 1 ELSE 0 END), 0) as superseded,
                AVG(CASE WHEN is_active=1 THEN confidence END) as avg_confidence
            FROM facts
        """).fetchone()
        return {
            "total": row["total"],
            "active": row["active"],
            "superseded": row["superseded"],
            "avg_confidence": round(row["avg_confidence"] or 0, 3),
        }

    # ── Internal ────────────────────────────────────────────────────

    def _insert_new(self, subject: str, predicate: str, obj: str,
                    confidence: float,
                    source_episode_id: str | None) -> Fact:
        now = datetime.now()
        fact = Fact(
            subject=subject, predicate=predicate, object=obj,
            confidence=confidence,
            source_episode_ids=[source_episode_id] if source_episode_id else [],
            first_observed=now, last_updated=now,
        )
        self.put(fact)
        return fact

    def _reinforce(self, existing: Fact,
                   source_episode_id: str | None) -> Fact:
        boost = self._config.reinforce_boost_rate
        new_conf = min(1.0, existing.confidence + (1.0 - existing.confidence) * boost)
        eps = existing.source_episode_ids.copy()
        if source_episode_id:
            eps.append(source_episode_id)
        self._conn.execute(
            "UPDATE facts SET confidence=?, source_episode_ids=?, last_updated=? WHERE id=?",
            (new_conf, json.dumps(eps), datetime.now().isoformat(), existing.id),
        )
        self._conn.commit()
        existing.confidence = new_conf
        existing.source_episode_ids = eps
        return existing

    def _contradict(self, existing: Fact,
                    subject: str, predicate: str, obj: str,
                    confidence: float,
                    source_episode_id: str | None) -> Fact:
        self._conn.execute(
            "UPDATE facts SET is_active=0, last_updated=? WHERE id=?",
            (datetime.now().isoformat(), existing.id),
        )
        return self._insert_new(subject, predicate, obj, confidence, source_episode_id)

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
