"""SQLite-backed semantic fact store with conflict resolution.

Implements SemanticStoreProtocol. Supports reinforcement (repeated facts
boost confidence) and contradiction (new facts supersede old ones).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from remembox.config import MemoryConfig
from remembox.connection import create_connection
from remembox.migrations import migrate
from remembox.models import Fact
from remembox.tokens import escape_like

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
    is_active         INTEGER NOT NULL DEFAULT 1,
    valid_from        TEXT,
    valid_until       TEXT,
    recurrence        TEXT,
    owner_id          TEXT    NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_f_sp     ON facts(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_f_subj   ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_f_active ON facts(is_active);
CREATE INDEX IF NOT EXISTS idx_f_valid  ON facts(valid_from, valid_until);
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
                 config: MemoryConfig | None = None,
                 owner_id: str = "default",
                 connection: sqlite3.Connection | None = None) -> None:
        self._owner_id = owner_id
        self._owns_connection = connection is None
        self._conn = create_connection(db_path, shared_memory_conn=connection)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        migrate(self._conn)
        self._config = config or MemoryConfig()

    # ── Write ───────────────────────────────────────────────────────

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None,
              valid_from: datetime | None = None,
              valid_until: datetime | None = None,
              recurrence: str | None = None) -> tuple[Fact, str]:
        """Learn a fact. Returns (fact, action) where action is one of:
        'new', 'reinforced', 'contradicted'.

        Optional temporal fields:
            valid_from / valid_until: when the fact is known to be true.
            recurrence: e.g. "weekday_mornings", "quarterly".

        Contradiction is temporal-aware: facts with non-overlapping validity
        windows for the same (subject, predicate) can coexist as active facts.
        """
        # Look for an active fact with the same subject/predicate that overlaps
        # the new validity window.
        candidates = self._conn.execute(
            "SELECT * FROM facts WHERE owner_id=? AND subject=? AND predicate=? AND is_active=1",
            (self._owner_id, subject, predicate),
        ).fetchall()

        overlapping = None
        for row in candidates:
            other = Fact.from_dict(row)
            if self._windows_overlap(other.valid_from, other.valid_until,
                                     valid_from, valid_until):
                overlapping = other
                break

        if not overlapping:
            return self._insert_new(subject, predicate, obj, confidence,
                                    source_episode_id,
                                    valid_from=valid_from,
                                    valid_until=valid_until,
                                    recurrence=recurrence), "new"

        if overlapping.object == obj:
            return self._reinforce(overlapping, source_episode_id,
                                   valid_from=valid_from,
                                   valid_until=valid_until,
                                   recurrence=recurrence), "reinforced"
        else:
            return self._contradict(overlapping, subject, predicate, obj,
                                    confidence, source_episode_id,
                                    valid_from=valid_from,
                                    valid_until=valid_until,
                                    recurrence=recurrence), "contradicted"

    def put(self, fact: Fact) -> None:
        """Insert or replace a fact directly (bypass conflict resolution)."""
        fact.owner_id = self._owner_id
        self._conn.execute(
            """INSERT OR REPLACE INTO facts
               (id, subject, predicate, object, confidence,
                source_episode_ids, first_observed, last_updated, is_active,
                valid_from, valid_until, recurrence, owner_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact.id, fact.subject, fact.predicate, fact.object,
             fact.confidence, json.dumps(fact.source_episode_ids),
             fact.first_observed.isoformat(), fact.last_updated.isoformat(),
             int(fact.is_active),
             fact.valid_from.isoformat() if fact.valid_from else None,
             fact.valid_until.isoformat() if fact.valid_until else None,
             fact.recurrence, fact.owner_id),
        )
        self._conn.commit()

    # ── Read ────────────────────────────────────────────────────────

    def get(self, fact_id: str) -> Fact | None:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE owner_id = ? AND id = ?",
            (self._owner_id, fact_id),
        ).fetchone()
        return Fact.from_dict(row) if row else None

    def find(self, subject: str,
             predicate: str | None = None,
             at_time: datetime | None = None) -> list[Fact]:
        """Find active facts for a subject, optionally filtered by predicate.

        If ``at_time`` is provided, only return facts known to be true at that
        instant (valid_from <= at_time <= valid_until, or open-ended bounds).
        """
        sql = "SELECT * FROM facts WHERE owner_id=? AND subject=? AND is_active=1"
        params: list = [self._owner_id, subject]
        if predicate:
            sql += " AND predicate=?"
            params.append(predicate)
        if at_time is not None:
            ts = at_time.isoformat()
            sql += " AND (valid_from IS NULL OR valid_from <= ?) AND (valid_until IS NULL OR valid_until >= ?)"
            params.extend([ts, ts])
        sql += " ORDER BY confidence DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [Fact.from_dict(r) for r in rows]

    def about(self, subject: str, at_time: datetime | None = None) -> list[Fact]:
        """Alias for find(subject, at_time=at_time) — reads more naturally."""
        return self.find(subject, at_time=at_time)

    def search(self, keyword: str, limit: int = 10) -> list[Fact]:
        """Keyword search across all fact fields.

        LIKE wildcards in ``keyword`` (``%``/``_``) are escaped so they match
        literally rather than acting as wildcards.
        """
        like = f"%{escape_like(keyword)}%"
        rows = self._conn.execute(
            """SELECT * FROM facts
               WHERE owner_id=? AND is_active=1 AND (
                   subject LIKE ? ESCAPE '\\' OR predicate LIKE ? ESCAPE '\\'
                   OR object LIKE ? ESCAPE '\\'
               ) ORDER BY confidence DESC LIMIT ?""",
            (self._owner_id, like, like, like, limit),
        ).fetchall()
        return [Fact.from_dict(r) for r in rows]

    def all_active(self) -> list[Fact]:
        """Return all active facts."""
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE owner_id = ? AND is_active=1 ORDER BY confidence DESC",
            (self._owner_id,),
        ).fetchall()
        return [Fact.from_dict(r) for r in rows]

    # ── Delete / Deactivate ─────────────────────────────────────────

    def deactivate(self, fact_id: str) -> bool:
        """Mark a fact as inactive (soft delete). Returns True if found."""
        cursor = self._conn.execute(
            "UPDATE facts SET is_active=0 WHERE owner_id=? AND id=?",
            (self._owner_id, fact_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def edit_fact(self, fact_id: str,
                  obj: str | None = None,
                  predicate: str | None = None,
                  confidence: float | None = None,
                  source_episode_ids: list[str] | None = None,
                  is_active: bool | None = None) -> Fact | None:
        """Edit a fact in place. Returns the updated fact, or None if not found."""
        existing = self.get(fact_id)
        if existing is None:
            return None
        if obj is not None:
            existing.object = obj
        if predicate is not None:
            existing.predicate = predicate
        if confidence is not None:
            existing.confidence = confidence
        if source_episode_ids is not None:
            existing.source_episode_ids = source_episode_ids
        if is_active is not None:
            existing.is_active = is_active
        existing.last_updated = datetime.now()
        self.put(existing)
        return existing

    def correct_fact(self, fact_id: str,
                     new_object: str | None = None,
                     new_predicate: str | None = None,
                     new_confidence: float | None = None) -> tuple[Fact, str]:
        """Correct a fact: deactivate the old version and insert a corrected copy.

        Unlike ``edit_fact`` this keeps the old (now inactive) fact in the
        database as an audit trail. Provenance (source episode IDs) is copied
        to the corrected fact.
        """
        old = self.get(fact_id)
        if old is None:
            raise KeyError(fact_id)
        self.deactivate(fact_id)
        corrected = Fact(
            subject=old.subject,
            predicate=new_predicate if new_predicate is not None else old.predicate,
            object=new_object if new_object is not None else old.object,
            confidence=new_confidence if new_confidence is not None else old.confidence,
            source_episode_ids=old.source_episode_ids.copy(),
            first_observed=old.first_observed,
            last_updated=datetime.now(),
            is_active=True,
            owner_id=old.owner_id,
        )
        self.put(corrected)
        return corrected, "corrected"

    def delete(self, fact_ids: list[str]) -> int:
        """Hard-delete facts. Returns count deleted."""
        if not fact_ids:
            return 0
        placeholders = ",".join("?" for _ in fact_ids)
        cursor = self._conn.execute(
            f"DELETE FROM facts WHERE owner_id = ? AND id IN ({placeholders})",
            (self._owner_id, *fact_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE owner_id = ? AND is_active=1",
            (self._owner_id,),
        ).fetchone()[0]

    def stats(self) -> dict:
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN is_active=1 THEN 1 ELSE 0 END), 0) as active,
                COALESCE(SUM(CASE WHEN is_active=0 THEN 1 ELSE 0 END), 0) as superseded,
                AVG(CASE WHEN is_active=1 THEN confidence END) as avg_confidence
            FROM facts
            WHERE owner_id = ?
        """, (self._owner_id,)).fetchone()
        return {
            "total": row["total"],
            "active": row["active"],
            "superseded": row["superseded"],
            "avg_confidence": round(row["avg_confidence"] or 0, 3),
        }

    # ── Internal ────────────────────────────────────────────────────

    def _insert_new(self, subject: str, predicate: str, obj: str,
                    confidence: float,
                    source_episode_id: str | None,
                    valid_from: datetime | None = None,
                    valid_until: datetime | None = None,
                    recurrence: str | None = None) -> Fact:
        now = datetime.now()
        fact = Fact(
            subject=subject, predicate=predicate, object=obj,
            confidence=confidence,
            source_episode_ids=[source_episode_id] if source_episode_id else [],
            first_observed=now, last_updated=now,
            valid_from=valid_from,
            valid_until=valid_until,
            recurrence=recurrence,
        )
        self.put(fact)
        return fact

    def _reinforce(self, existing: Fact,
                   source_episode_id: str | None,
                   valid_from: datetime | None = None,
                   valid_until: datetime | None = None,
                   recurrence: str | None = None) -> Fact:
        boost = self._config.reinforce_boost_rate
        new_conf = min(1.0, existing.confidence + (1.0 - existing.confidence) * boost)
        eps = existing.source_episode_ids.copy()
        if source_episode_id:
            eps.append(source_episode_id)
        updates = {
            "confidence": new_conf,
            "source_episode_ids": json.dumps(eps),
            "last_updated": datetime.now().isoformat(),
        }
        if valid_from is not None:
            updates["valid_from"] = valid_from.isoformat()
        if valid_until is not None:
            updates["valid_until"] = valid_until.isoformat()
        if recurrence is not None:
            updates["recurrence"] = recurrence
        cols = ", ".join(f"{k}=?" for k in updates)
        self._conn.execute(
            f"UPDATE facts SET {cols} WHERE owner_id=? AND id=?",
            (*updates.values(), self._owner_id, existing.id),
        )
        self._conn.commit()
        existing.confidence = new_conf
        existing.source_episode_ids = eps
        if valid_from is not None:
            existing.valid_from = valid_from
        if valid_until is not None:
            existing.valid_until = valid_until
        if recurrence is not None:
            existing.recurrence = recurrence
        return existing

    def _contradict(self, existing: Fact,
                    subject: str, predicate: str, obj: str,
                    confidence: float,
                    source_episode_id: str | None,
                    valid_from: datetime | None = None,
                    valid_until: datetime | None = None,
                    recurrence: str | None = None) -> Fact:
        self._conn.execute(
            "UPDATE facts SET is_active=0, last_updated=? WHERE owner_id=? AND id=?",
            (datetime.now().isoformat(), self._owner_id, existing.id),
        )
        return self._insert_new(subject, predicate, obj, confidence, source_episode_id,
                                valid_from=valid_from,
                                valid_until=valid_until,
                                recurrence=recurrence)

    @staticmethod
    def _windows_overlap(a_from: datetime | None, a_until: datetime | None,
                         b_from: datetime | None, b_until: datetime | None) -> bool:
        """Return True if two validity windows overlap (or are open-ended)."""
        # Two unbounded/eternal facts always overlap.
        if a_from is None and a_until is None and b_from is None and b_until is None:
            return True
        # If one window has no end, overlap only if it starts before/at the other's end.
        a_start = a_from or datetime.min
        a_end = a_until or datetime.max
        b_start = b_from or datetime.min
        b_end = b_until or datetime.max
        return a_start <= b_end and b_start <= a_end

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
