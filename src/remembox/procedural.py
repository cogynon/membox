"""Procedural memory: routines, skills, and rules.

Procedural memory stores "when X, do Y" patterns. Examples:
- "When user says 'goodnight', dim lights and set alarm"
- "When user mentions server outage, run diagnostics first"

This is the third pillar of the memory architecture taught in the
lessons, alongside episodic and semantic memory.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from remembox.connection import create_connection
from remembox.migrations import migrate
from remembox.models import Procedure

_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    id          TEXT    PRIMARY KEY,
    trigger     TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.5,
    owner_id    TEXT    NOT NULL DEFAULT 'default',
    created_at  TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_proc_trigger ON procedures(trigger);
"""


class ProceduralStore:
    """SQLite-backed procedural memory store.

    Stores trigger-action rules scoped by owner_id.
    """

    def __init__(self, db_path: str = ":memory:",
                 owner_id: str = "default",
                 connection: sqlite3.Connection | None = None) -> None:
        self._owner_id = owner_id
        self._owns_connection = connection is None
        self._conn = create_connection(db_path, shared_memory_conn=connection)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        migrate(self._conn)

    # ── Write ───────────────────────────────────────────────────────

    def record(self, trigger: str, action: str,
               confidence: float = 0.5,
               metadata: dict | None = None) -> Procedure:
        """Store a new procedure rule."""
        proc = Procedure(
            trigger=trigger,
            action=action,
            confidence=confidence,
            owner_id=self._owner_id,
            metadata=metadata or {},
        )
        data = proc.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO procedures
               (id, trigger, action, confidence, owner_id, created_at, metadata)
               VALUES (:id, :trigger, :action, :confidence, :owner_id, :created_at, :metadata)""",
            data,
        )
        self._conn.commit()
        return proc

    def put(self, proc: Procedure) -> None:
        """Insert or replace a Procedure object directly."""
        proc.owner_id = self._owner_id
        data = proc.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO procedures
               (id, trigger, action, confidence, owner_id, created_at, metadata)
               VALUES (:id, :trigger, :action, :confidence, :owner_id, :created_at, :metadata)""",
            data,
        )
        self._conn.commit()

    # ── Match ───────────────────────────────────────────────────────

    def match(self, text: str) -> list[Procedure]:
        """Return all stored procedures whose trigger appears in text.

        Matches are case-insensitive substring matches. Results are sorted
        by confidence descending.

        Note: this loads all of an owner's procedures and filters in Python
        (substring matching can't be trivially SQL-indexed). Fine at small
        scale; for very large procedure sets, switch to FTS5 or trigger
        tokenization.
        """
        rows = self._conn.execute(
            "SELECT * FROM procedures WHERE owner_id = ? ORDER BY confidence DESC",
            (self._owner_id,),
        ).fetchall()

        text_lower = text.lower()
        matched = []
        for row in rows:
            trigger = row["trigger"].lower()
            # Simple substring match. In future, this could use regex or embeddings.
            if trigger in text_lower:
                matched.append(Procedure.from_row(row))

        return matched

    def match_best(self, text: str) -> Procedure | None:
        """Return the highest-confidence matching procedure, or None."""
        matches = self.match(text)
        return matches[0] if matches else None

    # ── Read ────────────────────────────────────────────────────────

    def all(self) -> list[Procedure]:
        """Return all procedures for this owner, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM procedures WHERE owner_id = ? ORDER BY created_at DESC",
            (self._owner_id,),
        ).fetchall()
        return [Procedure.from_row(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM procedures WHERE owner_id = ?",
            (self._owner_id,),
        ).fetchone()[0]

    # ── Delete ──────────────────────────────────────────────────────

    def delete(self, procedure_id: str) -> bool:
        """Delete a procedure by ID. Returns True if found."""
        cursor = self._conn.execute(
            "DELETE FROM procedures WHERE owner_id = ? AND id = ?",
            (self._owner_id, procedure_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Delete all procedures for this owner. Returns count."""
        cursor = self._conn.execute(
            "DELETE FROM procedures WHERE owner_id = ?",
            (self._owner_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
