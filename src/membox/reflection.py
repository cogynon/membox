"""Reflection: synthesize higher-order patterns across episodes.

Reflection is an optional second-stage consolidation that looks at groups
of recent episodes and extracts temporal patterns, recurring habits, and
relationship dynamics that single-episode consolidation cannot see.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from membox.connection import create_connection
from membox.migrations import migrate
from membox.models import Episode, Fact
from membox.tokens import tokenize_list

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id          TEXT    PRIMARY KEY,
    subject     TEXT    NOT NULL,
    predicate   TEXT    NOT NULL,
    object      TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.5,
    evidence    TEXT    NOT NULL DEFAULT '[]',  -- JSON list of episode IDs
    first_seen  TEXT    NOT NULL,
    last_seen   TEXT    NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    owner_id    TEXT    NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_ref_subj_pred ON reflections(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_ref_active    ON reflections(is_active);
"""


@dataclass(slots=True)
class Reflection:
    """A higher-order pattern learned across multiple episodes.

    Reflections are stored as (subject, predicate, object) triples but
    represent syntheses such as "user gets stressed before earnings" or
    "user prefers quiet places on weekends". The ``evidence`` field keeps
    the episode IDs that support the pattern.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    owner_id: str = "default"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "evidence": json.dumps(self.evidence),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "is_active": int(self.is_active),
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reflection":
        if hasattr(d, "keys"):
            d = dict(d)
        return cls(
            id=d["id"],
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            confidence=d["confidence"],
            evidence=json.loads(d["evidence"]) if isinstance(d["evidence"], str) else d["evidence"],
            first_seen=datetime.fromisoformat(d["first_seen"]),
            last_seen=datetime.fromisoformat(d["last_seen"]),
            is_active=bool(d["is_active"]),
            owner_id=d.get("owner_id", "default"),
        )


class ReflectionExtractor(ABC):
    """Strategy interface for turning a batch of episodes into reflections."""

    @abstractmethod
    def extract(self, episodes: list[Episode],
                now: datetime | None = None) -> list[Reflection]:
        """Return reflections inferred from the given episodes."""
        ...


class RuleBasedReflectionExtractor(ReflectionExtractor):
    """Dependency-free reflection extractor.

    Detects simple recurring patterns such as repeated mentions of the
    same emotion in a short window. LLM-based subclasses can do much
    more; this is the safe default.
    """

    def __init__(self, min_mentions: int = 3, lookback_days: int = 30) -> None:
        self.min_mentions = min_mentions
        self.lookback_days = lookback_days

    def extract(self, episodes: list[Episode],
                now: datetime | None = None) -> list[Reflection]:
        if not episodes:
            return []

        # Recent episodes only
        now = now or datetime.now()
        cutoff = now - timedelta(days=self.lookback_days)
        recent = [ep for ep in episodes if ep.timestamp >= cutoff]

        reflections: list[Reflection] = []

        # Pattern: recurring emotion -> "often feels X"
        emotion_counts: dict[str, list[str]] = {}
        for ep in recent:
            if ep.emotion:
                emotion_counts.setdefault(ep.emotion, []).append(ep.id)
        for emotion, ids in emotion_counts.items():
            if len(ids) >= self.min_mentions:
                reflections.append(Reflection(
                    subject="user",
                    predicate="often_feels",
                    object=emotion,
                    confidence=min(0.5 + 0.1 * len(ids), 0.95),
                    evidence=ids,
                ))

        # Pattern: repeated keyword phrase -> "frequently mentions X"
        # Count DISTINCT episodes per token (a set), not raw occurrences, so
        # "coffee coffee coffee" in one episode counts once. Uses the shared
        # tokenizer so "coffee" and "coffee." are the same token.
        keyword_hits: dict[str, set[str]] = {}
        for ep in recent:
            for token in set(tokenize_list(ep.content)):
                # ignore short/common words
                if len(token) <= 3:
                    continue
                keyword_hits.setdefault(token, set()).add(ep.id)
        for token, ids in keyword_hits.items():
            if len(ids) >= self.min_mentions:
                reflections.append(Reflection(
                    subject="user",
                    predicate="frequently_mentions",
                    object=token,
                    confidence=min(0.5 + 0.05 * len(ids), 0.9),
                    evidence=list(ids)[:10],  # cap evidence
                ))

        return reflections


class ReflectionStore:
    """SQLite-backed store for reflection patterns."""

    def __init__(self, db_path: str = ":memory:",
                 owner_id: str = "default",
                 connection: sqlite3.Connection | None = None) -> None:
        self._owner_id = owner_id
        self._owns_connection = connection is None
        self._conn = create_connection(db_path, shared_memory_conn=connection)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        migrate(self._conn)

    def record(self, reflection: Reflection) -> Reflection:
        """Insert or replace a reflection."""
        reflection.owner_id = self._owner_id
        self._conn.execute(
            """INSERT OR REPLACE INTO reflections
               (id, subject, predicate, object, confidence, evidence,
                first_seen, last_seen, is_active, owner_id)
               VALUES (:id, :subject, :predicate, :object, :confidence,
                       :evidence, :first_seen, :last_seen, :is_active, :owner_id)""",
            reflection.to_dict(),
        )
        self._conn.commit()
        return reflection

    def get(self, reflection_id: str) -> Reflection | None:
        row = self._conn.execute(
            "SELECT * FROM reflections WHERE owner_id = ? AND id = ?",
            (self._owner_id, reflection_id),
        ).fetchone()
        return Reflection.from_dict(row) if row else None

    def find(self, subject: str,
             predicate: str | None = None,
             active_only: bool = True) -> list[Reflection]:
        sql = "SELECT * FROM reflections WHERE owner_id = ? AND subject = ?"
        params: list = [self._owner_id, subject]
        if active_only:
            sql += " AND is_active = 1"
        if predicate:
            sql += " AND predicate = ?"
            params.append(predicate)
        sql += " ORDER BY confidence DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [Reflection.from_dict(r) for r in rows]

    def deactivate(self, reflection_id: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE reflections SET is_active = 0 WHERE owner_id = ? AND id = ?",
            (self._owner_id, reflection_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, reflection_ids: list[str]) -> int:
        if not reflection_ids:
            return 0
        placeholders = ",".join("?" for _ in reflection_ids)
        cursor = self._conn.execute(
            f"DELETE FROM reflections WHERE owner_id = ? AND id IN ({placeholders})",
            (self._owner_id, *reflection_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM reflections WHERE owner_id = ? AND is_active = 1",
            (self._owner_id,),
        ).fetchone()[0]

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reflect(episodes: list[Episode],
            store: ReflectionStore | None = None,
            extractor: ReflectionExtractor | None = None,
            now: datetime | None = None,
            ) -> list[Reflection]:
    """Run reflection over a set of episodes and persist the results.

    If a store is provided, reflections are learned idempotently:
    identical (subject, predicate, object) rows have their evidence and
    confidence updated instead of creating duplicates.
    """
    now = now or datetime.now()
    extractor = extractor or RuleBasedReflectionExtractor()
    reflections = extractor.extract(episodes, now=now)
    if store is None:
        return reflections

    persisted = []
    for r in reflections:
        # Merge with existing reflection if one exists for this (subject, predicate, object).
        existing_rows = store._conn.execute(
            "SELECT * FROM reflections WHERE owner_id = ? AND subject = ? AND predicate = ? AND object = ? AND is_active = 1",
            (store._owner_id, r.subject, r.predicate, r.object),
        ).fetchall()

        merged_evidence = list(dict.fromkeys(r.evidence))
        confidence = r.confidence
        first_seen = r.first_seen
        if existing_rows:
            existing = Reflection.from_dict(existing_rows[0])
            merged_evidence = list(dict.fromkeys(existing.evidence + r.evidence))
            confidence = min(max(existing.confidence, r.confidence) + 0.05, 0.99)
            first_seen = min(existing.first_seen, r.first_seen)
            # update in place
            r.id = existing.id

        r.evidence = merged_evidence
        r.confidence = confidence
        r.first_seen = first_seen
        r.last_seen = now
        store.record(r)
        persisted.append(r)

    return persisted
