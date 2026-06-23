"""Core data models. Lean, JSON-serializable, no external dependencies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class Episode:
    """A single memory event — what happened, when, and how important it was.

    This is the fundamental unit of episodic memory. Think of it as one
    entry in a diary: timestamped, scored for importance, optionally tagged
    with emotion and arbitrary context.

    Attributes:
        id: Globally unique identifier (UUID4). Auto-generated.
        content: The textual description of what happened.
        timestamp: When this episode occurred. Defaults to now.
        importance: 0.0 (trivial) to 1.0 (life-changing). Drives retention.
        emotion: Optional emotion tag (e.g., "happy", "stressed").
        source: Origin of this episode (e.g., "conversation", "observation").
        context: Arbitrary metadata dict. Stored as JSON in the DB.
        consolidated: Whether this episode has been compressed into facts.
        access_count: How many times this has been retrieved. Drives retention.
    """
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    emotion: Optional[str] = None
    source: str = "conversation"
    context: dict = field(default_factory=dict)
    consolidated: bool = False
    access_count: int = 0
    thread_id: str | None = None
    parent_id: str | None = None
    depth: int = 0
    archived: bool = False
    owner_id: str = "default"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        """Serialize to a flat dict suitable for SQLite insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "emotion": self.emotion,
            "source": self.source,
            "context": json.dumps(self.context),
            "consolidated": int(self.consolidated),
            "access_count": self.access_count,
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "archived": int(self.archived),
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        """Deserialize from a dict or sqlite3.Row."""
        if hasattr(d, "keys"):
            d = dict(d)
        return cls(
            id=d["id"],
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            importance=d["importance"],
            emotion=d["emotion"],
            source=d["source"],
            context=json.loads(d["context"]) if isinstance(d["context"], str) else d["context"],
            consolidated=bool(d["consolidated"]),
            access_count=d["access_count"],
            thread_id=d.get("thread_id"),
            parent_id=d.get("parent_id"),
            depth=d.get("depth", 0),
            archived=bool(d["archived"]) if d.get("archived") is not None else False,
            owner_id=d.get("owner_id", "default"),
        )


@dataclass(slots=True)
class Fact:
    """A semantic fact — a stable piece of knowledge extracted from episodes.

    Stored as (subject, predicate, object) triples with confidence tracking
    and provenance. Facts can be reinforced (same info repeated) or
    contradicted (new info replaces old).

    Attributes:
        id: Globally unique identifier.
        subject: Who/what this fact is about (e.g., "user").
        predicate: The relationship (e.g., "prefers", "lives_in").
        object: The value (e.g., "black coffee", "Mumbai").
        confidence: 0.0 to 1.0. Increases on reinforcement, resets on contradiction.
        source_episode_ids: Episode IDs that contributed to this fact (provenance).
        first_observed: When this fact was first learned.
        last_updated: When this fact was last reinforced or modified.
        is_active: False if superseded by a newer contradicting fact.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source_episode_ids: list[str] = field(default_factory=list)
    first_observed: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    recurrence: str | None = None
    owner_id: str = "default"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_episode_ids": json.dumps(self.source_episode_ids),
            "first_observed": self.first_observed.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "is_active": int(self.is_active),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "recurrence": self.recurrence,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fact:
        if hasattr(d, "keys"):
            d = dict(d)
        return cls(
            id=d["id"],
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            confidence=d["confidence"],
            source_episode_ids=(
                json.loads(d["source_episode_ids"])
                if isinstance(d["source_episode_ids"], str)
                else d["source_episode_ids"]
            ),
            first_observed=datetime.fromisoformat(d["first_observed"]),
            last_updated=datetime.fromisoformat(d["last_updated"]),
            is_active=bool(d["is_active"]),
            valid_from=datetime.fromisoformat(d["valid_from"]) if d.get("valid_from") else None,
            valid_until=datetime.fromisoformat(d["valid_until"]) if d.get("valid_until") else None,
            recurrence=d.get("recurrence"),
            owner_id=d.get("owner_id", "default"),
        )

    def __repr__(self) -> str:
        return f"Fact({self.subject} → {self.predicate} → {self.object}, conf={self.confidence:.0%})"


@dataclass(slots=True)
class Procedure:
    """A procedural rule: when trigger, do action."""

    trigger: str
    action: str
    confidence: float = 0.5
    owner_id: str = "default"
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "action": self.action,
            "confidence": self.confidence,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, d: dict) -> "Procedure":
        if hasattr(d, "keys"):
            d = dict(d)
        created_at = d.get("created_at")
        return cls(
            id=d["id"],
            trigger=d["trigger"],
            action=d["action"],
            confidence=d["confidence"],
            owner_id=d.get("owner_id", "default"),
            metadata=json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"],
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
        )


@dataclass(slots=True)
class RetrievalResult:
    """A scored memory retrieval result with component breakdown.

    Returned by recall(). The component scores (recency, relevance,
    importance) allow callers to understand _why_ a memory was ranked
    where it was — important for debugging and trust.
    """
    episode: Episode
    score: float              # Final combined score (0–1)
    recency: float = 0.0     # Exponential decay component
    relevance: float = 0.0   # Keyword/semantic similarity component
    importance: float = 0.0  # Stored importance component

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(score={self.score:.3f}, "
            f"R={self.recency:.2f}/V={self.relevance:.2f}/I={self.importance:.2f}, "
            f"{self.episode.content[:40]}...)"
        )
