"""SQLite-backed embedding persistence for guaranteed sync with episodes.

Every embedding is stored in the same SQLite database as episodes,
scoped by owner_id, and linked to episode_id via a foreign-key-like
relationship. When episodes are deleted, their embeddings are deleted
too — no orphaned vectors, no sync drift across restarts.

This replaces the in-memory-only EmbeddingIndex with durable,
scopable, and synchronised storage.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from typing import Optional

from remembox.connection import create_connection
from remembox.migrations import migrate
from remembox.models import Episode

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    episode_id  TEXT    PRIMARY KEY,
    owner_id    TEXT    NOT NULL,
    embedding   TEXT    NOT NULL,   -- JSON list of floats
    model_name  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _encode_with_model(model, texts: list[str]) -> list[list[float]]:
    """Encode texts using a sentence-transformers model."""
    embeddings = model.encode(texts)
    # Handle both numpy arrays and lists
    return [
        emb.tolist() if hasattr(emb, "tolist") else list(emb)
        for emb in embeddings
    ]


class EmbeddingStore:
    """SQLite-backed persistent embedding store.

    Guarantees that embeddings stay in sync with episodes:
    - record()  → compute + store embedding atomically (via Remembox)
    - delete()  → remove embedding when episode is removed
    - All operations scoped by owner_id for multi-tenant safety.

    Usage (internal — managed by Remembox):
        store = EmbeddingStore("agent.db", owner_id="user1", model=model)
        store.add(episode)              # stores embedding
        sims = store.similarity(query_emb, k=5)  # semantic search
        store.delete([episode.id])      # clean up on forget()
    """

    def __init__(self, db_path: str = ":memory:",
                 owner_id: str = "default",
                 model: Optional[object] = None,
                 model_name: str = "unknown",
                 connection: sqlite3.Connection | None = None) -> None:
        self._owner_id = owner_id
        self._owns_connection = connection is None
        self._model = model
        self._model_name = model_name
        self._conn = create_connection(db_path, shared_memory_conn=connection)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        migrate(self._conn)

    # ── Write ───────────────────────────────────────────────────────

    def add(self, episode: Episode) -> None:
        """Compute and store embedding for an episode."""
        if self._model is None:
            return
        embedding = _encode_with_model(self._model, [episode.content])[0]
        self._conn.execute(
            """INSERT OR REPLACE INTO embeddings
               (episode_id, owner_id, embedding, model_name, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (episode.id, self._owner_id, json.dumps(embedding),
             self._model_name, datetime.now().isoformat()),
        )
        self._conn.commit()

    def add_batch(self, episodes: list[Episode]) -> int:
        """Batch-add embeddings. Returns count stored."""
        if self._model is None or not episodes:
            return 0
        contents = [ep.content for ep in episodes]
        embeddings = _encode_with_model(self._model, contents)
        rows = []
        for ep, emb in zip(episodes, embeddings):
            rows.append((
                ep.id, self._owner_id, json.dumps(emb),
                self._model_name, datetime.now().isoformat(),
            ))
        self._conn.executemany(
            """INSERT OR REPLACE INTO embeddings
               (episode_id, owner_id, embedding, model_name, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ── Delete (sync guarantee) ─────────────────────────────────────

    def delete(self, episode_ids: list[str]) -> int:
        """Delete embeddings for given episode IDs."""
        if not episode_ids:
            return 0
        placeholders = ",".join("?" for _ in episode_ids)
        cursor = self._conn.execute(
            f"DELETE FROM embeddings WHERE owner_id = ? AND episode_id IN ({placeholders})",
            (self._owner_id, *episode_ids),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_before(self, cutoff: datetime) -> int:
        """Delete embeddings for episodes older than cutoff."""
        # Join with episodes table to get timestamps
        cursor = self._conn.execute(
            """DELETE FROM embeddings
               WHERE owner_id = ? AND episode_id IN (
                   SELECT id FROM episodes
                   WHERE owner_id = ? AND timestamp < ?
               )""",
            (self._owner_id, self._owner_id, cutoff.isoformat()),
        )
        self._conn.commit()
        return cursor.rowcount

    def clear(self) -> int:
        """Delete all embeddings for this owner. Returns count."""
        cursor = self._conn.execute(
            "DELETE FROM embeddings WHERE owner_id = ?", (self._owner_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Read ────────────────────────────────────────────────────────

    def get(self, episode_id: str) -> list[float] | None:
        """Retrieve embedding vector for an episode."""
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE owner_id = ? AND episode_id = ?",
            (self._owner_id, episode_id),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["embedding"])

    def get_batch(self, episode_ids: list[str]) -> dict[str, list[float]]:
        """Retrieve embeddings for multiple episodes."""
        if not episode_ids:
            return {}
        placeholders = ",".join("?" for _ in episode_ids)
        rows = self._conn.execute(
            f"SELECT episode_id, embedding FROM embeddings WHERE owner_id = ? AND episode_id IN ({placeholders})",
            (self._owner_id, *episode_ids),
        ).fetchall()
        return {r["episode_id"]: json.loads(r["embedding"]) for r in rows}

    def get_all_for_owner(self) -> dict[str, list[float]]:
        """Retrieve all embeddings for this owner."""
        rows = self._conn.execute(
            "SELECT episode_id, embedding FROM embeddings WHERE owner_id = ?",
            (self._owner_id,),
        ).fetchall()
        return {r["episode_id"]: json.loads(r["embedding"]) for r in rows}

    # ── Similarity search ───────────────────────────────────────────

    def similarity(self, query_text: str, k: int = 5) -> list[tuple[str, float]]:
        """Find the k most similar episodes to a query text.

        Returns list of (episode_id, cosine_similarity), sorted descending.
        """
        if self._model is None:
            return []

        query_emb = _encode_with_model(self._model, [query_text])[0]
        return self.similarity_by_vector(query_emb, k=k)

    def similarity_by_vector(self, query_vec: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Find the k most similar episodes given an embedding vector.

        Brute-force cosine over all of an owner's embeddings: O(N×d) in Python
        per query, with no vector index. Fine below ~10k rows; for 100k+ use an
        approximate-nearest-neighbour index (FAISS/Annoy) instead.
        """
        rows = self._conn.execute(
            "SELECT episode_id, embedding FROM embeddings WHERE owner_id = ?",
            (self._owner_id,),
        ).fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = _cosine_similarity(query_vec, emb)
            scored.append((row["episode_id"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── Stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE owner_id = ?",
            (self._owner_id,),
        ).fetchone()[0]

    def stats(self) -> dict:
        row = self._conn.execute(
            """SELECT COUNT(*) as total, MIN(model_name) as model_name
               FROM embeddings WHERE owner_id = ?""",
            (self._owner_id,),
        ).fetchone()
        return {
            "total": row["total"] if row else 0,
            # Prefer the configured model name; fall back to whatever is stored.
            "model_name": self._model_name or (row["model_name"] if row else None),
        }

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
