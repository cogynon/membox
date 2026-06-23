"""Thread summarization: compress long conversation threads in place.

Inspired by pi's compaction (https://github.com/earendil-works/pi-mono): when a
conversation thread grows past a token budget, summarize the older episodes into
a single structured summary episode while preserving the most recent ones intact.

Where consolidation extracts *durable facts* and reflection extracts *patterns*,
summarization preserves the *narrative* of a thread so it can keep going without
replaying every message. The summary is stored as a new episode (source
``"thread_summary"``) and the summarized episodes are marked consolidated so they
fall out of the active working set without being lost.

This mirrors pi's two key ideas:
1. A token-budgeted cut point that keeps recent context verbatim.
2. A structured summary format (Goal / Decisions / Next Steps / Critical Context).

The core ships a dependency-free ``RuleBasedSummarizer``. Subclass ``Summarizer``
to plug in an LLM, exactly like ``Consolidator`` and ``ReflectionExtractor``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from membox._store import EpisodicStoreProtocol
from membox.config import MemoryConfig
from membox.models import Episode


def estimate_tokens(text: str) -> int:
    """~4 chars per token for English. Good enough for budgeting.

    Shared with the context builder so summarization and prompt assembly
    use the same heuristic.
    """
    return max(1, len(text) // 4)


def serialize_episodes(episodes: list[Episode], max_chars: int = 2000) -> str:
    """Render episodes to plain text for summarization input.

    Mirrors pi's ``serializeConversation``: each episode becomes a labeled
    line so the model treats it as material to summarize, not a conversation
    to continue. Long episode bodies are truncated to ``max_chars`` with a
    marker, since a few huge episodes otherwise dominate the token budget.
    """
    lines = []
    for ep in episodes:
        content = ep.content
        if len(content) > max_chars:
            dropped = len(content) - max_chars
            content = content[:max_chars] + f"\n…[{dropped} chars truncated]"
        ts = ep.timestamp.isoformat(timespec="minutes")
        tag = f" [{ep.emotion}]" if ep.emotion else ""
        lines.append(f"[{ts}{tag}] ({ep.source}) {content}")
    return "\n".join(lines)


@dataclass(slots=True)
class ThreadSummaryResult:
    """Outcome of a summarization pass over one thread."""
    thread_id: str
    summary_episode: Episode | None
    summarized_ids: list[str]
    kept_ids: list[str]
    tokens_before: int
    tokens_after: int

    @property
    def did_summarize(self) -> bool:
        return self.summary_episode is not None


class Summarizer(ABC):
    """Strategy interface for turning a span of episodes into a summary string.

    Implement ``summarize`` to support different strategies:
    - ``RuleBasedSummarizer``: deterministic, no deps (ships with core)
    - LLM-based: subclass and call your preferred model
    """

    @abstractmethod
    def summarize(self, episodes: list[Episode],
                  previous_summary: str | None = None,
                  custom_instructions: str | None = None) -> str:
        """Return a summary covering ``episodes``.

        Args:
            episodes: The span to summarize, oldest first.
            previous_summary: Prior summary text for this thread, if any, so
                summaries compound iteratively (as in pi's repeated compaction).
            custom_instructions: Optional focus for the summary.
        """
        ...


class RuleBasedSummarizer(Summarizer):
    """Dependency-free summarizer producing pi's structured format.

    Deterministic and offline: ranks episodes by importance/recency to build
    a readable summary. Good enough for demos and as a safe default; swap in an
    LLM-backed subclass for production-quality prose.
    """

    def summarize(self, episodes: list[Episode],
                  previous_summary: str | None = None,
                  custom_instructions: str | None = None) -> str:
        if not episodes:
            return previous_summary or ""

        ranked = sorted(
            episodes,
            key=lambda e: (e.importance, e.timestamp),
            reverse=True,
        )
        key_points = ranked[: min(8, len(ranked))]

        def snippet(text: str, limit: int = 160) -> str:
            text = " ".join(text.split())  # collapse whitespace
            return text if len(text) <= limit else text[:limit].rstrip() + "…"

        emotions = sorted({e.emotion for e in episodes if e.emotion})
        first_ts = min(e.timestamp for e in episodes)
        last_ts = max(e.timestamp for e in episodes)

        out: list[str] = []
        if custom_instructions:
            out.append(f"## Focus\n{custom_instructions}")

        if previous_summary:
            # Compound on top of the earlier summary (pi passes the previous
            # summary as iterative context on repeated compactions).
            out.append("## Earlier Summary\n" + previous_summary.strip())

        out.append(
            "## Goal\n"
            f"Continue the thread spanning {first_ts.date()} → {last_ts.date()} "
            f"({len(episodes)} episodes summarized)."
        )

        progress = "\n".join(f"- {snippet(e.content)}" for e in key_points)
        out.append("## Progress\n" + progress)

        if emotions:
            out.append("## Affective Context\n- Recurring emotions: " + ", ".join(emotions))

        # The most recent summarized episode is the best handoff anchor.
        latest = max(episodes, key=lambda e: e.timestamp)
        out.append("## Critical Context\n- Last point before recent window: " + snippet(latest.content))

        return "\n\n".join(out)


def _find_cut_index(episodes: list[Episode], keep_recent_tokens: int,
                    max_chars: int) -> int:
    """Walk backward accumulating tokens until ``keep_recent_tokens`` is hit.

    Returns the index of the first episode to KEEP (everything before it is
    summarized). Mirrors pi's cut-point search. Always keeps at least the most
    recent episode; never summarizes the entire thread down to nothing.
    """
    accumulated = 0
    # Default: keep only the last episode if the budget is tiny.
    cut = len(episodes) - 1
    for i in range(len(episodes) - 1, -1, -1):
        body = episodes[i].content[:max_chars]
        accumulated += estimate_tokens(body)
        if accumulated > keep_recent_tokens:
            cut = i + 1
            break
        cut = i
    # Always keep at least the newest episode, even if it alone blows the budget.
    cut = min(cut, len(episodes) - 1)
    # And always summarize at least the oldest when there's more than one.
    if cut <= 0 and len(episodes) > 1:
        cut = 1
    return cut


def summarize_thread(episodic: EpisodicStoreProtocol,
                     thread_id: str,
                     summarizer: Summarizer | None = None,
                     config: MemoryConfig | None = None,
                     owner_id: str = "default",
                     keep_recent_tokens: int | None = None,
                     custom_instructions: str | None = None,
                     now: datetime | None = None) -> ThreadSummaryResult:
    """Summarize the older portion of a thread, keeping recent episodes intact.

    Steps (mirroring pi's compaction):
    1. Load the thread oldest-first.
    2. Find a token-budgeted cut point; keep recent episodes verbatim.
    3. Summarize everything before the cut (compounding any prior summary).
    4. Store the summary as a new episode (``source="thread_summary"``).
    5. Mark the summarized episodes consolidated so they leave the active set.

    Returns a ``ThreadSummaryResult`` describing what happened. A no-op (nothing
    to summarize) returns ``did_summarize == False``.
    """
    config = config or MemoryConfig()
    summarizer = summarizer or RuleBasedSummarizer()
    now = now or datetime.now()
    keep = keep_recent_tokens if keep_recent_tokens is not None else config.summary_keep_recent_tokens
    max_chars = config.max_serialized_chars

    episodes = episodic.by_thread(thread_id)  # oldest first
    tokens_before = sum(estimate_tokens(e.content) for e in episodes)

    no_op = ThreadSummaryResult(
        thread_id=thread_id,
        summary_episode=None,
        summarized_ids=[],
        kept_ids=[e.id for e in episodes],
        tokens_before=tokens_before,
        tokens_after=tokens_before,
    )

    if len(episodes) < 2:
        return no_op

    cut = _find_cut_index(episodes, keep, max_chars)
    to_summarize = episodes[:cut]
    kept = episodes[cut:]
    if not to_summarize:
        return no_op

    # Reuse any prior summary for this thread so summaries compound.
    previous = next(
        (e.content for e in reversed(to_summarize) if e.source == "thread_summary"),
        None,
    )

    summary_text = summarizer.summarize(
        to_summarize,
        previous_summary=previous,
        custom_instructions=custom_instructions,
    )

    summary_episode = Episode(
        content=summary_text,
        timestamp=to_summarize[-1].timestamp,  # anchor at the cut boundary
        importance=max(e.importance for e in to_summarize),
        source="thread_summary",
        context={
            "summarized_ids": [e.id for e in to_summarize],
            "summarized_count": len(to_summarize),
            "tokens_before": tokens_before,
        },
        thread_id=thread_id,
        owner_id=owner_id,
    )
    episodic.record(summary_episode)
    episodic.mark_consolidated([e.id for e in to_summarize])

    tokens_after = (
        estimate_tokens(summary_text)
        + sum(estimate_tokens(e.content) for e in kept)
    )

    return ThreadSummaryResult(
        thread_id=thread_id,
        summary_episode=summary_episode,
        summarized_ids=[e.id for e in to_summarize],
        kept_ids=[e.id for e in kept],
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
