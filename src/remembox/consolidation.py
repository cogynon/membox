"""Consolidation: extract structured facts from episode batches.

Rule-based extraction for the core (no LLM dependency).
Provides abstract interface to plug in LLM-based extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from remembox._store import EpisodicStoreProtocol, SemanticStoreProtocol
from remembox.config import MemoryConfig
from remembox.models import Episode, Fact


class Consolidator(ABC):
    """Abstract interface for episode → fact extraction.

    Implement extract_facts() to support different extraction strategies:
    - RuleBasedConsolidator: keyword patterns (no deps, ships with core)
    - LLM-based: subclass and call your preferred LLM API
    """

    @abstractmethod
    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        """Extract facts from a list of Episode objects.

        Returns list of dicts with keys: subject, predicate, object, confidence.

        Receives full Episode objects (not just strings) so consolidators
        can use timestamp, importance, emotion, and context when deciding
        what facts to extract.
        """
        ...

    def extract(self, contents: list[str]) -> list[dict]:
        """Deprecated compatibility shim.

        Kept for backward compatibility; forwards to extract_facts().
        New code should override extract_facts().
        """
        episodes = [
            Episode(content=c, timestamp=datetime.now(), importance=0.5)
            for c in contents
        ]
        return self.extract_facts(episodes)


class RuleBasedConsolidator(Consolidator):
    """Pattern-matching fact extractor. No external dependencies.

    Catches common patterns like "I prefer X", "I live in Y", etc.
    Good enough for demos; swap with LLM-based for production accuracy.
    """

    _PATTERNS = [
        # (trigger_phrases, predicate)
        (["i prefer ", "i like ", "i love ", "i enjoy "], "prefers"),
        (["i live in ", "i moved to ", "i'm based in ", "i am based in "], "lives_in"),
        (["i work at ", "i work for ", "i joined "], "works_at"),
        (["my name is ", "i'm ", "i am "], "name"),
        (["my favorite ", "my favourite "], "favorite"),
    ]

    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        """Extract facts from episodes, preserving source episode IDs.

        Each episode yields at most one fact: patterns are tried in order and
        the *earliest-matching* (lowest index in the content) trigger wins, so
        overlapping triggers like "i'm" vs "i'm based in" don't both fire.
        """
        results = []
        for ep in episodes:
            cl = ep.content.lower()
            best: tuple[int, int, str] | None = None  # (position, trigger_len, predicate)
            for triggers, predicate in self._PATTERNS:
                for trigger in triggers:
                    pos = cl.find(trigger)
                    if pos == -1:
                        continue
                    # Prefer the trigger that appears earliest; on a tie, prefer
                    # the longer (more specific) trigger.
                    if best is None or pos < best[0] or (
                        pos == best[0] and len(trigger) > best[1]
                    ):
                        best = (pos, len(trigger), predicate)
            if best is None:
                continue
            pos, trigger_len, predicate = best
            obj = ep.content[pos + trigger_len:]
            # Clean: stop at sentence boundaries
            for sep in [".", ",", "!", "?", " and ", " but "]:
                obj = obj.split(sep)[0]
            obj = obj.strip().strip("'\"")
            # Strip trailing time adverbs that get greedily captured
            # ("Berlin now" -> "Berlin"). Known limitation: this is a coarse
            # heuristic; LLM-based consolidation is the production path.
            for adverb in (" now", " today", " currently", " these days", " lately"):
                if obj.lower().endswith(adverb):
                    obj = obj[: -len(adverb)].rstrip()
            if obj and 2 < len(obj) < 50:
                results.append({
                    "subject": "user",
                    "predicate": predicate,
                    "object": obj,
                    "confidence": 0.7,
                    "source_episode_id": ep.id,
                })
        return results


def consolidate(episodic: EpisodicStoreProtocol, semantic: SemanticStoreProtocol,
                consolidator: Consolidator | None = None,
                config: MemoryConfig | None = None,
                now: datetime | None = None) -> dict:
    """Run consolidation: extract facts from unconsolidated episodes.

    1. Get batch of unconsolidated episodes older than min_age
    2. Extract facts using the consolidator
    3. Learn each fact in the semantic store
    4. Mark episodes as consolidated

    Returns summary: {"episodes_processed": N, "facts_extracted": N, "facts": [...]}
    """
    config = config or MemoryConfig()
    now = now or datetime.now()
    consolidator = consolidator or RuleBasedConsolidator()

    # Get unconsolidated episodes older than min_age
    min_age = timedelta(hours=config.consolidation_min_age_hours)
    candidates = episodic.unconsolidated(limit=config.consolidation_batch_size)
    eligible = [ep for ep in candidates if (now - ep.timestamp) >= min_age]

    if not eligible:
        return {"episodes_processed": 0, "facts_extracted": 0, "facts": []}

    # Extract facts, passing full Episode objects to the consolidator.
    raw_facts = consolidator.extract_facts(eligible)

    # Learn each fact, linking it to the specific episode it came from.
    learned = []
    productive_ids: set[str] = set()
    for fact_data in raw_facts:
        source_id = fact_data.get("source_episode_id")
        fact, action = semantic.learn(
            subject=fact_data["subject"],
            predicate=fact_data["predicate"],
            obj=fact_data["object"],
            confidence=fact_data["confidence"],
            source_episode_id=source_id,
        )
        learned.append({"fact": fact, "action": action})
        if source_id:
            productive_ids.add(source_id)

    # Only mark episodes that actually produced ≥1 fact as consolidated.
    # Episodes that extracted nothing stay unconsolidated so a better
    # extractor can reprocess them later.
    to_mark = [ep.id for ep in eligible if ep.id in productive_ids]
    episodic.mark_consolidated(to_mark)

    return {
        "episodes_processed": len(to_mark),
        "facts_extracted": len(learned),
        "facts": learned,
    }


_SYSTEM_PROMPT = """\
You extract durable, long-lived facts from a coding agent's conversation log.

Each input item is a user prompt observed during a coding session. Your job:
identify facts worth remembering across sessions — stable preferences,
decisions, environment details, and project context. Ignore transient questions,
greetings, and one-off task descriptions.

Return ONLY a JSON array (no prose, no markdown fences). Each element:
  {"subject": "user", "predicate": "<snake_case>", "object": "<short value>", "confidence": <0.0-1.0>}

Guidelines for predicates:
- preferences: prefers, likes, dislikes, avoids
- environment: uses, runs, editor, shell, os
- decisions: decided, chose, uses_for, standardized_on
- profile: name, role, team, works_on

Examples:
  "I prefer tabs"                       → {"subject":"user","predicate":"prefers","object":"tabs","confidence":0.9}
  "we decided to use postgres"          → {"subject":"user","predicate":"decided","object":"postgres","confidence":0.85}
  "I'm on zsh"                          → {"subject":"user","predicate":"shell","object":"zsh","confidence":0.8}
  "what does this function do"          → (skip — transient question)
  "hey thanks"                          → (skip — small talk)

If nothing is worth remembering, return [].
"""


class LLMConsolidator(Consolidator):
    """LLM-based episode → fact extractor.

    Requires the `openai` library to be installed (e.g. `pip install remembox[llm]`).
    """

    def __init__(self, client: Any, model: str, max_episodes: int = 20) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai is required to use LLMConsolidator. "
                "Install it with: pip install 'remembox[llm]'"
            )
        self.client = client
        self.model = model
        self.max_episodes = max_episodes

    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        if not episodes:
            return []
        # Bound the batch so the prompt stays small and cheap.
        batch = episodes[: self.max_episodes]
        user_msg = self._format_batch(batch)
        raw = self._call_llm(user_msg)
        return self._parse(raw)

    def _format_batch(self, episodes: list[Episode]) -> str:
        import json
        items = []
        for ep in episodes:
            content = ep.content
            # Strip the role prefix the pi extension adds ("User: " / "Assistant: ").
            for prefix in ("User: ", "Assistant: "):
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
            # Bound per-episode size so a few huge replies don't blow up the
            # consolidation prompt. 20 episodes × ~1k chars ≈ 5k input tokens.
            if len(content) > 1000:
                content = content[:1000] + " …[truncated]"
            items.append(json.dumps({"id": ep.id, "content": content}))
        return "Extract facts from these observed prompts:\n" + "\n".join(items)

    def _call_llm(self, user_msg: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            # Some OpenRouter reasoning models return content=None when they
            # spend the whole budget on thinking tokens. Treat as "no facts".
            return (response.choices[0].message.content or "").strip()
        except Exception:
            # Fail soft: no facts rather than crash consolidation. The
            # rule-based path can still run as a fallback if configured.
            return "[]"

    def _parse(self, raw: str) -> list[dict]:
        """Parse the LLM's JSON-array response, tolerating markdown fences."""
        import json
        import re
        if not raw:
            return []
        # Strip ```json ... ``` fences if the model wrapped the output.
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        facts: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject", "user")
            predicate = item.get("predicate")
            obj = item.get("object")
            if not predicate or not obj:
                continue
            confidence = float(item.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            facts.append({
                "subject": subject,
                "predicate": str(predicate),
                "object": str(obj),
                "confidence": confidence,
                "source_episode_id": None,  # LLM facts aren't tied to one episode
            })
        return facts

