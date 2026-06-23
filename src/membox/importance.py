"""Importance and emotion scoring for memory events.

The production package is dependency-light, so the core ships with a
rule-based scorer. For higher-quality, context-aware scoring, inject
an LLM-based scorer (works with any client providing a chat.completions
interface, e.g. OpenAI, Anthropic, OpenRouter).

Usage:
    from membox import Membox
    from membox.importance import LLMImportanceScorer

    scorer = LLMImportanceScorer(client=openai_client)
    memory = Membox("agent.db", importance_scorer=scorer)

    # importance and emotion are inferred automatically
    memory.record("I just got promoted to CEO!")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoreResult:
    """Result of importance/emotion scoring."""
    importance: float  # 0.0 to 1.0
    emotion: str | None = None  # e.g., "happy", "stressed"


class ImportanceScorer(ABC):
    """Abstract importance + emotion scorer."""

    @abstractmethod
    def score(self, content: str) -> ScoreResult:
        """Return importance and optional emotion for a piece of text."""
        ...


class RuleBasedImportanceScorer(ImportanceScorer):
    """Fast, dependency-free importance scorer.

    Uses keyword heuristics. Life events and emotional intensity score
    higher; greetings and questions score lower.
    """

    # (regex pattern, importance boost, optional emotion)
    _RULES: list[tuple[re.Pattern, float, str | None]] = [
        # Life events
        (re.compile(r"\b(promoted|fired|married|divorced|moved|graduated|pregnant|accident|hospital)\b", re.I), 0.9, None),
        (re.compile(r"\b(died|born|engaged|retired|diagnosed)\b", re.I), 1.0, None),
        # Emotional intensity
        (re.compile(r"\b(love|hate|amazing|terrible|devastated|ecstatic|furious|grateful|brilliant|awful)\b", re.I), 0.7, None),
        (re.compile(r"\b(excited|happy|glad|great|wonderful)\b", re.I), 0.4, "happy"),
        (re.compile(r"\b(stressed|overwhelmed|panic|crisis|ugh)\b", re.I), 0.5, "stressed"),
        (re.compile(r"\b(frustrated|annoying|broken|doesn't work)\b", re.I), 0.4, "frustrated"),
        (re.compile(r"\b(sad|lonely|depressed|upset)\b", re.I), 0.4, "sad"),
        # Preferences / identity
        (re.compile(r"\b(i prefer|i like|my favorite|i always|i never|my name is|i work at|i live in)\b", re.I), 0.6, None),
    ]

    def score(self, content: str) -> ScoreResult:
        cl = content.lower()
        score = 0.3
        detected_emotion: str | None = None

        for pattern, boost, emotion in self._RULES:
            if pattern.search(content):
                score = max(score, boost)
                if emotion and not detected_emotion:
                    detected_emotion = emotion

        # Questions → lower importance
        if content.strip().endswith("?"):
            score = min(score, 0.3)

        # Greetings → lowest
        greetings = ["hello", "hi ", "hey ", "good morning", "good night", "goodnight", "bye"]
        if any(cl.startswith(g) or cl == g.strip() for g in greetings):
            score = min(score, 0.2)

        return ScoreResult(
            importance=round(max(0.0, min(1.0, score)), 3),
            emotion=detected_emotion,
        )


class LLMImportanceScorer(ImportanceScorer):
    """Use an LLM to score importance and detect emotion.

    Works with any client exposing `client.chat.completions.create(...)`
    compatible with the OpenAI API shape.

    Example:
        scorer = LLMImportanceScorer(client=openai_client, model="gpt-4o-mini")
        scorer.score("I got promoted today!")
        # → ScoreResult(importance=0.95, emotion="ecstatic")
    """

    _SYSTEM_PROMPT = (
        "You rate how important a user message is for a memory system. "
        "Respond with exactly two lines:\n"
        "importance: <float 0.0-1.0>\n"
        "emotion: <single lowercase word or 'none'>\n\n"
        "Guidelines:\n"
        "- 1.0 = life-changing events (death, marriage, birth, major promotion)\n"
        "- 0.8 = significant personal/professional news\n"
        "- 0.5 = preferences, routines, work updates\n"
        "- 0.2 = small talk, greetings, transient questions\n"
        "- 0.0 = spam, system messages"
    )

    def __init__(self, client, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def score(self, content: str) -> ScoreResult:
        if not content or not content.strip():
            return ScoreResult(importance=0.0, emotion=None)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception:
            # Graceful fallback on LLM failure
            return RuleBasedImportanceScorer().score(content)

        importance: float = 0.3
        emotion: str | None = None

        for line in text.splitlines():
            line = line.strip().lower()
            if line.startswith("importance:"):
                try:
                    importance = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("emotion:"):
                raw = line.split(":", 1)[1].strip()
                if raw and raw != "none":
                    emotion = raw

        return ScoreResult(
            importance=max(0.0, min(1.0, importance)),
            emotion=emotion,
        )
