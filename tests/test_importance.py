"""Tests for importance + emotion scoring."""

import pytest

from remembox import Remembox, MemoryConfig
from remembox.importance import RuleBasedImportanceScorer, LLMImportanceScorer, ScoreResult


class TestRuleBasedImportanceScorer:

    def test_life_event_scores_high(self):
        scorer = RuleBasedImportanceScorer()
        result = scorer.score("I just got promoted to CEO today!")
        assert result.importance >= 0.9

    def test_greeting_scores_low(self):
        scorer = RuleBasedImportanceScorer()
        result = scorer.score("Hi there")
        assert result.importance <= 0.2

    def test_question_scores_low(self):
        scorer = RuleBasedImportanceScorer()
        result = scorer.score("What is the weather today?")
        assert result.importance <= 0.3

    def test_preference_scores_moderate(self):
        scorer = RuleBasedImportanceScorer()
        result = scorer.score("I prefer black coffee in the morning")
        assert 0.5 <= result.importance <= 1.0

    def test_emotion_detected(self):
        scorer = RuleBasedImportanceScorer()
        result = scorer.score("I'm so happy today!")
        assert result.emotion == "happy"


class FakeLLMResponse:
    def __init__(self, content: str):
        self.choices = [self]
        self.message = self
        self.content = content


class FakeCompletionsAPI:
    def __init__(self, response_text: str):
        self._response = response_text

    def create(self, **kwargs):
        return FakeLLMResponse(self._response)


class FakeLLMClient:
    """Mock OpenAI-compatible client."""

    def __init__(self, response_text: str):
        self.chat = self
        self.completions = FakeCompletionsAPI(response_text)


class TestLLMImportanceScorer:

    def test_parses_importance_and_emotion(self):
        client = FakeLLMClient("importance: 0.92\nemotion: ecstatic")
        scorer = LLMImportanceScorer(client=client, model="test-model")
        result = scorer.score("I got promoted!")
        assert result.importance == pytest.approx(0.92, abs=0.01)
        assert result.emotion == "ecstatic"

    def test_clamps_out_of_range(self):
        client = FakeLLMClient("importance: 1.5\nemotion: none")
        scorer = LLMImportanceScorer(client=client, model="test-model")
        result = scorer.score("something")
        assert result.importance == pytest.approx(1.0, abs=0.01)

    def test_falls_back_on_llm_error(self):
        class BrokenClient:
            def chat_completions_create(self, **kwargs):
                raise RuntimeError("API down")

        scorer = LLMImportanceScorer(client=BrokenClient(), model="test-model")
        result = scorer.score("I got promoted!")
        assert result.importance >= 0.8  # rule-based fallback handles promotions

    def test_empty_content(self):
        client = FakeLLMClient("")
        scorer = LLMImportanceScorer(client=client, model="test-model")
        result = scorer.score("")
        assert result.importance == 0.0


class TestRememboxAutoImportance:

    def test_auto_score_with_config(self):
        config = MemoryConfig(auto_score_importance=True)
        memory = Remembox(":memory:", config=config)
        ep = memory.record("I just got promoted to CEO today!")
        assert ep.importance >= 0.9

    def test_manual_importance_overrides_scorer(self):
        config = MemoryConfig(auto_score_importance=True)
        memory = Remembox(":memory:", config=config)
        ep = memory.record("I just got promoted!", importance=0.2)
        assert ep.importance == pytest.approx(0.2, abs=0.01)

    def test_injected_scorer(self):
        class ConstantScorer:
            def score(self, content: str) -> ScoreResult:
                return ScoreResult(importance=0.77, emotion="calm")

        memory = Remembox(":memory:", importance_scorer=ConstantScorer())
        ep = memory.record("anything")
        assert ep.importance == pytest.approx(0.77, abs=0.01)
        assert ep.emotion == "calm"

    def test_no_scorer_uses_default(self):
        memory = Remembox(":memory:")
        ep = memory.record("hello")
        assert ep.importance == pytest.approx(0.5, abs=0.01)
