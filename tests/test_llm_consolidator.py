import sys
from unittest.mock import MagicMock
import pytest

from remembox.models import Episode
from remembox.consolidation import LLMConsolidator


class TestLLMConsolidator:

    def test_llm_consolidator_missing_openai(self, monkeypatch):
        # Simulate openai not being installed by setting sys.modules['openai'] to None
        monkeypatch.setitem(sys.modules, "openai", None)

        client = MagicMock()
        with pytest.raises(ImportError) as excinfo:
            LLMConsolidator(client, model="gpt-4")
        assert "openai is required to use LLMConsolidator" in str(excinfo.value)

    def test_llm_consolidator_extracts_facts(self):
        client = MagicMock()
        mock_response = MagicMock()
        # Mock completions.create response
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        ```json
        [
            {"subject": "user", "predicate": "prefers", "object": "black coffee", "confidence": 0.9},
            {"subject": "user", "predicate": "decided", "object": "postgres", "confidence": 0.85}
        ]
        ```
        """
        client.chat.completions.create.return_value = mock_response

        consolidator = LLMConsolidator(client, model="gpt-4")
        episodes = [
            Episode(content="I prefer black coffee", importance=0.7),
            Episode(content="We decided to use postgres for audit log", importance=0.8),
        ]
        facts = consolidator.extract_facts(episodes)

        assert len(facts) == 2
        assert facts[0]["subject"] == "user"
        assert facts[0]["predicate"] == "prefers"
        assert facts[0]["object"] == "black coffee"
        assert facts[0]["confidence"] == 0.9

        assert facts[1]["subject"] == "user"
        assert facts[1]["predicate"] == "decided"
        assert facts[1]["object"] == "postgres"
        assert facts[1]["confidence"] == 0.85

    def test_llm_consolidator_strips_role_prefixes(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"
        client.chat.completions.create.return_value = mock_response

        consolidator = LLMConsolidator(client, model="gpt-4")
        episodes = [
            Episode(content="User: hello world", importance=0.5),
            Episode(content="Assistant: how can I help", importance=0.5),
        ]
        consolidator.extract_facts(episodes)

        # Retrieve the user message argument sent to create()
        call_args = client.chat.completions.create.call_args[1]
        user_msg = call_args["messages"][1]["content"]

        assert "hello world" in user_msg
        assert "how can I help" in user_msg
        assert "User: hello world" not in user_msg
        assert "Assistant: how can I help" not in user_msg

    def test_llm_consolidator_truncates_long_episodes(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"
        client.chat.completions.create.return_value = mock_response

        consolidator = LLMConsolidator(client, model="gpt-4")
        long_content = "a" * 2000
        episodes = [Episode(content=long_content, importance=0.5)]
        consolidator.extract_facts(episodes)

        call_args = client.chat.completions.create.call_args[1]
        user_msg = call_args["messages"][1]["content"]

        assert "truncated" in user_msg
        assert len(long_content) > 1000

    def test_llm_consolidator_fails_soft_on_error(self):
        client = MagicMock()
        # Force completions.create to raise an exception
        client.chat.completions.create.side_effect = Exception("API Error")

        consolidator = LLMConsolidator(client, model="gpt-4")
        episodes = [Episode(content="I love python", importance=0.5)]

        # Should not raise exception, should fail soft and return []
        facts = consolidator.extract_facts(episodes)
        assert facts == []
