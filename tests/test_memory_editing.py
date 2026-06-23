"""Tests for memory editing and correction APIs."""

from datetime import datetime, timedelta

import pytest

from remembox import Remembox


@pytest.fixture
def memory(tmp_path):
    db = tmp_path / "edit.db"
    m = Remembox(str(db))
    yield m
    m.close()


class TestEpisodeEditing:
    def test_update_episode_content(self, memory):
        ep = memory.record("User said they love hiking")
        updated = memory.update_episode(ep.id, content="User said they love trail running")
        assert updated is not None
        assert updated.content == "User said they love trail running"
        # Retrieval reflects the edit
        fetched = memory._episodic.get(ep.id)
        assert fetched.content == updated.content

    def test_update_episode_importance_emotion(self, memory):
        ep = memory.record("small thing", importance=0.2)
        updated = memory.update_episode(ep.id, importance=0.9, emotion="excited")
        assert updated.importance == 0.9
        assert updated.emotion == "excited"

    def test_update_episode_merges_context_by_default(self, memory):
        ep = memory.record("hello", context={"channel": "slack"})
        updated = memory.update_episode(ep.id, context={"thread_id": "123"})
        assert updated.context["channel"] == "slack"
        assert updated.context["thread_id"] == "123"

    def test_update_episode_can_replace_context(self, memory):
        ep = memory.record("hello", context={"channel": "slack"})
        updated = memory.update_episode(ep.id, context={"thread_id": "123"}, merge_context=False)
        assert "channel" not in updated.context
        assert updated.context["thread_id"] == "123"

    def test_update_episode_preserves_access_and_consolidated(self, memory):
        ep = memory.record("to edit")
        memory._episodic.increment_access(ep.id)
        memory._episodic.mark_consolidated([ep.id])
        updated = memory.update_episode(ep.id, content="edited")
        assert updated.access_count == 1
        assert updated.consolidated is True

    def test_update_episode_missing_id_returns_none(self, memory):
        assert memory.update_episode("no-such-id", content="x") is None

    def test_annotate_episode_correction(self, memory):
        ep = memory.record("User said they live in Mumbai")
        annotated = memory.annotate_episode(
            ep.id,
            correction="Actually Delhi",
            accuracy="flagged",
            notes="User corrected themselves",
        )
        annotations = annotated.context["__annotations__"]
        assert len(annotations) == 1
        assert annotations[0]["correction"] == "Actually Delhi"
        assert annotations[0]["accuracy"] == "flagged"
        assert "timestamp" in annotations[0]

    def test_annotate_episode_appends(self, memory):
        ep = memory.record("base")
        memory.annotate_episode(ep.id, notes="first")
        memory.annotate_episode(ep.id, notes="second")
        fetched = memory._episodic.get(ep.id)
        assert len(fetched.context["__annotations__"]) == 2

    def test_annotate_episode_missing_id_returns_none(self, memory):
        assert memory.annotate_episode("missing", notes="x") is None


class TestFactEditing:
    def test_edit_fact_in_place(self, memory):
        fact, action = memory.learn("user", "prefers", "tea", confidence=0.6)
        assert action == "new"
        updated = memory.edit_fact(fact.id, confidence=0.95, obj="coffee")
        assert updated.object == "coffee"
        assert updated.confidence == 0.95
        assert updated.predicate == "prefers"
        # Same ID preserved
        assert updated.id == fact.id

    def test_edit_fact_missing_id_returns_none(self, memory):
        assert memory.edit_fact("nope", confidence=0.5) is None

    def test_correct_fact_deactivates_old_and_inserts_new(self, memory):
        old, _ = memory.learn("user", "lives_in", "Mumbai", confidence=0.8)
        old_id = old.id
        corrected, action = memory.correct_fact(old_id, new_object="Delhi", new_confidence=0.95)
        assert action == "corrected"
        assert corrected.object == "Delhi"
        assert corrected.confidence == 0.95
        assert corrected.id != old_id
        assert corrected.is_active is True
        assert corrected.subject == "user"
        assert corrected.predicate == "lives_in"

        old_refetched = memory._semantic.get(old_id)
        assert old_refetched.is_active is False

    def test_correct_fact_preserves_provenance(self, memory):
        ep = memory.record("source text")
        fact, _ = memory.learn("user", "works_at", "Google", source_episode_id=ep.id)
        corrected, _ = memory.correct_fact(fact.id, new_object="OpenAI")
        assert ep.id in corrected.source_episode_ids

    def test_correct_fact_missing_id_raises(self, memory):
        with pytest.raises(KeyError):
            memory.correct_fact("missing", new_object="x")


class TestOwnerIsolationEditing:
    def test_cannot_edit_other_owner_episode(self, tmp_path):
        db = tmp_path / "shared.db"
        alice = Remembox(str(db), owner_id="alice")
        bob = Remembox(str(db), owner_id="bob")

        ep = alice.record("Alice secret")
        assert bob.update_episode(ep.id, content="hacked") is None
        assert bob.annotate_episode(ep.id, notes="x") is None

        alice.close()
        bob.close()
