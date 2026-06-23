"""Tests for hierarchical / threaded episodes."""

import pytest

from membox import Membox


@pytest.fixture
def memory(tmp_path):
    m = Membox(str(tmp_path / "threads.db"))
    yield m
    m.close()


class TestThreadRecording:
    def test_record_with_thread_id(self, memory):
        ep = memory.record("hello", thread_id="t1")
        assert ep.thread_id == "t1"
        assert ep.depth == 0
        assert ep.parent_id is None

    def test_record_with_parent_and_depth(self, memory):
        parent = memory.record("top level", thread_id="t1")
        child = memory.record("reply", thread_id="t1", parent_id=parent.id, depth=1)
        assert child.parent_id == parent.id
        assert child.depth == 1

    def test_thread_round_trip(self, memory):
        a = memory.record("first", thread_id="incident-42")
        b = memory.record("second", thread_id="incident-42")
        c = memory.record("other", thread_id="incident-7")

        thread = memory.thread("incident-42")
        assert len(thread) == 2
        assert {e.id for e in thread} == {a.id, b.id}

    def test_thread_children(self, memory):
        parent = memory.record("parent", thread_id="t1")
        c1 = memory.record("child 1", thread_id="t1", parent_id=parent.id, depth=1)
        c2 = memory.record("child 2", thread_id="t1", parent_id=parent.id, depth=1)
        memory.record("orphan")

        children = memory.thread_children(parent.id)
        assert len(children) == 2
        assert {e.id for e in children} == {c1.id, c2.id}

    def test_threads_list(self, memory):
        memory.record("a", thread_id="t1")
        memory.record("b", thread_id="t2")
        memory.record("c")
        assert set(memory.threads()) == {"t1", "t2"}


class TestThreadOwnerIsolation:
    def test_cannot_see_other_owner_thread(self, tmp_path):
        db = tmp_path / "shared.db"
        alice = Membox(str(db), owner_id="alice")
        bob = Membox(str(db), owner_id="bob")
        alice.record("x", thread_id="secret")
        assert alice.threads() == ["secret"]
        assert bob.threads() == []
        alice.close()
        bob.close()
