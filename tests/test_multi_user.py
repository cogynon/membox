"""Tests for multi-user isolation (P0.1)."""

from datetime import datetime, timedelta

import pytest

from remembox import Remembox, MemoryConfig

NOW = datetime(2026, 3, 25, 12, 0, 0)


class TestMultiUserIsolation:
    """Episodes and facts must be scoped to owner_id — no cross-contamination."""

    def test_owner_id_defaults_to_default(self):
        m = Remembox(":memory:")
        ep = m.record("hello")
        assert ep.owner_id == "default"
        facts = m.about("user")
        assert all(f.owner_id == "default" for f in facts)

    def test_different_owners_dont_see_each_other_episodes(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.record("Alice's secret", importance=0.9)
        bob.record("Bob's secret", importance=0.9)

        alice_recent = alice.recent(10)
        bob_recent = bob.recent(10)

        assert len(alice_recent) == 1
        assert alice_recent[0].content == "Alice's secret"
        assert "Bob" not in alice_recent[0].content

        assert len(bob_recent) == 1
        assert bob_recent[0].content == "Bob's secret"

    def test_different_owners_dont_see_each_other_facts(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.learn("user", "name", "Alice", confidence=0.95)
        bob.learn("user", "name", "Bob", confidence=0.95)

        alice_facts = alice.about("user")
        bob_facts = bob.about("user")

        assert len(alice_facts) == 1
        assert alice_facts[0].object == "Alice"

        assert len(bob_facts) == 1
        assert bob_facts[0].object == "Bob"

    def test_same_db_same_owner_shares_data(self, tmp_db):
        m1 = Remembox(tmp_db, owner_id="charlie")
        m1.record("shared memory")
        m1.learn("user", "prefers", "coffee")

        m2 = Remembox(tmp_db, owner_id="charlie")
        assert len(m2.recent(10)) == 1
        assert len(m2.about("user")) == 1

    def test_forget_is_owner_scoped(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.record("old trivial alice", importance=0.1,
                     timestamp=NOW - timedelta(days=30))
        bob.record("old trivial bob", importance=0.1,
                   timestamp=NOW - timedelta(days=30))

        alice.forget(now=NOW)

        # Bob's memory should survive Alice's forget pass
        assert bob._episodic.count() == 1
        assert bob.recent(1)[0].content == "old trivial bob"

    def test_search_is_owner_scoped(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.record("coffee in Paris")
        bob.record("coffee in London")

        alice_results = alice.search("coffee")
        bob_results = bob.search("coffee")

        assert len(alice_results) == 1
        assert "Paris" in alice_results[0].content

        assert len(bob_results) == 1
        assert "London" in bob_results[0].content

    def test_recall_is_owner_scoped(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.record("I love hiking in the Alps", importance=0.8)
        bob.record("I love coding in Python", importance=0.8)

        alice_results = alice.recall("hiking", k=3, now=NOW)
        bob_results = bob.recall("coding", k=3, now=NOW)

        assert len(alice_results) >= 1
        assert "Alps" in alice_results[0].episode.content

        assert len(bob_results) >= 1
        assert "Python" in bob_results[0].episode.content

    def test_stats_reflect_owner_scope(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        alice.record("alice event")
        alice.learn("user", "name", "Alice")
        bob.record("bob event")

        alice_stats = alice.stats()
        bob_stats = bob.stats()

        assert alice_stats["owner_id"] == "alice"
        assert alice_stats["episodes"]["total"] == 1
        assert alice_stats["facts"]["active"] == 1

        assert bob_stats["owner_id"] == "bob"
        assert bob_stats["episodes"]["total"] == 1
        assert bob_stats["facts"]["active"] == 0

    def test_repr_shows_owner(self, tmp_db):
        m = Remembox(tmp_db, owner_id="dave")
        m.record("test")
        assert "owner='dave'" in repr(m)

    def test_delete_is_owner_scoped(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        bob = Remembox(tmp_db, owner_id="bob")

        ep = alice.record("delete me")
        bob.record("keep me")

        alice._episodic.delete([ep.id])

        assert alice._episodic.count() == 0
        assert bob._episodic.count() == 1


class TestMultiUserPersistence:
    """Data must survive restarts and remain scoped."""

    def test_persistence_across_restarts(self, tmp_db):
        m1 = Remembox(tmp_db, owner_id="eve")
        m1.record("persistent event", importance=0.9)
        m1.learn("user", "name", "Eve", confidence=0.95)
        m1.close()

        m2 = Remembox(tmp_db, owner_id="eve")
        assert m2.recent(1)[0].content == "persistent event"
        assert m2.about("user")[0].object == "Eve"

    def test_other_owner_persistence_not_leaked(self, tmp_db):
        alice = Remembox(tmp_db, owner_id="alice")
        alice.record("Alice's private data")
        alice.close()

        mallory = Remembox(tmp_db, owner_id="mallory")
        assert mallory.recent(10) == []
        assert mallory.search("Alice") == []

    def test_default_owner_persists(self, tmp_db):
        m1 = Remembox(tmp_db)  # default owner_id="default"
        m1.record("default user data")
        m1.close()

        m2 = Remembox(tmp_db)
        assert len(m2.recent(10)) == 1
