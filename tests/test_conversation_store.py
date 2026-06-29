"""
Tests for backend.chat.conversation.ConversationStore.

Covers:
  - basic add/get history
  - max-history trimming
  - TTL-based eviction (per-call + cleanup_all)
  - thread-safety under concurrent writers
  - edge cases (None conversation_id, empty store, etc.)
"""

import sys
import os
import time
import threading
from unittest.mock import patch

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chat.conversation import (
    ConversationStore,
    MAX_CONVERSATION_HISTORY,
    CONVERSATION_TTL_SECONDS,
)


class TestConversationStoreBasic:
    """Basic add / get / trim behavior."""

    def test_get_history_returns_empty_for_none_id(self):
        store = ConversationStore()
        assert store.get_history(None) == []

    def test_get_history_returns_empty_for_unknown_id(self):
        store = ConversationStore()
        assert store.get_history("unknown-id") == []

    def test_add_then_get_returns_message(self):
        store = ConversationStore()
        store.add_message("cid-1", "user", "hello")
        history = store.get_history("cid-1")
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "hello"}

    def test_get_history_returns_copy_not_reference(self):
        """Callers must be able to mutate the returned list without
        affecting the store's internal state."""
        store = ConversationStore()
        store.add_message("cid-1", "user", "hello")
        history = store.get_history("cid-1")
        history.append({"role": "assistant", "content": "mutated"})
        # Internal store should NOT have the mutated message.
        assert len(store.get_history("cid-1")) == 1

    def test_add_message_ignores_none_id(self):
        store = ConversationStore()
        store.add_message(None, "user", "should-be-ignored")
        # No assertion needed — if this raises, the test fails. The
        # important check is that the store is still empty afterwards,
        # which we verify via the absence of state.
        assert store.cleanup_all() == 0

    def test_multiple_messages_preserve_order(self):
        store = ConversationStore()
        store.add_message("cid-1", "user", "first")
        store.add_message("cid-1", "assistant", "second")
        store.add_message("cid-1", "user", "third")
        history = store.get_history("cid-1")
        assert [m["content"] for m in history] == ["first", "second", "third"]


class TestConversationStoreTrimming:
    """Max-history trimming behavior."""

    def test_default_max_history_is_20(self):
        assert MAX_CONVERSATION_HISTORY == 20

    def test_history_trimmed_to_max(self):
        store = ConversationStore(max_history=3)
        for i in range(5):
            store.add_message("cid-1", "user", f"msg-{i}")
        history = store.get_history("cid-1")
        assert len(history) == 3
        # Oldest messages dropped, newest 3 kept.
        assert [m["content"] for m in history] == ["msg-2", "msg-3", "msg-4"]

    def test_history_at_max_not_trimmed(self):
        store = ConversationStore(max_history=3)
        for i in range(3):
            store.add_message("cid-1", "user", f"msg-{i}")
        history = store.get_history("cid-1")
        assert len(history) == 3


class TestConversationStoreTTL:
    """TTL-based eviction behavior."""

    def test_default_ttl_is_one_hour(self):
        assert CONVERSATION_TTL_SECONDS == 3600

    def test_expired_conversation_evicted_on_get(self):
        store = ConversationStore(ttl_seconds=1)
        store.add_message("cid-1", "user", "hello")
        # Patch time.time so we don't have to actually sleep.
        with patch("chat.conversation.time.time", return_value=time.time() + 10):
            history = store.get_history("cid-1")
        assert history == []

    def test_expired_conversation_evicted_by_cleanup_all(self):
        store = ConversationStore(ttl_seconds=1)
        store.add_message("cid-1", "user", "hello")
        store.add_message("cid-2", "user", "world")
        # Advance the clock past TTL.
        with patch("chat.conversation.time.time", return_value=time.time() + 10):
            evicted = store.cleanup_all()
        assert evicted == 2
        assert store.get_history("cid-1") == []
        assert store.get_history("cid-2") == []

    def test_non_expired_conversation_survives_cleanup(self):
        store = ConversationStore(ttl_seconds=3600)
        store.add_message("cid-1", "user", "hello")
        evicted = store.cleanup_all()
        assert evicted == 0
        assert len(store.get_history("cid-1")) == 1


class TestConversationStoreThreadSafety:
    """Thread-safety: concurrent writers must not corrupt the store."""

    def test_concurrent_writers_produce_consistent_state(self):
        """Multiple threads adding messages to the same conversation_id
        must produce exactly N total messages (no lost updates, no
        duplicates from races)."""
        store = ConversationStore(max_history=10000)
        cid = "concurrent-cid"
        n_threads = 8
        n_per_thread = 50

        def writer(thread_id: int):
            for i in range(n_per_thread):
                store.add_message(cid, "user", f"t{thread_id}-m{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        history = store.get_history(cid)
        # No messages lost, no duplicates from races.
        assert len(history) == n_threads * n_per_thread
        # All messages are unique (no double-counting).
        contents = [m["content"] for m in history]
        assert len(set(contents)) == len(contents)

    def test_concurrent_writers_different_ids_isolated(self):
        store = ConversationStore(max_history=1000)
        cids = [f"cid-{i}" for i in range(5)]
        n_per_cid = 20

        def writer(cid: str):
            for i in range(n_per_cid):
                store.add_message(cid, "user", f"m{i}")

        threads = [threading.Thread(target=writer, args=(c,)) for c in cids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for cid in cids:
            assert len(store.get_history(cid)) == n_per_cid
