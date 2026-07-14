"""
In-memory conversation store with TTL enforcement.

Fixes Critical Issue #3: conversations now expire after CONVERSATION_TTL_SECONDS.

Thread-safety: all public methods are guarded by an instance-level
``threading.Lock`` so the store can be safely shared across WSGI worker
threads (e.g. ``gunicorn -w 4 --threads 2``). Without this lock, concurrent
requests for the same ``conversation_id`` could corrupt the history list or
race the TTL-eviction check.
"""

import threading
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CONVERSATION_HISTORY = 20
CONVERSATION_TTL_SECONDS = 3600  # 1 hour


class ConversationStore:
    """Thread-safe in-memory conversation store with automatic TTL cleanup."""

    def __init__(
        self,
        max_history: int = MAX_CONVERSATION_HISTORY,
        ttl_seconds: int = CONVERSATION_TTL_SECONDS,
    ):
        self._max_history = max_history
        self._ttl = ttl_seconds
        self._conversations: Dict[str, List[Dict]] = {}
        self._timestamps: Dict[str, float] = {}
        # Single coarse-grained lock — sufficient for in-memory store with
        # short critical sections. Switch to per-conversation locks only if
        # profiling shows contention.
        self._lock = threading.Lock()

    # --- public API ---

    def get_history(self, conversation_id: Optional[str]) -> List[Dict]:
        """Return a copy of the conversation messages, evicting expired entries first.

        The returned list is a shallow copy so callers can iterate / mutate
        it without holding the lock.
        """
        if not conversation_id:
            return []

        with self._lock:
            self._evict_if_expired(conversation_id)
            return list(self._conversations.get(conversation_id, []))

    def add_message(self, conversation_id: Optional[str], role: str, content: str) -> None:
        """Append a message and trim to max length.

        Thread-safe: holds the lock for the entire append+trim+timestamp
        sequence so concurrent writers cannot interleave.
        """
        if not conversation_id:
            return

        with self._lock:
            history = self._conversations.setdefault(conversation_id, [])
            history.append({"role": role, "content": content})
            self._timestamps[conversation_id] = time.time()

            if len(history) > self._max_history:
                # Trim in place — keeps the same list object referenced by
                # the dict, which is important if other code holds a ref.
                del history[:len(history) - self._max_history]

    # --- internals (callers must already hold self._lock) ---

    def _evict_if_expired(self, conversation_id: str) -> None:
        """Remove a conversation if its TTL has elapsed.

        MUST be called with ``self._lock`` held.
        """
        ts = self._timestamps.get(conversation_id)
        if ts is not None and (time.time() - ts) > self._ttl:
            self._conversations.pop(conversation_id, None)
            self._timestamps.pop(conversation_id, None)
            logger.debug(f"Evicted expired conversation: {conversation_id[:12]}...")

    def cleanup_all(self) -> int:
        """Evict all expired conversations. Returns count of evicted entries.

        Thread-safe. Safe to call from a background sweeper thread.
        """
        now = time.time()
        with self._lock:
            expired = [
                cid for cid, ts in self._timestamps.items()
                if (now - ts) > self._ttl
            ]
            for cid in expired:
                self._conversations.pop(cid, None)
                self._timestamps.pop(cid, None)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired conversations")
        return len(expired)


# Module-level singleton
conversation_store = ConversationStore()
