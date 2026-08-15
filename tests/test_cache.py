"""
Tests for app/cache.py

These tests only go through ResponseCache's public interface
(get / set / stats) rather than reaching into private helpers like
_make_key, since that's an implementation detail and not part of the
contract the rest of the app relies on.

For the TTL-expiry test, we monkeypatch time.time() inside the cache
module instead of doing a real time.sleep(), so the test stays fast
and deterministic.
"""

import time

import pytest

from app.cache import ResponseCache
import app.cache as cache_module


class TestResponseCacheBasics:
    def setup_method(self):
        self.cache = ResponseCache(ttl_seconds=300)

    def test_get_on_empty_cache_returns_none(self):
        assert self.cache.get("What is RAG?") is None

    def test_set_then_get_returns_the_same_response(self):
        self.cache.set("What is RAG?", "Retrieval Augmented Generation")
        assert self.cache.get("What is RAG?") == "Retrieval Augmented Generation"

    def test_get_for_a_different_question_returns_none(self):
        self.cache.set("What is RAG?", "Retrieval Augmented Generation")
        assert self.cache.get("What is an LLM?") is None

    def test_overwriting_a_key_returns_the_latest_value(self):
        self.cache.set("What is RAG?", "First answer")
        self.cache.set("What is RAG?", "Second answer")
        assert self.cache.get("What is RAG?") == "Second answer"


class TestResponseCacheKeyNormalization:
    """
    _make_key() lowercases and strips the query before hashing, so
    these should all be treated as the SAME cached question.
    """

    def setup_method(self):
        self.cache = ResponseCache()

    def test_matching_is_case_insensitive(self):
        self.cache.set("What is RAG?", "Retrieval Augmented Generation")
        assert self.cache.get("WHAT IS RAG?") == "Retrieval Augmented Generation"
        assert self.cache.get("what is rag?") == "Retrieval Augmented Generation"

    def test_matching_ignores_leading_and_trailing_whitespace(self):
        self.cache.set("What is RAG?", "Retrieval Augmented Generation")
        assert self.cache.get("   What is RAG?   ") == "Retrieval Augmented Generation"

    def test_matching_does_not_ignore_internal_whitespace_differences(self):
        """
        Only .lower().strip() is applied — internal spacing changes are
        NOT normalized, so these should be treated as different
        questions. This documents current behavior rather than
        asserting an idealized "smart" normalization that doesn't
        actually exist in _make_key().
        """
        self.cache.set("What is  RAG?", "Answer with double space")
        assert self.cache.get("What is RAG?") is None


class TestResponseCacheExpiry:
    def test_entry_within_ttl_is_still_returned(self, monkeypatch):
        cache = ResponseCache(ttl_seconds=300)
        cache.set("What is RAG?", "Retrieval Augmented Generation")

        # Move the clock forward, but still inside the TTL window.
        future_time = time.time() + 100
        monkeypatch.setattr(cache_module.time, "time", lambda: future_time)

        assert cache.get("What is RAG?") == "Retrieval Augmented Generation"

    def test_entry_past_ttl_returns_none(self, monkeypatch):
        cache = ResponseCache(ttl_seconds=10)
        cache.set("What is RAG?", "Retrieval Augmented Generation")

        # Move the clock forward past the TTL window.
        future_time = time.time() + 11
        monkeypatch.setattr(cache_module.time, "time", lambda: future_time)

        assert cache.get("What is RAG?") is None

    def test_expired_entry_is_evicted_from_internal_store(self, monkeypatch):
        """
        get() deletes the expired entry from self._cache once it finds
        it's stale, rather than just ignoring it. We check this via the
        public stats() property (cache_entries) instead of touching
        _cache directly.
        """
        cache = ResponseCache(ttl_seconds=10)
        cache.set("What is RAG?", "Retrieval Augmented Generation")

        future_time = time.time() + 11
        monkeypatch.setattr(cache_module.time, "time", lambda: future_time)

        cache.get("What is RAG?")  # triggers the eviction
        assert cache.stats["cache_entries"] == 0


class TestResponseCacheStats:
    def test_stats_on_a_fresh_cache(self):
        cache = ResponseCache()
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cache_entries"] == 0
        assert stats["hit_rate"] == "0.0%"

    def test_misses_are_counted(self):
        cache = ResponseCache()
        cache.get("unknown question 1")
        cache.get("unknown question 2")
        assert cache.stats["misses"] == 2
        assert cache.stats["hits"] == 0

    def test_hits_are_counted(self):
        cache = ResponseCache()
        cache.set("What is RAG?", "Retrieval Augmented Generation")
        cache.get("What is RAG?")
        cache.get("What is RAG?")
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 0

    def test_cache_entries_reflects_number_of_stored_keys(self):
        cache = ResponseCache()
        cache.set("Question 1", "Answer 1")
        cache.set("Question 2", "Answer 2")
        assert cache.stats["cache_entries"] == 2

    def test_hit_rate_is_computed_from_hits_and_misses(self):
        cache = ResponseCache()
        cache.set("What is RAG?", "Retrieval Augmented Generation")

        cache.get("What is RAG?")   # hit
        cache.get("unknown")        # miss
        cache.get("unknown2")       # miss
        cache.get("unknown3")       # miss

        # 1 hit out of 4 total lookups = 25.0%
        assert cache.stats["hit_rate"] == "25.0%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])