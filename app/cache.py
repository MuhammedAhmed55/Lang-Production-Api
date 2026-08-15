"""
===============================================================
RESPONSE CACHE
===============================================================

This file creates a temporary memory/cache for AI responses.

WHY DO WE NEED CACHE?

Imagine the user asks:

    "What is RAG?"

The AI generates an answer.

If the user asks the exact same question again a few seconds
later, we don't want to call the AI again.

Instead:

    First request:
        User → AI → Answer → Save in Cache

    Second request:
        User → Cache → Answer ⚡

This gives us:

    ✅ Faster responses
    ✅ Fewer AI/API calls
    ✅ Lower cost
    ✅ Better application performance


HOW LONG IS THE ANSWER STORED?

By default:

    300 seconds = 5 minutes

After 5 minutes, the cached answer expires and the AI will
generate a fresh answer.

Simple flow:

    User Question
          ↓
      Check Cache
          ↓
      ┌───┴────┐
      ↓        ↓
    FOUND   NOT FOUND
      ↓        ↓
   Check TTL  AI Model
      ↓        ↓
   < 5 min?   New Answer
      ↓        ↓
     YES     Save Cache
      ↓        ↓
   Return    Return
   Answer    Answer
"""

import hashlib
import time
from typing import Optional


class ResponseCache:
    """Temporarily stores AI responses."""

    def __init__(self, ttl_seconds: int = 300):
        # How long a response stays valid (default: 5 minutes)
        self.ttl = ttl_seconds

        # Stores cached responses
        self._cache: dict[str, dict] = {}

        # Cache performance counters
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str) -> str:
        """
        Creates a unique key from the question.

        Hashing makes it easy to find the cached response.
        """

        # Normalize question before creating the key
        normalized = query.lower().strip()

        # Convert question into a unique SHA-256 key
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """
        Get a cached response (saved answer) if it exists and hasn't expired.
        """

        key = self._make_key(query)

        # Check if this question exists in cache
        if key in self._cache:
            entry = self._cache[key]

            # Use cached response if it is still fresh
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                return entry["response"]

            # Remove expired response
            del self._cache[key]

        # No valid cached response found
        self.misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Save a new AI response in the cache."""

        key = self._make_key(query)

        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

    @property
    def stats(self) -> dict:
        """Return cache performance statistics."""

        total = self.hits + self.misses

        # Calculate percentage of requests served from cache
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "cache_entries": len(self._cache),
            "hit_rate": f"{hit_rate:.1%}",
        }

"""
_make_key() → Makes unique ID
get()       → Gets saved answer
set()       → Saves new answer
TTL         → How long answer stays
stats       → Shows cache performance

"""