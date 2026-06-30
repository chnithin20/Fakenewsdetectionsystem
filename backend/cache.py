"""
cache.py
─────────
Redis-based cache for analysis results.
Key: SHA-256 hash of the input content
TTL: configurable via CACHE_TTL_SECONDS (default 1 hour)

Why cache?
  - RoBERTa inference takes ~200-500ms; caching saves GPU/CPU for repeated URLs
  - Google Fact Check API has daily quota limits
  - Same article URL submitted by multiple users → single fetch + analysis
"""

import json
import hashlib
import logging
from redis import asyncio as aioredis
from config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

_redis_client = None


async def get_redis():
    """Lazy singleton — returns an async Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis_client.ping()
            logger.info("Redis connection established.")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Caching disabled.")
            _redis_client = None
    return _redis_client


def _cache_key(content: str) -> str:
    """Deterministic cache key from input content."""
    digest = hashlib.sha256(content.encode()).hexdigest()
    return f"verifyai:analysis:{digest}"


async def get_cached(content: str) -> dict | None:
    """Return cached analysis dict or None if not cached."""
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(_cache_key(content))
        if raw:
            logger.debug("Cache HIT")
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None


async def set_cached(content: str, data: dict) -> None:
    """Persist an analysis result dict to Redis with TTL."""
    client = await get_redis()
    if client is None:
        return
    try:
        await client.setex(
            _cache_key(content),
            settings.cache_ttl_seconds,
            json.dumps(data, default=str),   # default=str handles datetime
        )
        logger.debug("Cache SET")
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


async def close_redis():
    """Call on app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
