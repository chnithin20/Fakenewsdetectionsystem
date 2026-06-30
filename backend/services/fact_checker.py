"""
fact_checker.py
────────────────
Queries the Google Fact Check Tools API for claims found in the article.

Steps:
  1. Extract key claims from the text using simple NLP heuristics
  2. Query Google Fact Check API for each claim
  3. Return structured FactCheckClaim results

API Docs: https://developers.google.com/fact-check/tools/api/reference/rest
Free tier: 1,000 queries/day
"""

import re
import logging
import httpx
from dataclasses import dataclass
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Phrases that often introduce a factual claim
CLAIM_LEAD_PATTERNS = [
    r"(?:according to|reports?|claimed?|said|stated|alleged|insists?)[^.]{10,}[.]",
    r"(?:study|research|survey|report)[^.]{10,}(?:found|shows?|reveals?|confirms?)[^.]+[.]",
    r"\d+[\s%][^.]{5,}[.]",  # sentences with numbers / stats
]


@dataclass
class FactCheckResult:
    text: str
    status: str       # "verified" | "debunked" | "check"
    source: str       # e.g. "Verified · Reuters Fact Check"
    url: str = ""


async def check_claims(text: str) -> list[FactCheckResult]:
    """
    Extract candidate claims from text and look them up via the API.
    Returns up to 5 FactCheckResult objects.
    """
    claims = _extract_candidate_claims(text)
    if not claims:
        return []

    results: list[FactCheckResult] = []
    # Limit API calls to avoid quota burn
    for claim in claims[:5]:
        result = await _query_fact_check_api(claim)
        if result:
            results.append(result)

    return results[:5]


async def _query_fact_check_api(query: str) -> FactCheckResult | None:
    """
    Hit the Google Fact Check API for a single claim query.
    Returns None if no results found or API key is not set.
    """
    if not settings.google_fact_check_api_key:
        logger.warning("GOOGLE_FACT_CHECK_API_KEY not set — skipping fact check.")
        return None

    params = {
        "query": query[:200],            # API max query length
        "key": settings.google_fact_check_api_key,
        "languageCode": "en",
        "pageSize": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(FACT_CHECK_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        claims_list = data.get("claims", [])
        if not claims_list:
            # No matching fact-check found
            return FactCheckResult(
                text=_truncate(query, 120),
                status="check",
                source="No fact-check found in database",
            )

        first = claims_list[0]
        review = first.get("claimReview", [{}])[0]
        rating_str = review.get("textualRating", "").lower()
        publisher_name = review.get("publisher", {}).get("name", "Unknown")
        review_url = review.get("url", "")

        # Map common rating strings to our 3-tier status
        status = _map_rating(rating_str)
        source_label = _make_source_label(status, publisher_name)

        return FactCheckResult(
            text=_truncate(first.get("text", query), 140),
            status=status,
            source=source_label,
            url=review_url,
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Fact Check API HTTP error: {e}")
    except Exception as e:
        logger.error(f"Fact Check API error: {e}")

    return None


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_candidate_claims(text: str) -> list[str]:
    """Pull sentences likely to contain checkable claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        for pattern in CLAIM_LEAD_PATTERNS:
            if re.search(pattern, sent, re.IGNORECASE):
                candidates.append(sent)
                break

    # Fallback: take first 5 sentences if no claims detected
    if not candidates:
        candidates = [s.strip() for s in sentences if len(s.strip()) > 40][:5]

    return candidates[:5]


def _map_rating(rating: str) -> str:
    """
    Map free-text ratings from fact-checkers to our 3 statuses.
    Rating examples: "false", "pants on fire", "mostly true", "half true"
    """
    false_keywords = {
        "false", "fake", "incorrect", "pants", "debunked",
        "misleading", "wrong", "baseless", "fabricated", "lie",
    }
    true_keywords = {
        "true", "correct", "accurate", "verified", "confirmed",
        "mostly true", "largely true",
    }

    r = rating.lower()
    if any(k in r for k in false_keywords):
        return "debunked"
    if any(k in r for k in true_keywords):
        return "verified"
    return "check"


def _make_source_label(status: str, publisher: str) -> str:
    if status == "verified":
        return f"Verified · {publisher}"
    if status == "debunked":
        return f"Debunked · {publisher}"
    return f"Mixed rating · {publisher}"


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
