"""
source_credibility.py
──────────────────────
Scores the credibility of a news source domain.

Approach:
  - Maintain a curated local database of known domains with credibility scores
  - Fall back to heuristics for unknown domains (age, TLD, HTTPS, etc.)
  - In production: integrate Media Bias / Fact Check or NewsGuard APIs
"""

import re
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class CredibilityResult:
    domain: str
    score: int              # 0 – 100
    bias: str               # "Center", "Lean Left", "Lean Right", "Extreme Right" etc.
    tags: list[str] = field(default_factory=list)


# ── Curated Domain Database ────────────────────────────────────────────────
# Format: domain → (score, bias, [tags])
# Sources: Media Bias/Fact Check, AllSides, NewsGuard
# Expand this dict as needed — or load from a CSV/DB in production.

DOMAIN_DB: dict[str, tuple[int, str, list[str]]] = {
    # ── High credibility ──
    "reuters.com":          (95, "Center",     ["International", "Established", "Fact-Checked"]),
    "apnews.com":           (94, "Center",     ["Wire Service", "Established", "Fact-Checked"]),
    "bbc.com":              (91, "Center",     ["Public Broadcaster", "International", "Established"]),
    "bbc.co.uk":            (91, "Center",     ["Public Broadcaster", "International"]),
    "nytimes.com":          (88, "Lean Left",  ["Established", "Pulitzer Winner", "Paywalled"]),
    "washingtonpost.com":   (87, "Lean Left",  ["Established", "Pulitzer Winner", "Paywalled"]),
    "theguardian.com":      (86, "Lean Left",  ["International", "Established"]),
    "wsj.com":              (88, "Lean Right", ["Established", "Financial News", "Paywalled"]),
    "economist.com":        (90, "Center",     ["International", "Established", "Paywalled"]),
    "ft.com":               (89, "Center",     ["Financial News", "Established", "International"]),
    "npr.org":              (89, "Lean Left",  ["Public Broadcaster", "Established"]),
    "pbs.org":              (88, "Center",     ["Public Broadcaster", "Established"]),
    "bloomberg.com":        (87, "Center",     ["Financial News", "Established"]),
    "nature.com":           (96, "Center",     ["Scientific", "Peer-Reviewed", "Academic"]),
    "science.org":          (96, "Center",     ["Scientific", "Peer-Reviewed"]),
    "who.int":              (95, "Center",     ["Official", "International", "Health"]),
    "cdc.gov":              (94, "Center",     ["Official", "Government", "Health"]),
    "snopes.com":           (85, "Center",     ["Fact-Checker", "Established"]),
    "politifact.com":       (84, "Center",     ["Fact-Checker", "Established"]),
    "factcheck.org":        (85, "Center",     ["Fact-Checker", "Non-Profit"]),

    # ── Medium credibility ──
    "foxnews.com":          (58, "Lean Right", ["Established", "Opinion Heavy", "Partisan"]),
    "cnn.com":              (62, "Lean Left",  ["Established", "Opinion Heavy"]),
    "msnbc.com":            (55, "Lean Left",  ["Opinion Heavy", "Partisan"]),
    "nypost.com":           (52, "Lean Right", ["Tabloid", "Partisan"]),
    "huffpost.com":         (57, "Left",       ["Opinion Heavy", "Established"]),
    "breitbart.com":        (28, "Extreme Right", ["Partisan", "Misleading History", "Sensationalist"]),
    "dailymail.co.uk":      (45, "Lean Right", ["Tabloid", "Sensationalist", "UK"]),
    "thesun.co.uk":         (40, "Lean Right", ["Tabloid", "UK"]),
    "theblaze.com":         (30, "Extreme Right", ["Partisan", "Sensationalist"]),
    "occupydemocrats.com":  (25, "Extreme Left", ["Partisan", "Sensationalist", "Clickbait"]),
    "naturalnews.com":      (8,  "Extreme Right", ["Conspiracy", "Medical Misinformation", "Debunked Sources"]),
    "infowars.com":         (5,  "Extreme Right", ["Conspiracy", "Banned", "Misinformation"]),
    "theonion.com":         (40, "Satire",     ["Satire", "Not News", "Clearly Labeled"]),
    "babylonbee.com":       (35, "Satire",     ["Satire", "Conservative", "Not News"]),
}


# ── TLD credibility penalties/bonuses ─────────────────────────────────────
TLD_SCORES = {
    ".gov":    +15, ".edu":  +12, ".org": +5,
    ".com":    0,   ".net":  -2,  ".co":  -5,
    ".info":   -8,  ".biz": -10,  ".xyz": -15,
    ".click":  -20, ".online": -12, ".site": -10,
}


# ── Public API ─────────────────────────────────────────────────────────────

def score_domain(domain_or_url: str) -> CredibilityResult:
    """
    Return a CredibilityResult for the given domain or URL.
    Checks the curated DB first; falls back to heuristics.
    """
    domain = _extract_domain(domain_or_url)

    # Direct match
    if domain in DOMAIN_DB:
        score, bias, tags = DOMAIN_DB[domain]
        return CredibilityResult(domain=domain, score=score, bias=bias, tags=tags)

    # Partial match (e.g. subdomain.reuters.com → reuters.com)
    for known_domain, (score, bias, tags) in DOMAIN_DB.items():
        if domain.endswith("." + known_domain) or domain == known_domain:
            return CredibilityResult(domain=domain, score=score, bias=bias, tags=list(tags))

    # Heuristics for unknown domains
    return _heuristic_score(domain)


# ── Private helpers ────────────────────────────────────────────────────────

def _extract_domain(value: str) -> str:
    """Extract clean domain from URL or return as-is if already a domain."""
    value = value.strip()
    if value.startswith("http"):
        try:
            return urlparse(value).netloc.replace("www.", "").lower()
        except Exception:
            pass
    return value.replace("www.", "").lower()


def _heuristic_score(domain: str) -> CredibilityResult:
    """
    Score an unknown domain using structural heuristics:
    - TLD reputation
    - Suspicious keywords in domain name
    - Domain length / complexity
    """
    base_score = 45
    tags: list[str] = ["Unknown", "No Verified Record"]
    bias = "Unknown"

    # TLD bonus/penalty
    for tld, delta in TLD_SCORES.items():
        if domain.endswith(tld):
            base_score += delta
            break

    # Suspicious keywords in the domain name itself
    SUSPICIOUS_KEYWORDS = [
        "truth", "real", "freedom", "patriot", "expose",
        "alert", "news24", "infowars", "wakeup", "thegreat",
        "alternative", "underground", "rebellion",
    ]
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in domain:
            base_score -= 12
            tags.append("Suspicious Domain Name")
            break

    # Very long domains tend to be less reputable
    if len(domain) > 30:
        base_score -= 5
        tags.append("Long Domain")

    # Numeric-heavy domains
    if re.search(r"\d{3,}", domain):
        base_score -= 5

    # .gov / .edu are almost certainly trustworthy
    if domain.endswith(".gov"):
        bias = "Center"
        tags = ["Official", "Government"]
    elif domain.endswith(".edu"):
        bias = "Center"
        tags = ["Academic", "Educational"]

    score = max(0, min(100, base_score))
    return CredibilityResult(domain=domain, score=score, bias=bias, tags=tags)
