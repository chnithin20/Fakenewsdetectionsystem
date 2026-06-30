"""
analysis_engine.py
───────────────────
Orchestrates the full analysis pipeline:
  1. Fetch article text (if URL)
  2. Run NLP classifier
  3. Query Fact Check API (async)
  4. Score source credibility
  5. Compute weighted final verdict
  6. Return AnalyzeResponse
"""

import uuid
import asyncio
import logging
from datetime import datetime
from urllib.parse import urlparse

from config import get_settings
from schemas import (
    AnalyzeRequest, AnalyzeResponse, InputType, VerdictLabel,
    ModelScores, FactCheckClaim, SourceCredibility, SignalPhrase,
)
from services.article_fetcher   import fetch_article, ArticleFetchError
from services.nlp_classifier    import classify_text, extract_signal_phrases
from services.fact_checker      import check_claims
from services.source_credibility import score_domain

logger   = logging.getLogger(__name__)
settings = get_settings()


async def run_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Main entry point.  Accepts an AnalyzeRequest and returns a full AnalyzeResponse.
    Raises ValueError for bad inputs and ArticleFetchError for unresolvable URLs.
    """
    analysis_id = str(uuid.uuid4())
    domain = "unknown"
    text   = ""
    title  = ""

    # ── Step 1: Resolve input to plain text ───────────────────────────────
    if req.type == InputType.url:
        article = await fetch_article(req.content)
        text    = f"{article.title}\n\n{article.text}"
        title   = article.title
        domain  = article.domain

    elif req.type == InputType.text:
        text   = req.content
        domain = _guess_domain_from_text(text)   # may find a URL mentioned in body

    elif req.type == InputType.headline:
        text   = req.content
        # Headlines are short → boost sensationalism weight, skip long fact-check

    if not text.strip():
        raise ValueError("Could not extract usable text for analysis.")

    # ── Step 2: Run NLP + source score concurrently ───────────────────────
    loop = asyncio.get_running_loop()

    # NLP classifier is CPU-bound → run in a thread pool
    nlp_future  = loop.run_in_executor(None, classify_text, text)
    cred_result = score_domain(domain)

    # Fact-check is I/O-bound → run directly async
    fact_future = check_claims(text)

    nlp_result, fact_results = await asyncio.gather(nlp_future, fact_future)

    signal_phrases = extract_signal_phrases(text)

    # ── Step 3: Compute weighted credibility score (0–100) ────────────────
    #
    # NLP model contributes: based on real_probability
    # Fact-check contributes: ratio of verified vs debunked claims
    # Source credibility contributes: normalized domain score
    #
    nlp_score    = int(nlp_result.real_probability * 100)
    fact_score   = _compute_fact_check_score(fact_results)
    source_score = cred_result.score

    final_score = int(
        nlp_score    * settings.weight_nlp_model          +
        fact_score   * settings.weight_fact_check         +
        source_score * settings.weight_source_credibility
    )
    final_score = max(0, min(100, final_score))

    # ── Step 4: Verdict label ─────────────────────────────────────────────
    if final_score >= 65:
        verdict = VerdictLabel.real
        label   = "Likely Real"
        summary = _build_summary("real",   final_score, nlp_result, cred_result)
    elif final_score >= 35:
        verdict = VerdictLabel.suspicious
        label   = "Suspicious"
        summary = _build_summary("suspicious", final_score, nlp_result, cred_result)
    else:
        verdict = VerdictLabel.fake
        label   = "Likely Fake"
        summary = _build_summary("fake",   final_score, nlp_result, cred_result)

    # ── Step 5: Assemble response ─────────────────────────────────────────
    return AnalyzeResponse(
        id=analysis_id,
        verdict=verdict,
        label=label,
        credibility_score=final_score,
        summary=summary,

        model_scores=ModelScores(
            label=VerdictLabel(nlp_result.label),
            confidence=nlp_result.confidence,
            sensationalism=nlp_result.sensationalism,
            clickbait_probability=nlp_result.clickbait_probability,
            emotional_language_index=nlp_result.emotional_language_index,
        ),

        source_credibility=SourceCredibility(
            domain=cred_result.domain,
            score=cred_result.score,
            bias=cred_result.bias,
            tags=cred_result.tags,
        ),

        fact_check_claims=[
            FactCheckClaim(
                text=fc.text,
                status=fc.status,
                source=fc.source,
                url=fc.url,
            )
            for fc in fact_results
        ],

        signal_phrases=[
            SignalPhrase(text=sp["text"], level=sp["level"])
            for sp in signal_phrases
        ],

        input_type=req.type,
        input_preview=req.content[:120],
        analyzed_at=datetime.utcnow(),
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _compute_fact_check_score(fact_results) -> int:
    """
    Converts a list of FactCheckResult objects into a 0–100 score.
    verified = +100 pts, debunked = +0 pts, check = +50 pts.
    Averages across all claims.
    """
    if not fact_results:
        return 50     # No data → neutral

    STATUS_SCORES = {"verified": 100, "check": 50, "debunked": 0}
    scores = [STATUS_SCORES.get(fc.status, 50) for fc in fact_results]
    return int(sum(scores) / len(scores))


def _guess_domain_from_text(text: str) -> str:
    """Try to extract a URL mentioned in the pasted text body."""
    import re
    urls = re.findall(r"https?://[^\s]+", text)
    if urls:
        try:
            return urlparse(urls[0]).netloc.replace("www.", "")
        except Exception:
            pass
    return "unknown"


def _build_summary(verdict: str, score: int, nlp, cred) -> str:
    if verdict == "real":
        return (
            f"This content scores {score}/100 on our credibility scale. "
            f"The NLP model classifies it as {int(nlp.real_probability*100)}% real, "
            f"with low sensationalism ({int(nlp.sensationalism*100)}%) and "
            f"low emotional language ({int(nlp.emotional_language_index*100)}%). "
            f"The source '{cred.domain}' has a credibility rating of {cred.score}/100. "
            "No major red flags were detected."
        )
    elif verdict == "suspicious":
        return (
            f"This content scores {score}/100 — mixed signals detected. "
            f"The model is uncertain ({int(nlp.confidence*100)}% confidence). "
            f"Sensationalism score is {int(nlp.sensationalism*100)}% and "
            f"source credibility is {cred.score}/100 ({cred.bias}). "
            "Cross-check with additional sources before sharing."
        )
    else:
        return (
            f"This content scores only {score}/100 on credibility. "
            f"The NLP model classifies it as {int(nlp.fake_probability*100)}% fake. "
            f"High sensationalism ({int(nlp.sensationalism*100)}%) and "
            f"clickbait patterns ({int(nlp.clickbait_probability*100)}%) were detected. "
            f"Source credibility: {cred.score}/100. This content exhibits strong misinformation signals."
        )
