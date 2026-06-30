"""
Flask application entry point for VerifyAI.

Run:
    python main.py
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

from cache import close_redis, get_cached, set_cached
from config import get_settings
from database import AnalysisRecord, SessionLocal, create_tables
from schemas import AnalyzeRequest
from services.analysis_engine import run_analysis
from services.article_fetcher import ArticleFetchError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_NAME = "fake-news-detector.html"

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": settings.origins_list}},
    supports_credentials=True,
)


def _run_async(awaitable):
    """Run async code safely from Flask sync request handlers."""
    try:
        return asyncio.run(awaitable)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()


def _serialize_record(record: AnalysisRecord) -> dict:
    return {
        "id": record.id,
        "verdict": record.verdict,
        "label": record.label,
        "credibility_score": record.credibility_score,
        "summary": record.summary or "",
        "model_scores": {
            "label": record.nlp_label or "suspicious",
            "confidence": record.nlp_confidence or 0.0,
            "sensationalism": record.sensationalism or 0.0,
            "clickbait_probability": record.clickbait_probability or 0.0,
            "emotional_language_index": record.emotional_index or 0.0,
        },
        "source_credibility": {
            "domain": record.source_domain or "unknown",
            "score": record.source_score or 50,
            "bias": record.source_bias or "Unknown",
            "tags": record.source_tags or [],
        },
        "fact_check_claims": record.fact_check_claims or [],
        "signal_phrases": record.signal_phrases or [],
        "input_type": record.input_type,
        "input_preview": record.input_preview or "",
        "analyzed_at": record.analyzed_at.isoformat() if record.analyzed_at else None,
    }


def _startup() -> None:
    logger.info("Starting %s [%s]", settings.app_name, settings.app_env)
    try:
        create_tables()
        logger.info("Database tables ready.")
    except Exception as exc:
        logger.warning("Database unavailable at startup (%s).", exc)


@atexit.register
def _shutdown() -> None:
    try:
        _run_async(close_redis())
        logger.info("Redis connection closed.")
    except Exception as exc:
        logger.warning("Redis close warning: %s", exc)


_startup()


@app.get("/")
def root():
    return send_from_directory(PROJECT_ROOT, FRONTEND_NAME)


@app.get(f"/{FRONTEND_NAME}")
def frontend_file():
    return send_from_directory(PROJECT_ROOT, FRONTEND_NAME)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "app": settings.app_name, "env": settings.app_env})


@app.get("/api/docs")
def api_docs():
    return jsonify(
        {
            "name": "VerifyAI Flask API",
            "version": "1.0.0",
            "endpoints": [
                {"method": "GET", "path": "/"},
                {"method": "GET", "path": "/health"},
                {"method": "POST", "path": "/api/analyze"},
                {"method": "GET", "path": "/api/history?page=1&per_page=20"},
                {"method": "GET", "path": "/api/analysis/<analysis_id>"},
            ],
        }
    )


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    try:
        req = AnalyzeRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"detail": "Invalid request body.", "errors": json.loads(exc.json())}), 422

    cached = _run_async(get_cached(req.content))
    if cached:
        logger.info("Returning cached result.")
        return jsonify(cached)

    try:
        result = _run_async(run_analysis(req))
    except ArticleFetchError as exc:
        return jsonify({"detail": f"Could not fetch article: {exc}"}), 422
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception:
        logger.exception("Unexpected error during analysis")
        return (
            jsonify({"detail": "Analysis failed due to an internal error. Please try again."}),
            500,
        )

    db = SessionLocal()
    try:
        record = AnalysisRecord(
            id=result.id,
            verdict=result.verdict.value,
            label=result.label,
            credibility_score=result.credibility_score,
            input_type=result.input_type.value,
            input_preview=result.input_preview,
            full_input=req.content,
            nlp_label=result.model_scores.label.value,
            nlp_confidence=result.model_scores.confidence,
            sensationalism=result.model_scores.sensationalism,
            clickbait_probability=result.model_scores.clickbait_probability,
            emotional_index=result.model_scores.emotional_language_index,
            source_domain=result.source_credibility.domain,
            source_score=result.source_credibility.score,
            source_bias=result.source_credibility.bias,
            fact_check_claims=[c.model_dump(mode="json") for c in result.fact_check_claims],
            signal_phrases=[s.model_dump(mode="json") for s in result.signal_phrases],
            source_tags=result.source_credibility.tags,
            summary=result.summary,
            analyzed_at=result.analyzed_at,
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Could not persist analysis to DB: %s", exc)
    finally:
        db.close()

    response_payload = result.model_dump(mode="json")
    _run_async(set_cached(req.content, response_payload))
    return jsonify(response_payload)


@app.get("/api/history")
def get_history():
    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = max(min(int(request.args.get("per_page", 20)), 100), 1)
    except ValueError:
        return jsonify({"detail": "page and per_page must be integers."}), 400

    db = SessionLocal()
    try:
        offset = (page - 1) * per_page
        total = db.query(AnalysisRecord).count()
        rows = (
            db.query(AnalysisRecord)
            .order_by(AnalysisRecord.analyzed_at.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )
        items = [
            {
                "id": row.id,
                "verdict": row.verdict,
                "label": row.label,
                "credibility_score": row.credibility_score,
                "input_preview": row.input_preview or "",
                "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
            }
            for row in rows
        ]
        return jsonify({"items": items, "total": total})
    except Exception as exc:
        logger.warning("History DB error: %s", exc)
        return jsonify({"items": [], "total": 0})
    finally:
        db.close()


@app.get("/api/analysis/<analysis_id>")
def get_analysis(analysis_id: str):
    db = SessionLocal()
    try:
        record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        if not record:
            return jsonify({"detail": "Analysis not found."}), 404
        return jsonify(_serialize_record(record))
    except Exception as exc:
        logger.warning("Could not fetch analysis %s: %s", analysis_id, exc)
        return jsonify({"detail": "Could not load analysis."}), 500
    finally:
        db.close()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=(settings.app_env == "development"),
    )
