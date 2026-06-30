from datetime import datetime

import pytest

import main
from schemas import (
    AnalyzeResponse,
    FactCheckClaim,
    InputType,
    ModelScores,
    SignalPhrase,
    SourceCredibility,
    VerdictLabel,
)


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    with main.app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_history(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "items" in response.get_json()


def test_invalid_url(client):
    payload = {
        "type": "url",
        "content": "not-a-url",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_headline_with_mocked_pipeline(client, monkeypatch):
    async def fake_get_cached(_content):
        return None

    async def fake_set_cached(_content, _data):
        return None

    async def fake_run_analysis(req):
        return AnalyzeResponse(
            id="test-analysis-id",
            verdict=VerdictLabel.real,
            label="Likely Real",
            credibility_score=82,
            summary="This is a mocked analysis for endpoint verification.",
            model_scores=ModelScores(
                label=VerdictLabel.real,
                confidence=0.82,
                sensationalism=0.11,
                clickbait_probability=0.09,
                emotional_language_index=0.14,
            ),
            source_credibility=SourceCredibility(
                domain="example.com",
                score=80,
                bias="Center",
                tags=["Established"],
            ),
            fact_check_claims=[
                FactCheckClaim(
                    text="Sample claim",
                    status="check",
                    source="No fact-check found in database",
                    url=None,
                )
            ],
            signal_phrases=[SignalPhrase(text="according to", level="green")],
            input_type=InputType.headline,
            input_preview=req.content[:120],
            analyzed_at=datetime.utcnow(),
        )

    monkeypatch.setattr(main, "get_cached", fake_get_cached)
    monkeypatch.setattr(main, "set_cached", fake_set_cached)
    monkeypatch.setattr(main, "run_analysis", fake_run_analysis)

    payload = {
        "type": "headline",
        "content": "Scientists release a major climate report.",
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert data["verdict"] == "real"
    assert data["credibility_score"] == 82
    assert "model_scores" in data
