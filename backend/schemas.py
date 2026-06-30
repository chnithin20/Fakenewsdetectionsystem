from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class InputType(str, Enum):
    url      = "url"
    text     = "text"
    headline = "headline"

class VerdictLabel(str, Enum):
    real       = "real"
    suspicious = "suspicious"
    fake       = "fake"

class ClaimStatus(str, Enum):
    verified  = "verified"
    debunked  = "debunked"
    check     = "check"       # unverified / needs review


# ── Request Schemas ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    type: InputType
    content: str              # URL string, article text, or headline

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError("Content must not be empty.")
        
        # Accessing other fields in Pydantic v2
        data_type = info.data.get("type")
        if data_type == InputType.url:
            if not v.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            if "." not in v:
                raise ValueError("Invalid URL format.")
        
        return v.strip()


# ── Sub-models (used in AnalyzeResponse) ──────────────────────────────────

class ModelScores(BaseModel):
    label: VerdictLabel         # "real" | "fake"
    confidence: float           # 0.0 – 1.0
    sensationalism: float       # 0.0 – 1.0
    clickbait_probability: float
    emotional_language_index: float


class FactCheckClaim(BaseModel):
    text: str
    status: ClaimStatus
    source: str                 # e.g. "Verified · Reuters Fact Check"
    url: Optional[str] = None


class SourceCredibility(BaseModel):
    domain: str
    score: int                  # 0 – 100
    bias: str                   # e.g. "Center", "Lean Right"
    tags: list[str]             # e.g. ["Established", "Fact-Checked"]


class SignalPhrase(BaseModel):
    text: str
    level: Literal["red", "orange", "green"]


# ── Main Response Schema ───────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str                         # UUID for this analysis
    verdict: VerdictLabel
    label: str                      # Human-readable: "Likely Real" etc.
    credibility_score: int          # Final weighted score 0–100
    summary: str                    # One-paragraph explanation

    model_scores: ModelScores
    source_credibility: SourceCredibility
    fact_check_claims: list[FactCheckClaim]
    signal_phrases: list[SignalPhrase]

    input_type: InputType
    input_preview: str              # First 120 chars of input
    analyzed_at: datetime


# ── History schemas ────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: str
    verdict: VerdictLabel
    label: str
    credibility_score: int
    input_preview: str
    analyzed_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
