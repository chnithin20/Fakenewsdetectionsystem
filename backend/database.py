from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from config import get_settings

settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent


def _resolve_database_url(url: str) -> str:
    """
    Resolve relative sqlite paths against the backend directory, so launching
    from different working directories always points to one DB file.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url

    raw_path = url[len(prefix):]
    if raw_path in ("", ":memory:"):
        return url

    path = Path(raw_path)
    if raw_path.startswith("./"):
        path = BASE_DIR / raw_path[2:]
    elif not path.is_absolute() and ":" not in raw_path:
        path = BASE_DIR / raw_path

    return f"{prefix}{path.resolve().as_posix()}"

# ── Engine & Session ────────────────────────────────────────────────────────
resolved_database_url = _resolve_database_url(settings.database_url)
is_sqlite = resolved_database_url.startswith("sqlite")

engine = create_engine(
    resolved_database_url,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    pool_pre_ping=not is_sqlite,
    pool_size=1 if is_sqlite else 10,
    max_overflow=0 if is_sqlite else 20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────

class AnalysisRecord(Base):
    """Stores every completed fake-news analysis."""
    __tablename__ = "analysis_records"

    id                    = Column(String(36), primary_key=True)   # UUID
    verdict               = Column(String(20), nullable=False)      # real/fake/suspicious
    label                 = Column(String(50), nullable=False)
    credibility_score     = Column(Integer,    nullable=False)

    input_type            = Column(String(20), nullable=False)
    input_preview         = Column(String(200))
    full_input            = Column(Text)                            # stored for re-analysis

    # NLP scores
    nlp_label             = Column(String(10))
    nlp_confidence        = Column(Float)
    sensationalism        = Column(Float)
    clickbait_probability = Column(Float)
    emotional_index       = Column(Float)

    # Source
    source_domain         = Column(String(255))
    source_score          = Column(Integer)
    source_bias           = Column(String(50))

    # JSON blobs for rich data
    fact_check_claims     = Column(JSON)    # list of FactCheckClaim dicts
    signal_phrases        = Column(JSON)    # list of SignalPhrase dicts
    source_tags           = Column(JSON)    # list of str

    summary               = Column(Text)
    analyzed_at           = Column(DateTime, default=datetime.utcnow, server_default=func.now())

    def __repr__(self):
        return f"<Analysis {self.id[:8]} verdict={self.verdict} score={self.credibility_score}>"


# ── DB helpers ──────────────────────────────────────────────────────────────

def get_db():
    """Yield a DB session and close it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Call once at startup to create all tables."""
    Base.metadata.create_all(bind=engine)
