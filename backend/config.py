from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # App
    app_name: str = "VerifyAI"
    app_env: str = "development"
    secret_key: str = "change-me"
    allowed_origins: str = "http://localhost:3000,http://localhost:5500,null"

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/verifyai_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # APIs
    google_fact_check_api_key: str = ""
    news_api_key: str = ""

    # HuggingFace Model
    hf_model_name: str = "hamzab/roberta-fake-news-classification"
    hf_token: str = ""
    hf_model_cache_dir: str = "./model_cache"

    # Analysis Weights
    weight_nlp_model: float = 0.50
    weight_fact_check: float = 0.30
    weight_source_credibility: float = 0.20

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
