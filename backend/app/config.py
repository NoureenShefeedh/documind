from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://documind:documind@localhost:5432/documind"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 12

    embedding_provider: str = "hashing"  # hashing | sbert
    embedding_dim: int = 384

    llm_provider: str = "extractive"  # extractive | anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"

    chunk_words: int = 180
    chunk_overlap: int = 40
    top_k: int = 4
    max_upload_mb: int = 10

    cors_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
