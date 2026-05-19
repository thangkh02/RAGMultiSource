from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MONGODB_URI: str = "mongodb://mongodb:27017"
    MONGODB_DB_NAME: str = "rag_chatbot"
    CHROMA_PERSIST_DIR: str = "backend/chroma"
    CHROMA_COLLECTION_NAME: str = "rag_chunks"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    UPLOAD_DIR: str = "backend/storage/raw"
    MARKDOWN_DIR: str = "backend/storage/markdown"
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    @property
    def backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if value.startswith("backend/"):
            return self.backend_root.parent / path
        return self.backend_root / path

    @property
    def upload_dir_path(self) -> Path:
        return self._resolve_path(self.UPLOAD_DIR)

    @property
    def markdown_dir_path(self) -> Path:
        return self._resolve_path(self.MARKDOWN_DIR)

    @property
    def chroma_persist_dir_path(self) -> Path:
        return self._resolve_path(self.CHROMA_PERSIST_DIR)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
