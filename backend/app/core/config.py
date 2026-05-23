import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore")

    MONGODB_URI: str = "mongodb://mongodb:27017"
    MONGODB_DB_NAME: str = "rag_chatbot"
    CHROMA_PERSIST_DIR: str = "backend/chroma"
    CHROMA_COLLECTION_NAME: str = "rag_chunks"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_PROVIDER: str = "openai"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_QUERY_REWRITE_MODEL: str = "google/gemini-3.1-flash-lite-preview"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = ""
    OPENROUTER_APP_NAME: str = "RAG Chatbot"
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "RAGMultiDocs"
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

    @property
    def vector_collection_name(self) -> str:
        if self.OPENAI_API_KEY:
            return f"{self.CHROMA_COLLECTION_NAME}_openai"
        return self.CHROMA_COLLECTION_NAME


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def configure_langsmith() -> None:
    os.environ.setdefault("LANGSMITH_TRACING", settings.LANGSMITH_TRACING)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)

    # Compatibility with older LangChain releases that still read LANGCHAIN_*.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)


configure_langsmith()
