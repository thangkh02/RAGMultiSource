from app.core.config import settings
from app.rag.generation.openai_llm import OpenAILLMService


def test_openrouter_provider_initializes_chat_model(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(settings, "OPENROUTER_SITE_URL", "https://example.test")
    monkeypatch.setattr(settings, "OPENROUTER_APP_NAME", "RAG Chatbot Test")

    service = OpenAILLMService()

    assert service.llm is not None
    assert service.chain is not None
    assert service.llm.model_name == "openai/gpt-4o-mini"
    assert str(service.llm.openai_api_base).rstrip("/") == "https://openrouter.ai/api/v1"
