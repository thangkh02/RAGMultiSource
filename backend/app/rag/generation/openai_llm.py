from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.generation.prompts import RAG_SYSTEM_PROMPT


class OpenAILLMService:
    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "Question: {question}\n\nAnswer style: {answer_style}\n\nContext:\n{context}"),
            ]
        )
        self.llm = None
        self.chain = None
        provider = settings.LLM_PROVIDER.strip().lower()
        if provider == "openrouter" and settings.OPENROUTER_API_KEY:
            default_headers = {}
            if settings.OPENROUTER_SITE_URL:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            if settings.OPENROUTER_APP_NAME:
                default_headers["X-Title"] = settings.OPENROUTER_APP_NAME
            self.llm = ChatOpenAI(
                model=settings.OPENROUTER_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=0,
                default_headers=default_headers or None,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
        elif settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_context_text(self, contexts: list[dict]) -> str:
        if not contexts:
            return ""
        return "\n\n".join(
            [
                f"[{idx + 1}] {item.get('content', '')}\nMETADATA: {item.get('metadata', {})}\nSIMILARITY: {item.get('similarity')}"
                for idx, item in enumerate(contexts)
            ]
        )

    def generate_answer(self, question: str, contexts: list[dict], answer_style: str = "short_answer") -> str:
        if not contexts:
            return "Không tìm thấy thông tin này trong tài liệu phù hợp."
        context_text = self._build_context_text(contexts)
        if self.chain is None:
            return f"[LangChain] OPENAI_API_KEY chưa được cấu hình.\n\n{context_text[:1200]}"
        result = self.chain.invoke({"question": question, "answer_style": answer_style, "context": context_text})
        return result.strip() if isinstance(result, str) else str(result)
