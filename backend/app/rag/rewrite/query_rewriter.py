from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.utils.text_utils import count_tokens_rough


REWRITE_INTENT_FOLLOW_UP = "follow_up"


@dataclass
class QueryRewrite:
    original_question: str
    rewritten_question: str
    was_rewritten: bool
    reason: str
    stage: str = "post_intent"
    used_llm: bool = False

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class QueryRewriter:
    max_history_tokens = 1200

    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Bạn là bộ rewrite query cho hệ thống RAG tiếng Việt.\n"
                        "Nhiệm vụ: biến câu hỏi nối tiếp thành một câu hỏi độc lập để đưa vào retrieval.\n"
                        "Chỉ rewrite, không trả lời câu hỏi.\n"
                        "Giữ nguyên tên file, tên thủ tục, số liệu, ngày tháng nếu có.\n"
                        "Không thêm thông tin không có trong lịch sử.\n"
                        "Nếu câu hỏi đã đủ rõ hoặc không có lịch sử, trả lại đúng câu hỏi gốc.\n"
                        "Chỉ xuất một câu hỏi duy nhất, không giải thích."
                    ),
                ),
                (
                    "human",
                    (
                        "Lịch sử gần nhất:\n"
                        "user: File ho_so_alpha.pdf nói gì về lệ phí?\n"
                        "assistant: Theo file ho_so_alpha.pdf, lệ phí nội bộ là 75.000 đồng.\n\n"
                        "Câu hỏi mới:\n"
                        "còn thời hạn thì sao?\n\n"
                        "Câu hỏi độc lập:"
                    ),
                ),
                ("ai", "Trong file ho_so_alpha.pdf, thời hạn xử lý là bao lâu?"),
                (
                    "human",
                    (
                        "Lịch sử gần nhất:\n"
                        "user: Thủ tục đăng ký thành lập doanh nghiệp tư nhân cần hồ sơ gì?\n"
                        "assistant: Hồ sơ gồm giấy đề nghị đăng ký doanh nghiệp và giấy tờ pháp lý của cá nhân.\n\n"
                        "Câu hỏi mới:\n"
                        "lệ phí bao nhiêu?\n\n"
                        "Câu hỏi độc lập:"
                    ),
                ),
                ("ai", "Trong thủ tục đăng ký thành lập doanh nghiệp tư nhân, lệ phí là bao nhiêu?"),
                (
                    "human",
                    (
                        "Lịch sử gần nhất:\n{chat_history}\n\n"
                        "Câu hỏi mới:\n{question}\n\n"
                        "Câu hỏi độc lập:"
                    ),
                ),
            ]
        )
        self.chain = None
        if settings.OPENROUTER_API_KEY:
            default_headers = {}
            if settings.OPENROUTER_SITE_URL:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            if settings.OPENROUTER_APP_NAME:
                default_headers["X-Title"] = settings.OPENROUTER_APP_NAME
            llm = ChatOpenAI(
                model=settings.OPENROUTER_QUERY_REWRITE_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=0,
                default_headers=default_headers or None,
            )
            self.chain = self.prompt | llm | StrOutputParser()

    def _format_recent_history(self, conversation_state: dict[str, Any]) -> str:
        history = conversation_state.get("recent_chat_history") or []
        if not history:
            return ""

        selected_lines: list[str] = []
        token_count = 0
        for item in reversed(history):
            role = str(item.get("role", "user")).strip()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            line = f"{role}: {content}"
            line_tokens = count_tokens_rough(line)
            if selected_lines and token_count + line_tokens > self.max_history_tokens:
                break
            selected_lines.append(line)
            token_count += line_tokens
        return "\n".join(reversed(selected_lines))

    def _fallback_rewrite(self, question: str, conversation_state: dict[str, Any]) -> str:
        last_filename = conversation_state.get("last_filename")
        last_procedure_title = conversation_state.get("last_procedure_title")
        last_doc = conversation_state.get("last_referenced_doc") or {}
        if not last_filename and isinstance(last_doc, dict):
            last_filename = last_doc.get("filename")
        if not last_procedure_title and isinstance(last_doc, dict):
            last_procedure_title = last_doc.get("procedure_title")

        if last_procedure_title:
            return f'Trong thủ tục "{last_procedure_title}", {question.strip()}'
        if last_filename:
            return f'Trong tài liệu "{last_filename}", {question.strip()}'
        return question

    def rewrite_after_intent(
        self,
        question: str,
        intent_resolution: dict[str, Any],
        conversation_state: dict[str, Any] | None = None,
    ) -> QueryRewrite:
        conversation_state = conversation_state or {}
        if not intent_resolution.get("is_follow_up"):
            return QueryRewrite(
                original_question=question,
                rewritten_question=question,
                was_rewritten=False,
                reason="Intent router did not mark query as follow-up; query is passed through.",
                stage="post_intent",
            )

        chat_history = self._format_recent_history(conversation_state)
        if not chat_history:
            return QueryRewrite(
                original_question=question,
                rewritten_question=question,
                was_rewritten=False,
                reason="No chat_history available; query is passed through.",
                stage="post_intent",
            )

        if self.chain is None:
            rewritten = self._fallback_rewrite(question, conversation_state)
            return QueryRewrite(
                original_question=question,
                rewritten_question=rewritten,
                was_rewritten=rewritten != question,
                reason="OpenRouter query rewrite LLM is not configured; used deterministic fallback.",
                stage="post_intent",
                used_llm=False,
            )

        try:
            rewritten = self.chain.invoke({"chat_history": chat_history, "question": question}).strip()
        except Exception:
            rewritten = self._fallback_rewrite(question, conversation_state)
            return QueryRewrite(
                original_question=question,
                rewritten_question=rewritten,
                was_rewritten=rewritten != question,
                reason="OpenRouter query rewrite failed; used deterministic fallback.",
                stage="post_intent",
                used_llm=False,
            )

        rewritten = rewritten.strip().strip('"')
        if not rewritten:
            rewritten = question
        return QueryRewrite(
            original_question=question,
            rewritten_question=rewritten,
            was_rewritten=rewritten != question,
            reason="Follow-up question rewritten with OpenRouter history-aware prompt.",
            stage="post_intent",
            used_llm=True,
        )

    def rewrite_standalone(
        self,
        question: str,
        conversation_state: dict[str, Any] | None = None,
    ) -> QueryRewrite:
        rewrite = self.rewrite_after_intent(
            question=question,
            intent_resolution={"intent": REWRITE_INTENT_FOLLOW_UP, "is_follow_up": True},
            conversation_state=conversation_state,
        )
        rewrite.stage = "pre_intent"
        return rewrite

    def rewrite(
        self,
        question: str,
        intent_resolution: dict[str, Any],
        scope_resolution: dict[str, Any] | None = None,
        document_resolution: dict[str, Any] | None = None,
        conversation_state: dict[str, Any] | None = None,
    ) -> QueryRewrite:
        _ = scope_resolution, document_resolution
        return self.rewrite_after_intent(
            question=question,
            intent_resolution=intent_resolution,
            conversation_state=conversation_state,
        )
