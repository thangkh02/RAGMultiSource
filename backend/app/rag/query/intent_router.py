from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.constants import SOURCE_TYPE_SYSTEM, SOURCE_TYPE_USER_UPLOAD
from app.core.config import settings


INTENT_ASK_QUESTION = "ask_question"
INTENT_SUMMARIZE_DOCUMENT = "summarize_document"
INTENT_COMPARE_DOCUMENTS = "compare_documents"
INTENT_FIND_INFORMATION = "find_information"
INTENT_GENERAL_QUERY = "general_query"
INTENT_NEED_CLARIFICATION = "need_clarification"
INTENT_UNSUPPORTED = "unsupported"

ANSWER_STYLE_SHORT = "short_answer"
ANSWER_STYLE_BULLET_LIST = "bullet_list"
ANSWER_STYLE_SUMMARY = "summary"
ANSWER_STYLE_COMPARISON = "comparison"
ANSWER_STYLE_STEPS = "steps"

ACTION_RESOLVE_DOCUMENT = "resolve_document"
ACTION_MIXED_RETRIEVAL = "mixed_retrieval"
ACTION_REUSE_LAST_FILTER = "reuse_last_filter"
ACTION_NEED_CLARIFICATION = "need_clarification"
ACTION_DIRECT_ANSWER = "direct_answer"

SCOPE_SYSTEM_ONLY = "system_only"
SCOPE_CURRENT_UPLOADS_ONLY = "current_uploads_only"
SCOPE_PAST_UPLOADS_ONLY = "past_uploads_only"
SCOPE_USER_UPLOADS_ALL = "user_uploads_all"
SCOPE_MIXED = "mixed"
SCOPE_NONE = "none"
SCOPE_NEED_CLARIFICATION = "need_clarification"

ALLOWED_SCOPES = {
    SCOPE_SYSTEM_ONLY,
    SCOPE_CURRENT_UPLOADS_ONLY,
    SCOPE_PAST_UPLOADS_ONLY,
    SCOPE_USER_UPLOADS_ALL,
    SCOPE_MIXED,
    SCOPE_NONE,
    SCOPE_NEED_CLARIFICATION,
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    stripped = stripped.replace("đ", "d")
    return re.sub(r"\s+", " ", stripped).strip()


@dataclass
class IntentResolution:
    intent: str
    answer_style: str = ANSWER_STYLE_SHORT
    is_follow_up: bool = False
    needs_retrieval: bool = True
    action: str | None = None
    scope: str = SCOPE_NEED_CLARIFICATION
    targets: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.75
    matched_rules: list[str] = field(default_factory=list)
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("answer_style", None)
        payload.pop("reason", None)
        if payload.get("action") in {None, ACTION_RESOLVE_DOCUMENT, ACTION_DIRECT_ANSWER}:
            payload.pop("action", None)
        return payload


class IntentRouter:
    _document_terms = (
        "file",
        "tai lieu",
        "van ban",
        "thu tuc",
        "ho so",
        "quy dinh",
        "quy trinh",
        "le phi",
        "thoi han",
        "giay to",
        "trinh tu",
        "cach thuc",
        "thuc hien",
        "nop ho so",
        "co quan",
    )
    _ambiguous_terms = ("tai lieu do", "file do", "van ban do", "cai do")
    _current_upload_terms = (
        "vua upload",
        "vua up",
        "toi vua upload",
        "toi vua up",
        "file vua upload",
        "tai lieu vua upload",
        "file vua gui",
        "tai lieu vua gui",
        "file nay",
        "tai lieu nay",
        "file hien tai",
        "tai lieu hien tai",
    )
    _user_upload_terms = (
        "file toi",
        "tai lieu cua toi",
        "theo tai lieu cua toi",
        "upload",
        "up len",
        "file cu",
        "tai lieu cu",
        "file da tung upload",
        "tai lieu da tung upload",
        "tai lieu toi tung upload",
    )
    _system_terms = (
        "thu tuc",
        "ho so",
        "quy dinh",
        "quy trinh",
        "le phi",
        "phi",
        "thoi han",
        "giay to",
        "trinh tu",
        "cach thuc",
        "nop ho so",
        "co quan",
        "van ban buu chinh",
        "buu chinh",
        "dang ky",
        "cap lai",
    )
    _source_switch_terms = _current_upload_terms + (
        "hom qua",
        "hom truoc",
        "hom kia",
        "tuan truoc",
        "thang truoc",
        "lan truoc",
        "file cu",
        "tai lieu cu",
        "so sanh",
        "doi chieu",
        "voi quy dinh",
        "quy dinh he thong",
    )
    _follow_up_terms = (
        "the",
        "con",
        "thi sao",
        "bao lau",
        "le phi",
        "phi",
        "can chuan bi",
        "giay to",
        "ho so",
        "trinh tu",
        "doi tuong",
        "no",
        "cai do",
    )

    def __init__(self) -> None:
        self.chain = None
        if settings.INTENT_ROUTER_USE_LLM and settings.OPENROUTER_API_KEY:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an intent router for a Vietnamese administrative-document RAG system.\n"
                            "Choose intent and retrieval source scope in one step.\n"
                            "Intents: ask_question, summarize_document, compare_documents, find_information, "
                            "general_query, need_clarification, unsupported.\n"
                            "Scopes: system_only, current_uploads_only, past_uploads_only, user_uploads_all, "
                            "mixed, none, need_clarification.\n"
                            "Use system_only for administrative/system docs. Use current_uploads_only for files "
                            "just uploaded in this session. Use past_uploads_only for user uploads with time hints "
                            "like yesterday, last week, last month, or a date. Use user_uploads_all for generic "
                            "user uploads. Use mixed for comparing uploaded files with system regulations.\n"
                            "Set is_follow_up=true only when the query depends on recent conversation.\n"
                            "Only include action when action is reuse_last_filter. Omit action for normal retrieval.\n"
                            "Use general_query only for greetings or questions about the chatbot itself.\n"
                            "Do not create metadata filters. Do not include reason.\n"
                            "Return valid JSON only, no markdown, no explanation.\n"
                            "Schema: {\"intent\":\"...\",\"needs_retrieval\":true,"
                            "\"is_follow_up\":false,\"scope\":\"system_only\","
                            "\"targets\":[{\"source_type\":\"system\",\"session_scope\":null,"
                            "\"procedure_title_hint\":null,\"document_name_hint\":null,\"time_hint\":null}],"
                            "\"confidence\":0.0,\"matched_rules\":[]}"
                        ),
                    ),
                    ("human", "State: {state_json}"),
                ]
            )
            default_headers = {}
            if settings.OPENROUTER_SITE_URL:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            if settings.OPENROUTER_APP_NAME:
                default_headers["X-Title"] = settings.OPENROUTER_APP_NAME
            llm = ChatOpenAI(
                model=settings.OPENROUTER_INTENT_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=0,
                default_headers=default_headers or None,
            )
            self.chain = prompt | llm | StrOutputParser()

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def _answer_style(self, text: str, intent: str) -> str:
        if intent == INTENT_COMPARE_DOCUMENTS:
            return ANSWER_STYLE_COMPARISON
        if intent == INTENT_SUMMARIZE_DOCUMENT:
            return ANSWER_STYLE_SUMMARY
        if any(term in text for term in ("cac buoc", "quy trinh", "cach thuc")):
            return ANSWER_STYLE_STEPS
        if any(term in text for term in ("liet ke", "danh sach", "gom nhung gi", "can ho so", "ho so gi", "giay to gi")):
            return ANSWER_STYLE_BULLET_LIST
        return ANSWER_STYLE_SHORT

    def _filename_hint(self, question: str) -> str | None:
        match = re.search(r"\b([a-z0-9][a-z0-9_\-().\[\]]*\.(?:pdf|docx?|xlsx?|pptx?|txt|md))\b", question, re.I)
        return match.group(0).strip() if match else None

    def _procedure_hint(self, question: str) -> str | None:
        match = re.search(r"(?i)(?:thủ tục|thu tuc)\s+(.+?)(?:\s+cần|\s+can|\s+gồm|\s+gom|\s+là|\s+la|\?|$)", question)
        if not match:
            return None
        candidate = match.group(1).strip(" .,:;?")
        if _normalize_text(candidate) in {"gi", "nao", "nhu nao", "gi vay", "gi the"}:
            return None
        return candidate or None

    def _time_hint(self, question: str) -> str | None:
        text = _normalize_text(question)
        if "hom qua" in text or "hom truoc" in text:
            return "yesterday"
        if "hom kia" in text:
            return "two_days_ago"
        if "tuan truoc" in text or "last week" in text:
            return "last_week"
        if "thang truoc" in text or "last month" in text:
            return "last_month"
        match = re.search(r"(?i)ngày\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", question)
        if match:
            return match.group(1)
        return None

    def _has_source_switch_signal(self, question: str) -> bool:
        text = _normalize_text(question)
        return self._contains_any(text, self._source_switch_terms) or self._filename_hint(question) is not None

    def _public_scope(self, scope: str | None) -> str:
        if scope in {"system_docs", "system_procedure"}:
            return SCOPE_SYSTEM_ONLY
        if scope in {"current_upload", "current_session_uploads"}:
            return SCOPE_CURRENT_UPLOADS_ONLY
        if scope in {"all_user_uploads", "user_all_uploads", "user_file_name"}:
            return SCOPE_USER_UPLOADS_ALL
        if scope in {"hybrid_system_and_user", "mixed"}:
            return SCOPE_MIXED
        if scope in ALLOWED_SCOPES:
            return scope
        return SCOPE_NEED_CLARIFICATION

    def _looks_like_follow_up(self, question: str, conversation_state: dict[str, Any]) -> bool:
        last_context = conversation_state.get("last_resolved_context") or {}
        if not last_context.get("filter"):
            return False
        text = _normalize_text(question)
        return len(text.split()) <= 7 or self._contains_any(text, self._follow_up_terms)

    def _target(
        self,
        source_type: str,
        session_scope: str | None = None,
        procedure_title_hint: str | None = None,
        document_name_hint: str | None = None,
        time_hint: str | None = None,
    ) -> dict[str, Any]:
        return {
            "source_type": source_type,
            "session_scope": session_scope,
            "procedure_title_hint": procedure_title_hint,
            "document_name_hint": document_name_hint,
            "time_hint": time_hint,
        }

    def _normalize_targets(self, targets: Any) -> list[dict[str, Any]]:
        if not isinstance(targets, list):
            return []
        normalized: list[dict[str, Any]] = []
        for target in targets:
            if not isinstance(target, dict):
                continue
            normalized.append(
                self._target(
                    source_type=target.get("source_type") or "none",
                    session_scope=target.get("session_scope"),
                    procedure_title_hint=target.get("procedure_title_hint"),
                    document_name_hint=target.get("document_name_hint"),
                    time_hint=target.get("time_hint"),
                )
            )
        return normalized

    def _default_targets(
        self,
        scope: str,
        question: str,
        procedure_title_hint: str | None = None,
        document_name_hint: str | None = None,
        time_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        procedure_title_hint = procedure_title_hint or self._procedure_hint(question)
        document_name_hint = document_name_hint or self._filename_hint(question)
        if scope == SCOPE_SYSTEM_ONLY:
            return [self._target(SOURCE_TYPE_SYSTEM, procedure_title_hint=procedure_title_hint)]
        if scope == SCOPE_CURRENT_UPLOADS_ONLY:
            return [self._target(SOURCE_TYPE_USER_UPLOAD, session_scope="current_session", document_name_hint=document_name_hint)]
        if scope == SCOPE_PAST_UPLOADS_ONLY:
            return [self._target(SOURCE_TYPE_USER_UPLOAD, session_scope="past_sessions", document_name_hint=document_name_hint, time_hint=time_hint)]
        if scope == SCOPE_USER_UPLOADS_ALL:
            return [self._target(SOURCE_TYPE_USER_UPLOAD, session_scope="all_sessions", document_name_hint=document_name_hint, time_hint=time_hint)]
        if scope == SCOPE_MIXED:
            user_session_scope = "past_sessions" if time_hint else "current_session"
            return [
                self._target(SOURCE_TYPE_SYSTEM, procedure_title_hint=procedure_title_hint),
                self._target(SOURCE_TYPE_USER_UPLOAD, session_scope=user_session_scope, document_name_hint=document_name_hint, time_hint=time_hint),
            ]
        return []

    def _build_llm_state(self, question: str, conversation_state: dict[str, Any]) -> dict[str, Any]:
        last_context = conversation_state.get("last_resolved_context") or {}
        return {
            "query": question,
            "has_last_filter": bool(last_context.get("filter")),
            "last_scope": last_context.get("scope") or conversation_state.get("last_scope"),
            "last_source_type": last_context.get("source_type"),
            "last_procedure_title": last_context.get("procedure_title") or conversation_state.get("last_procedure_title"),
            "last_filename": last_context.get("filename") or conversation_state.get("last_filename"),
            "current_session_doc_count": len(conversation_state.get("current_session_docs") or []),
        }

    def _clean_payload(self, payload: dict[str, Any], question: str) -> IntentResolution | None:
        allowed_intents = {
            INTENT_ASK_QUESTION,
            INTENT_SUMMARIZE_DOCUMENT,
            INTENT_COMPARE_DOCUMENTS,
            INTENT_FIND_INFORMATION,
            INTENT_GENERAL_QUERY,
            INTENT_NEED_CLARIFICATION,
            INTENT_UNSUPPORTED,
        }
        intent = payload.get("intent")
        if intent not in allowed_intents:
            return None

        answer_style = payload.get("answer_style") or ANSWER_STYLE_SHORT
        if answer_style not in {
            ANSWER_STYLE_SHORT,
            ANSWER_STYLE_BULLET_LIST,
            ANSWER_STYLE_SUMMARY,
            ANSWER_STYLE_COMPARISON,
            ANSWER_STYLE_STEPS,
        }:
            answer_style = ANSWER_STYLE_SHORT

        scope = payload.get("scope") or SCOPE_NEED_CLARIFICATION
        if scope not in ALLOWED_SCOPES:
            scope = SCOPE_NEED_CLARIFICATION

        action = payload.get("action") or ACTION_RESOLVE_DOCUMENT
        if action not in {ACTION_RESOLVE_DOCUMENT, ACTION_MIXED_RETRIEVAL, ACTION_REUSE_LAST_FILTER, ACTION_NEED_CLARIFICATION, ACTION_DIRECT_ANSWER}:
            action = ACTION_RESOLVE_DOCUMENT
        if scope == SCOPE_MIXED:
            action = ACTION_MIXED_RETRIEVAL
        if intent == INTENT_GENERAL_QUERY or not bool(payload.get("needs_retrieval", True)):
            action = ACTION_DIRECT_ANSWER
            scope = SCOPE_NONE
        if intent == INTENT_NEED_CLARIFICATION:
            action = ACTION_NEED_CLARIFICATION
            scope = SCOPE_NEED_CLARIFICATION

        targets = self._normalize_targets(payload.get("targets"))
        if action == ACTION_REUSE_LAST_FILTER:
            targets = []
        elif not targets:
            targets = self._default_targets(
                scope=scope,
                question=question,
                time_hint=payload.get("time_hint"),
                procedure_title_hint=payload.get("procedure_title_hint"),
                document_name_hint=payload.get("document_name_hint"),
            )

        matched_rules = payload.get("matched_rules")
        if not isinstance(matched_rules, list):
            matched_rules = ["llm_intent_router"]
        elif "llm_intent_router" not in matched_rules:
            matched_rules.append("llm_intent_router")

        return IntentResolution(
            intent=intent,
            answer_style=answer_style,
            is_follow_up=bool(payload.get("is_follow_up")),
            needs_retrieval=bool(payload.get("needs_retrieval", intent not in {INTENT_GENERAL_QUERY, INTENT_NEED_CLARIFICATION, INTENT_UNSUPPORTED})),
            action=action,
            scope=scope,
            targets=targets,
            confidence=float(payload.get("confidence", 0.75)),
            matched_rules=matched_rules,
        )

    def _route_with_llm(self, question: str, conversation_state: dict[str, Any]) -> IntentResolution | None:
        if self.chain is None:
            return None
        try:
            raw = self.chain.invoke(
                {"state_json": json.dumps(self._build_llm_state(question, conversation_state), ensure_ascii=False, default=str)}
            ).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.removeprefix("json").strip()
            payload = json.loads(raw)
        except Exception:
            return None
        return self._clean_payload(payload, question)

    def _route_by_rules(self, question: str, conversation_state: dict[str, Any] | None = None) -> IntentResolution:
        conversation_state = conversation_state or {}
        text = _normalize_text(question)
        matched_rules: list[str] = []
        filename = self._filename_hint(question)
        procedure = self._procedure_hint(question)
        time_hint = self._time_hint(question)
        has_last_filter = bool((conversation_state.get("last_resolved_context") or {}).get("filter"))
        is_follow_up = self._looks_like_follow_up(question, conversation_state)
        source_switch = self._has_source_switch_signal(question)

        if self._contains_any(text, self._ambiguous_terms):
            return IntentResolution(
                intent=INTENT_NEED_CLARIFICATION,
                needs_retrieval=False,
                action=ACTION_NEED_CLARIFICATION,
                scope=SCOPE_NEED_CLARIFICATION,
                confidence=0.9,
                matched_rules=["ambiguous_reference_without_state"],
                reason="Question references a document ambiguously without conversation state.",
            )

        if any(term in text for term in ("ve tranh", "tao anh", "viet code", "lap trinh", "dat ve", "mua hang")):
            return IntentResolution(
                intent=INTENT_UNSUPPORTED,
                needs_retrieval=False,
                action=ACTION_DIRECT_ANSWER,
                scope=SCOPE_NONE,
                confidence=0.82,
                matched_rules=["unsupported_non_rag_task"],
                reason="Question is outside the supported document QA scope.",
            )

        if any(term in text for term in ("so sanh", "doi chieu", "khac nhau", "giong nhau", "dap ung")):
            intent = INTENT_COMPARE_DOCUMENTS
            scope = SCOPE_MIXED
            action = ACTION_MIXED_RETRIEVAL
            matched_rules.append("compare")
        elif any(term in text for term in ("tom tat", "noi dung chinh", "tong quan")):
            intent = INTENT_SUMMARIZE_DOCUMENT
            scope = SCOPE_CURRENT_UPLOADS_ONLY if self._contains_any(text, self._current_upload_terms) else SCOPE_SYSTEM_ONLY
            action = ACTION_RESOLVE_DOCUMENT
            matched_rules.append("summarize")
        elif any(term in text for term in ("tim", "tra cuu", "cho biet", "kiem tra")):
            intent = INTENT_FIND_INFORMATION
            scope = SCOPE_SYSTEM_ONLY
            action = ACTION_RESOLVE_DOCUMENT
            matched_rules.append("find_information")
        elif self._contains_any(text, self._document_terms):
            intent = INTENT_ASK_QUESTION
            scope = SCOPE_SYSTEM_ONLY
            action = ACTION_RESOLVE_DOCUMENT
            matched_rules.append("document_question")
        elif is_follow_up:
            intent = INTENT_ASK_QUESTION
            scope = self._public_scope((conversation_state.get("last_resolved_context") or {}).get("scope"))
            action = ACTION_REUSE_LAST_FILTER if has_last_filter and not source_switch else ACTION_RESOLVE_DOCUMENT
            matched_rules.append("follow_up")
        else:
            return IntentResolution(
                intent=INTENT_GENERAL_QUERY,
                needs_retrieval=False,
                action=ACTION_DIRECT_ANSWER,
                scope=SCOPE_NONE,
                confidence=0.82,
                matched_rules=["no_document_signal"],
                reason="Question does not appear to target documents.",
            )

        if action != ACTION_MIXED_RETRIEVAL:
            if self._contains_any(text, self._current_upload_terms):
                scope = SCOPE_CURRENT_UPLOADS_ONLY
                action = ACTION_RESOLVE_DOCUMENT
                matched_rules.append("current_upload")
            elif time_hint and (self._contains_any(text, self._user_upload_terms) or "upload" in text):
                scope = SCOPE_PAST_UPLOADS_ONLY
                action = ACTION_RESOLVE_DOCUMENT
                matched_rules.append("past_upload_time")
            elif filename:
                scope = SCOPE_USER_UPLOADS_ALL
                action = ACTION_RESOLVE_DOCUMENT
                matched_rules.append("filename_upload")
            elif self._contains_any(text, self._user_upload_terms):
                scope = SCOPE_USER_UPLOADS_ALL
                action = ACTION_RESOLVE_DOCUMENT
                matched_rules.append("user_upload")
            elif has_last_filter and is_follow_up and not source_switch:
                action = ACTION_REUSE_LAST_FILTER
                scope = self._public_scope((conversation_state.get("last_resolved_context") or {}).get("scope"))
                matched_rules.append("reuse_last_filter")
            elif self._contains_any(text, self._system_terms):
                scope = SCOPE_SYSTEM_ONLY
                matched_rules.append("system_docs")

        return IntentResolution(
            intent=intent,
            answer_style=self._answer_style(text, intent),
            is_follow_up=is_follow_up,
            needs_retrieval=True,
            action=action,
            scope=scope,
            targets=[] if action == ACTION_REUSE_LAST_FILTER else self._default_targets(scope, question, procedure_title_hint=procedure, document_name_hint=filename, time_hint=time_hint),
            confidence=0.8,
            matched_rules=matched_rules,
            reason="Resolved by deterministic query rules.",
        )

    def route(self, question: str, conversation_state: dict[str, Any] | None = None) -> IntentResolution:
        rule_result = self._route_by_rules(question, conversation_state=conversation_state)
        llm_result = self._route_with_llm(question, conversation_state=conversation_state or {})
        if llm_result is None:
            return rule_result

        # Deterministic rules are the safety rail: never let the LLM classify a
        # clear administrative/document question as a non-retrieval query.
        if rule_result.needs_retrieval and not llm_result.needs_retrieval:
            rule_result.matched_rules.append("llm_false_negative_guard")
            rule_result.reason = "Rule-based guard kept a document/admin question in the RAG path."
            return rule_result

        if rule_result.is_follow_up and not llm_result.is_follow_up:
            rule_result.matched_rules.append("llm_follow_up_guard")
            return rule_result

        if llm_result.action == ACTION_REUSE_LAST_FILTER and self._has_source_switch_signal(question):
            rule_result.matched_rules.append("llm_reuse_source_switch_guard")
            return rule_result

        if llm_result.confidence < 0.7 and rule_result.confidence >= llm_result.confidence:
            rule_result.matched_rules.append("low_confidence_llm_fallback")
            return rule_result

        return llm_result
