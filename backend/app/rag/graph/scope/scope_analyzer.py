# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.constants import (
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    RETRIEVAL_SCOPE_NEED_CLARIFICATION,
    RETRIEVAL_SCOPE_SYSTEM_DOCS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
    SOURCE_TYPE_SYSTEM,
    SOURCE_TYPE_USER_UPLOAD,
)


SCOPE_VALUES = {
    "system_only",
    "current_uploads_only",
    "past_uploads_only",
    "user_uploads_all",
    "mixed",
    "need_clarification",
}

RESOLUTION_MODES = {
    "reuse_last_context",
    "switch_scope",
    "resolve_new_procedure",
    "resolve_current_upload",
    "resolve_previous_upload",
    "resolve_by_filename",
    "resolve_by_time_hint",
    "semantic_document_search",
    "mixed",
    "need_clarification",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    stripped = stripped.replace("đ", "d")
    return re.sub(r"\s+", " ", stripped).strip()


def _trim_to_last_complete_json_fragment(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None

    end = max(raw.rfind("}"), raw.rfind("]"))
    if end < start:
        return None

    fragment = raw[start : end + 1]
    if fragment.endswith("]"):
        fragment += "}"
    return fragment


@dataclass
class StructuredScopeResolution:
    action: str = "resolve_document"
    scope: str = RETRIEVAL_SCOPE_NEED_CLARIFICATION
    targets: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    should_reuse_last_filter: bool = False
    source_type: str = "none"
    procedure_title_hint: str | None = None
    document_name_hint: str | None = None
    document_id_hint: str | None = None
    time_hint: str | None = None
    document_topic_hint: str | None = None
    resolution_mode: str = "need_clarification"
    needs_clarification: bool = False
    reason: str = ""
    used_llm: bool = False
    llm_status: str = "not_attempted"
    llm_failure_stage: str | None = None
    llm_failure_detail: str | None = None
    llm_raw_preview: str | None = None
    llm_used_recovered_json: bool = False

    def _default_targets(self, scope: str) -> list[dict[str, Any]]:
        if scope == "mixed":
            user_session_scope = "past_sessions" if self.time_hint else "current_session"
            return [
                {
                    "source_type": "system",
                    "session_scope": None,
                    "procedure_title_hint": self.procedure_title_hint,
                    "document_name_hint": None,
                    "time_hint": None,
                },
                {
                    "source_type": "user_upload",
                    "session_scope": user_session_scope,
                    "procedure_title_hint": None,
                    "document_name_hint": self.document_name_hint,
                    "time_hint": self.time_hint,
                },
            ]
        if scope == "current_uploads_only":
            return [
                {
                    "source_type": "user_upload",
                    "session_scope": "current_session",
                    "procedure_title_hint": None,
                    "document_name_hint": self.document_name_hint,
                    "time_hint": self.time_hint,
                }
            ]
        if scope == "past_uploads_only":
            return [
                {
                    "source_type": "user_upload",
                    "session_scope": "past_sessions",
                    "procedure_title_hint": None,
                    "document_name_hint": self.document_name_hint,
                    "time_hint": self.time_hint,
                }
            ]
        if scope == "user_uploads_all":
            return [
                {
                    "source_type": "user_upload",
                    "session_scope": "all_sessions",
                    "procedure_title_hint": None,
                    "document_name_hint": self.document_name_hint,
                    "time_hint": self.time_hint,
                }
            ]
        if scope == "need_clarification":
            return []
        return [
            {
                "source_type": "system",
                "session_scope": None,
                "procedure_title_hint": self.procedure_title_hint,
                "document_name_hint": None,
                "time_hint": None,
            }
        ]

    def _public_scope(self) -> str:
        if self.scope == RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER or self.resolution_mode == "mixed":
            return "mixed"
        if self.scope == RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS:
            return "current_uploads_only"
        if self.scope in {RETRIEVAL_SCOPE_USER_ALL_UPLOADS, RETRIEVAL_SCOPE_USER_FILE_NAME}:
            if self.resolution_mode == "resolve_by_time_hint" or self.time_hint:
                return "past_uploads_only"
            return "user_uploads_all"
        if self.scope in {RETRIEVAL_SCOPE_SYSTEM_DOCS, RETRIEVAL_SCOPE_SYSTEM_PROCEDURE}:
            return "system_only"
        if self.scope == RETRIEVAL_SCOPE_NEED_CLARIFICATION:
            return "need_clarification"
        return self.scope

    def model_dump(self) -> dict[str, Any]:
        scope = self._public_scope()
        action = self.action
        if self.should_reuse_last_filter:
            action = "reuse_last_filter"
        elif scope == "mixed" or self.resolution_mode == "mixed":
            action = "mixed_retrieval"
        elif self.needs_clarification or scope == "need_clarification":
            action = "need_clarification"
        elif action not in {"reuse_last_filter", "resolve_document", "mixed_retrieval", "need_clarification"}:
            action = "resolve_document"
        if action == "reuse_last_filter":
            targets = self.targets if self.targets else []
        else:
            targets = self.targets or self._default_targets(scope)
        payload = {
            "action": action,
            "scope": scope,
            "targets": targets,
            "confidence": self.confidence,
        }
        return payload


ScopeResolution = StructuredScopeResolution


class ScopeAnalyzer:
    source_switch_terms = (
        "vua upload",
        "vua up",
        "toi vua upload",
        "toi vua up",
        "file toi",
        "tai lieu cua toi",
        "theo tai lieu cua toi",
        "file nay",
        "tai lieu nay",
        "vua gui",
        "toi vua gui",
        "toi upload",
        "tai lieu toi upload",
        "file vua upload",
        "file vua gui",
        "tai lieu vua gui",
        "file toi upload hom qua",
        "file hom qua toi upload",
        "hom truoc",
        "tuan truoc",
        "tuan sau",
        "hom kia",
        "ngay ",
        "lan truoc",
        "tai lieu cu",
        "file da tung upload",
        "so sanh",
        "doi chieu",
        "voi quy dinh he thong",
    )
    system_document_terms = (
        "le phi",
        "phi",
        "cap lai",
        "thong bao",
        "van ban buu chinh",
        "buu chinh",
        "thoi han",
        "giay to",
        "ho so",
        "co quan",
        "noi nop",
        "quy dinh",
    )

    def __init__(self) -> None:
        self.chain = None
        if settings.SCOPE_RESOLVER_USE_LLM and settings.OPENROUTER_API_KEY:
            default_headers = {}
            if settings.OPENROUTER_SITE_URL:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            if settings.OPENROUTER_APP_NAME:
                default_headers["X-Title"] = settings.OPENROUTER_APP_NAME
            llm = ChatOpenAI(
                model=settings.OPENROUTER_SCOPE_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=0,
                max_tokens=settings.OPENROUTER_SCOPE_MAX_TOKENS,
                default_headers=default_headers or None,
            )
            self.chain = self._prompt() | llm

    def _prompt(self) -> ChatPromptTemplate:
        system_prompt = """Bạn phân loại scope truy hồi cho hệ thống hỏi đáp tài liệu hành chính Việt Nam.
Chỉ trả về đúng 1 object JSON hợp lệ. Không markdown. Không giải thích. Không thêm key.

Schema:
{
  "action": "resolve_document | mixed_retrieval | reuse_last_filter | need_clarification",
  "scope": "system_only | current_uploads_only | past_uploads_only | user_uploads_all | mixed | need_clarification",
  "targets": [
    {
      "source_type": "system | user_upload",
      "session_scope": "current_session | past_sessions | all_sessions | null",
      "procedure_title_hint": null,
      "document_name_hint": null,
      "time_hint": null
    }
  ],
  "confidence": 0.0
}

Quy tắc:
- `reuse_last_filter`: chỉ khi `was_rewritten=true`, `has_last_filter=true`, và câu hỏi không chuyển sang upload, tài liệu cũ, hay so sánh.
- `current_uploads_only`: file vừa upload, vừa gửi, file hiện tại, file của tôi trong session này.
- `past_uploads_only`: file cũ, hôm qua, hôm trước, tuần trước, hoặc có mốc ngày.
- `system_only`: thủ tục hành chính, lệ phí, giấy tờ, thời hạn, nơi nộp, quy định, không có tín hiệu upload.
- `mixed`: chỉ khi thực sự so sánh hoặc đối chiếu file upload với quy định hệ thống.
- `need_clarification`: rất hiếm, chỉ khi không thể phân biệt nguồn.

Quy tắc target:
- Trả JSON càng ngắn càng tốt.
- Có thể bỏ các key `null`.
- Với `targets`, chỉ cần giữ các key có giá trị thật sự cần thiết.
- `confidence` trong khoảng `0..1`, có thể bỏ nếu không chắc.
"""
        return ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Ví dụ 1: Lệ phí khi cấp lại thông báo văn bản bưu chính là bao nhiêu?"),
                AIMessage(
                    content='{"action":"resolve_document","scope":"system_only","targets":[{"source_type":"system","session_scope":null,"procedure_title_hint":"cấp lại thông báo văn bản bưu chính","document_name_hint":null,"time_hint":null}],"confidence":0.95}'
                ),
                HumanMessage(content="Ví dụ 2: Đối chiếu file tôi upload với quy định hệ thống"),
                AIMessage(
                    content='{"action":"mixed_retrieval","scope":"mixed","targets":[{"source_type":"system","session_scope":null,"procedure_title_hint":null,"document_name_hint":null,"time_hint":null},{"source_type":"user_upload","session_scope":"current_session","procedure_title_hint":null,"document_name_hint":null,"time_hint":null}],"confidence":0.9}'
                ),
                ("human", "Phân loại state sau đây:\n{state_json}\n\nChỉ trả JSON một dòng."),
            ]
        )
    def _has_source_switch_signal(self, query: str) -> bool:
        normalized = _normalize_text(query)
        return any(term in normalized for term in self.source_switch_terms) or bool(
            re.search(r"\b[a-z0-9][a-z0-9_\-\s().\[\]]*\.(?:pdf|docx?|xlsx?|pptx?|txt|md)\b", normalized)
        )

    def _has_system_document_signal(self, query: str) -> bool:
        normalized = _normalize_text(query)
        return any(term in normalized for term in self.system_document_terms)

    def _has_mixed_signal(self, query: str) -> bool:
        normalized = _normalize_text(query)
        return any(term in normalized for term in ("so sanh", "doi chieu", "khac nhau", "giong nhau", "dap ung", "voi quy dinh he thong"))

    def _filename_hint(self, query: str) -> str | None:
        match = re.search(r"\b([a-z0-9][a-z0-9_\-().\[\]]*\.(?:pdf|docx?|xlsx?|pptx?|txt|md))\b", query, re.I)
        return match.group(0).strip() if match else None

    def _procedure_hint(self, query: str) -> str | None:
        match = re.search(r"(?i)(?:thủ tục|thu tuc)\s+(.+?)(?:\s+cần|\s+can|\s+gồm|\s+gom|\s+là|\s+la|\?|$)", query)
        if not match:
            return None
        candidate = match.group(1).strip(" .,:;?")
        if _normalize_text(candidate) in {"gi", "nao", "nhu nao", "gi vay", "gi the"}:
            return None
        return candidate or None

    def _topic_hint(self, query: str) -> str | None:
        match = re.search(r"(?i)(?:upload về|upload ve|tài liệu.*về|tai lieu.*ve)\s+(.+?)(?:\s+có|\s+co|\s+nói|\s+noi|\?|$)", query)
        if not match:
            return None
        return match.group(1).strip(" .,:;?") or None

    def _time_hint(self, query: str) -> str | None:
        normalized = _normalize_text(query)
        if "hom qua" in normalized:
            return "yesterday"
        if "hom truoc" in normalized:
            return "yesterday"
        if "hom kia" in normalized:
            return "two_days_ago"
        if "tuan truoc" in normalized or "last week" in normalized:
            return "last_week"
        if "tuan sau" in normalized or "next week" in normalized:
            return "next_week"
        match = re.search(r"(?i)ngày\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", query)
        if match:
            return match.group(1)
        return None

    def _fallback(self, state: dict[str, Any], reason: str = "Resolved by deterministic fallback.") -> StructuredScopeResolution:
        query = state.get("final_query") or state.get("original_query") or ""
        normalized = _normalize_text(query)
        action = state.get("retrieval_plan", {}).get("action")
        last_context = (state.get("runtime_context") or {}).get("last_resolved_context") or {}
        filename = self._filename_hint(query)
        procedure = self._procedure_hint(query)
        topic = self._topic_hint(query)
        time_hint = self._time_hint(query)

        if action == "reuse_last_filter" and last_context.get("filter") and not self._has_source_switch_signal(query):
            return StructuredScopeResolution(
                scope=last_context.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION,
                resolution_mode="reuse_last_context",
                should_reuse_last_filter=True,
                source_type=last_context.get("source_type") or "none",
                procedure_title_hint=last_context.get("procedure_title"),
                document_name_hint=last_context.get("filename"),
                document_id_hint=last_context.get("document_id"),
                confidence=0.9,
                reason="Safe follow-up reused last resolved context.",
            )
        if any(term in normalized for term in ("so sanh", "doi chieu", "khac nhau", "giong nhau", "dap ung")):
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
                resolution_mode="mixed",
                source_type="hybrid",
                targets=[
                    {
                        "source_type": "system",
                        "session_scope": None,
                        "procedure_title_hint": procedure,
                        "document_name_hint": None,
                        "time_hint": None,
                    },
                    {
                        "source_type": "user_upload",
                        "session_scope": "past_sessions" if time_hint else "current_session",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": time_hint,
                    },
                ],
                confidence=0.84,
                reason=reason,
            )
        if filename:
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_USER_FILE_NAME,
                resolution_mode="resolve_by_filename",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                document_name_hint=filename,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "past_sessions",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": None,
                    }
                ],
                confidence=0.88,
                reason=reason,
            )
        if any(
            term in normalized
            for term in (
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
                "file cua toi",
                "tai lieu cua toi",
                "theo tai lieu cua toi",
            )
        ):
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
                resolution_mode="resolve_current_upload",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "current_session",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": None,
                    }
                ],
                confidence=0.86,
                reason=reason,
            )
        if time_hint:
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
                resolution_mode="resolve_by_time_hint",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                time_hint=time_hint,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "past_sessions",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": time_hint,
                    }
                ],
                confidence=0.84,
                reason=reason,
            )
        if topic and "upload" in normalized:
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
                resolution_mode="semantic_document_search",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                document_topic_hint=topic,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "all_sessions",
                        "procedure_title_hint": None,
                        "document_name_hint": None,
                        "time_hint": None,
                    }
                ],
                confidence=0.76,
                reason=reason,
            )
        if procedure:
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
                resolution_mode="resolve_new_procedure",
                source_type=SOURCE_TYPE_SYSTEM,
                procedure_title_hint=procedure,
                targets=[
                    {
                        "source_type": "system",
                        "session_scope": None,
                        "procedure_title_hint": procedure,
                        "document_name_hint": None,
                        "time_hint": None,
                    }
                ],
                confidence=0.82,
                reason=reason,
            )
        if any(
            term in normalized
            for term in (
                "le phi",
                "phi",
                "cap lai",
                "dang ky",
                "thong bao",
                "van ban buu chinh",
                "buu chinh",
                "ho so",
                "giay to",
                "thoi han",
                "noi nop",
                "co quan",
            )
        ):
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_SYSTEM_DOCS,
                resolution_mode="resolve_new_procedure",
                source_type=SOURCE_TYPE_SYSTEM,
                targets=[
                    {
                        "source_type": "system",
                        "session_scope": None,
                        "procedure_title_hint": procedure,
                        "document_name_hint": None,
                        "time_hint": None,
                    }
                ],
                confidence=0.7,
                reason="Administrative document question without upload signal; defaulted to system docs.",
            )
        if action == "resolve_system_procedure":
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
                resolution_mode="resolve_new_procedure",
                source_type=SOURCE_TYPE_SYSTEM,
                procedure_title_hint=procedure,
                targets=[
                    {
                        "source_type": "system",
                        "session_scope": None,
                        "procedure_title_hint": procedure,
                        "document_name_hint": None,
                        "time_hint": None,
                    }
                ],
                confidence=0.72,
                reason=reason,
            )
        if action == "resolve_current_upload":
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
                resolution_mode="resolve_current_upload",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "current_session",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": None,
                    }
                ],
                confidence=0.72,
                reason=reason,
            )
        if action == "resolve_previous_upload":
            return StructuredScopeResolution(
                scope=RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
                resolution_mode="resolve_previous_upload",
                source_type=SOURCE_TYPE_USER_UPLOAD,
                targets=[
                    {
                        "source_type": "user_upload",
                        "session_scope": "past_sessions",
                        "procedure_title_hint": None,
                        "document_name_hint": filename,
                        "time_hint": time_hint,
                    }
                ],
                confidence=0.72,
                reason=reason,
            )
        return StructuredScopeResolution(
            scope=RETRIEVAL_SCOPE_NEED_CLARIFICATION,
            resolution_mode="need_clarification",
            source_type="none",
            needs_clarification=True,
            confidence=0.45,
            reason=reason,
        )

    def _build_llm_input(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_context = state.get("runtime_context") or {}
        last_context = runtime_context.get("last_resolved_context") or {}
        current_session_docs = runtime_context.get("current_session_docs", []) or []
        active_document_ids = runtime_context.get("active_document_ids", []) or []
        selected_document_ids = state.get("selected_document_ids", []) or []
        return {
            "query": state.get("final_query") or state.get("original_query"),
            "original_query": state.get("original_query"),
            "was_rewritten": bool(state.get("was_rewritten")),
            "intent": (state.get("intent_resolution") or {}).get("intent"),
            "needs_retrieval": (state.get("intent_resolution") or {}).get("needs_retrieval"),
            "planner_target_scope": (state.get("retrieval_plan") or {}).get("target_scope"),
            "requested_scope": state.get("requested_scope"),
            "has_last_filter": bool(last_context.get("filter")),
            "last_scope": last_context.get("scope") or runtime_context.get("last_scope"),
            "last_source_type": last_context.get("source_type"),
            "last_procedure_title": last_context.get("procedure_title") or runtime_context.get("last_procedure_title"),
            "last_filename": last_context.get("filename") or runtime_context.get("last_filename"),
            "last_document_id": last_context.get("document_id"),
            "current_session_doc_count": len(current_session_docs),
            "active_document_count": len(active_document_ids),
            "selected_document_count": len(selected_document_ids),
            "candidate_count": len(state.get("document_candidates") or []),
        }

    def _clean_payload(self, payload: dict[str, Any]) -> StructuredScopeResolution:
        action = str(payload.get("action") or payload.get("mode") or "resolve_document")
        scope = str(payload.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION)
        if scope not in SCOPE_VALUES:
            scope = RETRIEVAL_SCOPE_NEED_CLARIFICATION
        mode = str(payload.get("resolution_mode") or payload.get("mode") or "need_clarification")
        if mode not in RESOLUTION_MODES:
            mode = "need_clarification"
        targets = payload.get("targets")
        if not isinstance(targets, list):
            targets = []
        normalized_targets: list[dict[str, Any]] = []
        for target in targets:
            if not isinstance(target, dict):
                continue
            normalized_targets.append(
                {
                    "source_type": target.get("source_type") or "none",
                    "session_scope": target.get("session_scope"),
                    "procedure_title_hint": target.get("procedure_title_hint"),
                    "document_name_hint": target.get("document_name_hint"),
                    "time_hint": target.get("time_hint"),
                }
            )
        if not normalized_targets:
            legacy_hints = payload.get("hints") if isinstance(payload.get("hints"), dict) else {}
            normalized_targets = [
                {
                    "source_type": payload.get("source_type") or payload.get("source") or "none",
                    "session_scope": payload.get("session_scope"),
                    "procedure_title_hint": payload.get("procedure_title_hint") or legacy_hints.get("procedure_title"),
                    "document_name_hint": payload.get("document_name_hint"),
                    "time_hint": payload.get("time_hint") or legacy_hints.get("time") or payload.get("time"),
                }
            ] if payload.get("source_type") or payload.get("source") or payload.get("procedure_title_hint") or payload.get("document_name_hint") or payload.get("time_hint") or legacy_hints else []
        should_reuse = action == "reuse_last_filter" or bool(payload.get("should_reuse_last_filter") or payload.get("reuse"))
        return StructuredScopeResolution(
            action=action,
            scope=scope,
            resolution_mode=mode,
            should_reuse_last_filter=should_reuse,
            targets=normalized_targets,
            source_type=str(payload.get("source_type") or payload.get("source") or "none"),
            procedure_title_hint=payload.get("procedure_title_hint"),
            document_name_hint=payload.get("document_name_hint"),
            document_id_hint=payload.get("document_id_hint"),
            time_hint=payload.get("time_hint") or payload.get("time"),
            document_topic_hint=payload.get("document_topic_hint") or payload.get("topic"),
            needs_clarification=bool(payload.get("needs_clarification") or payload.get("clarify")),
            confidence=float(payload.get("confidence") or 0.0),
            reason=str(payload.get("reason") or "Resolved by scope LLM."),
            used_llm=True,
        )

    def _parse_payload(self, raw: str) -> tuple[StructuredScopeResolution, bool]:
        try:
            return self._clean_payload(json.loads(raw)), False
        except json.JSONDecodeError:
            recovered = _trim_to_last_complete_json_fragment(raw)
            if recovered is None:
                raise
            return self._clean_payload(json.loads(recovered)), True

    def _apply_security_guards(self, resolution: StructuredScopeResolution, state: dict[str, Any]) -> StructuredScopeResolution:
        query = state.get("final_query") or state.get("original_query") or ""
        last_context = (state.get("runtime_context") or {}).get("last_resolved_context") or {}
        if resolution.should_reuse_last_filter:
            if not last_context.get("filter") or self._has_source_switch_signal(query):
                resolution.should_reuse_last_filter = False
                resolution.resolution_mode = "need_clarification"
                resolution.scope = RETRIEVAL_SCOPE_NEED_CLARIFICATION
                resolution.needs_clarification = True
                resolution.reason = "Reuse last filter blocked by source-switch/security guard."
        if resolution._public_scope() == "mixed" and not resolution.targets:
            resolution.targets = resolution._default_targets("mixed")
        return resolution

    def _apply_upload_priority_guards(
        self, resolution: StructuredScopeResolution, state: dict[str, Any]
    ) -> StructuredScopeResolution:
        query = state.get("final_query") or state.get("original_query") or ""
        if not self._has_source_switch_signal(query) or self._has_mixed_signal(query):
            return resolution
        if resolution._public_scope() != "mixed":
            return resolution

        filename = self._filename_hint(query)
        time_hint = self._time_hint(query)
        if time_hint:
            resolution.scope = RETRIEVAL_SCOPE_USER_ALL_UPLOADS
            resolution.resolution_mode = "resolve_by_time_hint"
            resolution.time_hint = time_hint
            session_scope = "past_sessions"
        else:
            resolution.scope = RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS
            resolution.resolution_mode = "resolve_current_upload"
            session_scope = "current_session"

        resolution.action = "resolve_document"
        resolution.source_type = SOURCE_TYPE_USER_UPLOAD
        resolution.document_name_hint = filename
        resolution.targets = [
            {
                "source_type": "user_upload",
                "session_scope": session_scope,
                "procedure_title_hint": None,
                "document_name_hint": filename,
                "time_hint": time_hint,
            }
        ]
        resolution.reason = "Upload source signal without comparison; routed to user upload scope."
        return resolution

    def _apply_administrative_guards(
        self, resolution: StructuredScopeResolution, state: dict[str, Any]
    ) -> StructuredScopeResolution:
        query = state.get("final_query") or state.get("original_query") or ""
        normalized_query = _normalize_text(query)
        if self._has_source_switch_signal(query) or not self._has_system_document_signal(query):
            return resolution
        if resolution._public_scope() != "need_clarification":
            return resolution

        procedure = self._procedure_hint(query)
        if procedure or any(term in normalized_query for term in ("thu tuc", "quy trinh", "dang ky")):
            resolution.scope = RETRIEVAL_SCOPE_SYSTEM_PROCEDURE
            resolution.resolution_mode = "resolve_new_procedure"
            resolution.source_type = SOURCE_TYPE_SYSTEM
            resolution.procedure_title_hint = procedure or resolution.procedure_title_hint
            resolution.needs_clarification = False
            resolution.confidence = max(resolution.confidence, 0.82)
            resolution.reason = "Strong administrative signal; routed to system procedure."
            return resolution

        resolution.scope = RETRIEVAL_SCOPE_SYSTEM_DOCS
        resolution.resolution_mode = "switch_scope"
        resolution.source_type = SOURCE_TYPE_SYSTEM
        resolution.needs_clarification = False
        resolution.confidence = max(resolution.confidence, 0.8)
        resolution.reason = "Strong administrative signal; routed to system docs."
        return resolution

    def resolve(self, state: dict[str, Any]) -> StructuredScopeResolution:
        if self.chain is None:
            resolution = self._fallback(state, reason="Scope LLM unavailable; used deterministic fallback.")
            resolution.llm_status = "chain_unavailable"
            resolution.llm_failure_stage = "chain_init"
            resolution.llm_failure_detail = "ScopeAnalyzer chain is None."
            return self._apply_administrative_guards(resolution, state)
        try:
            response = self.chain.invoke(
                {"state_json": json.dumps(self._build_llm_input(state), ensure_ascii=False, default=str)}
            )
            raw = response.content if hasattr(response, "content") else response
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
            raw = str(raw).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.removeprefix("json").strip()
            resolution, used_recovered_json = self._parse_payload(raw)
            resolution.llm_status = "parsed_with_recovery" if used_recovered_json else "parsed"
            resolution.llm_used_recovered_json = used_recovered_json
            resolution.llm_raw_preview = raw[:240]
        except Exception as exc:
            resolution = self._fallback(state, reason="Scope LLM failed; used deterministic fallback.")
            resolution.llm_status = "fallback"
            resolution.llm_failure_stage = "invoke_or_parse"
            resolution.llm_failure_detail = f"{type(exc).__name__}: {exc}"
            try:
                resolution.llm_raw_preview = raw[:240]
            except Exception:
                resolution.llm_raw_preview = None
        resolution = self._apply_security_guards(resolution, state)
        resolution = self._apply_upload_priority_guards(resolution, state)
        return self._apply_administrative_guards(resolution, state)
