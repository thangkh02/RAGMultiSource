from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from langsmith import traceable

from app.core.constants import (
    RETRIEVAL_SCOPE_AUTO,
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_CURRENT_UPLOAD,
    RETRIEVAL_SCOPE_GENERAL_QUERY,
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    RETRIEVAL_SCOPE_NEED_CLARIFICATION,
    RETRIEVAL_SCOPE_SYSTEM_DOCS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
    SOURCE_TYPE_SYSTEM,
    SOURCE_TYPE_USER_UPLOAD,
    VISIBILITY_GLOBAL,
)
from app.rag.generation.source_formatter import SourceFormatter
from app.rag.query.intent_router import (
    INTENT_GENERAL_QUERY,
    INTENT_NEED_CLARIFICATION,
    IntentResolution,
)
from app.rag.retrieval.context_validator import FALLBACK_NO_CONTEXT
from app.rag.retrieval.filters import build_retrieval_filter
from app.rag.retrieval.resolvers.document_resolver import DocumentResolution
from app.rag.retrieval.strategy import RetrievalPlan
from app.rag.rewrite import QueryRewrite
from app.schemas.common_schema import SourceItem


INTENT_UNSUPPORTED = "unsupported"

ACTION_REUSE_LAST_FILTER = "reuse_last_filter"
ACTION_RESOLVE_SYSTEM_PROCEDURE = "resolve_system_procedure"
ACTION_RESOLVE_CURRENT_UPLOAD = "resolve_current_upload"
ACTION_RESOLVE_PREVIOUS_UPLOAD = "resolve_previous_upload"
ACTION_RESOLVE_USER_FILE_NAME = "resolve_user_file_name"
ACTION_SEMANTIC_DOCUMENT_SEARCH = "semantic_document_search"
ACTION_MIXED_RETRIEVAL = "mixed_retrieval"
ACTION_NEED_CLARIFICATION = "need_clarification"
ACTION_GENERAL_QUERY = "general_query"


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    stripped = stripped.replace("đ", "d")
    return re.sub(r"\s+", " ", stripped).strip()


def _and(*conditions: dict[str, Any]) -> dict[str, Any]:
    clean_conditions = [condition for condition in conditions if condition]
    if not clean_conditions:
        return {}
    if len(clean_conditions) == 1:
        return clean_conditions[0]
    return {"$and": clean_conditions}


def _filter_has_value(metadata_filter: dict[str, Any], key: str, value: Any) -> bool:
    if not isinstance(metadata_filter, dict):
        return False
    if metadata_filter.get(key) == value:
        return True
    for operator in ("$and", "$or"):
        nested = metadata_filter.get(operator)
        if isinstance(nested, list) and any(_filter_has_value(item, key, value) for item in nested):
            return True
    return False


def _flatten_simple_and_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata_filter, dict) or "$and" not in metadata_filter:
        return metadata_filter
    merged: dict[str, Any] = {}
    for item in metadata_filter["$and"]:
        if not isinstance(item, dict) or any(key.startswith("$") for key in item):
            return metadata_filter
        merged.update(item)
    return merged


@dataclass
class CandidateSelection:
    selected_document_ids: list[str]
    selected_documents: list[dict[str, Any]]
    confident: bool
    needs_clarification: bool = False
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RAGGraphNodes:
    source_switch_terms = (
        "file upload",
        "vua upload",
        "toi vua upload",
        "file toi",
        "tai lieu cua toi",
        "file vua upload",
        "tai lieu vua upload",
        "file vua gui",
        "file nay",
        "tai lieu nay",
        "hom qua",
        "hom truoc",
        "hom kia",
        "tuan truoc",
        "lan truoc",
        "file cu",
        "tai lieu cu",
        "so sanh",
        "doi chieu",
        "quy dinh he thong",
        ".pdf",
        ".doc",
        ".docx",
    )
    follow_up_terms = (
        "the",
        "con",
        "thi sao",
        "bao lau",
        "le phi",
        "phi",
        "can chuan bi",
        "giay to",
        "ho so",
        "no",
        "cai do",
    )

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.source_formatter = SourceFormatter()

    def _last_context(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_state = state.get("runtime_context") or {}
        return runtime_state.get("last_resolved_context") or {}

    def _has_source_switch_signal(self, query: str) -> bool:
        normalized = _normalize_text(query)
        return any(term in normalized for term in self.source_switch_terms)

    def _looks_like_follow_up(self, query: str) -> bool:
        normalized = _normalize_text(query)
        return len(normalized.split()) <= 7 or any(term in normalized for term in self.follow_up_terms)

    def _can_reuse_last_filter(self, state: dict[str, Any]) -> bool:
        last_context = self._last_context(state)
        if not last_context.get("filter"):
            return False
        original_query = state.get("original_query") or ""
        final_query = state.get("final_query") or original_query
        if self._has_source_switch_signal(original_query) or self._has_source_switch_signal(final_query):
            return False
        return bool(state.get("was_rewritten")) or self._looks_like_follow_up(original_query)

    def _scope_from_action(self, action: str) -> str:
        return {
            ACTION_REUSE_LAST_FILTER: RETRIEVAL_SCOPE_AUTO,
            ACTION_RESOLVE_SYSTEM_PROCEDURE: RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
            ACTION_RESOLVE_CURRENT_UPLOAD: RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
            ACTION_RESOLVE_PREVIOUS_UPLOAD: RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
            ACTION_RESOLVE_USER_FILE_NAME: RETRIEVAL_SCOPE_USER_FILE_NAME,
            ACTION_SEMANTIC_DOCUMENT_SEARCH: RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
            ACTION_MIXED_RETRIEVAL: RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
            ACTION_NEED_CLARIFICATION: RETRIEVAL_SCOPE_NEED_CLARIFICATION,
            ACTION_GENERAL_QUERY: RETRIEVAL_SCOPE_GENERAL_QUERY,
        }.get(action, RETRIEVAL_SCOPE_AUTO)

    def _targets(self, scope_resolution: dict[str, Any]) -> list[dict[str, Any]]:
        targets = scope_resolution.get("targets")
        return targets if isinstance(targets, list) else []

    def _target_for_source(self, scope_resolution: dict[str, Any], source_type: str) -> dict[str, Any]:
        for target in self._targets(scope_resolution):
            if isinstance(target, dict) and target.get("source_type") == source_type:
                return target
        return {}

    def _last_scope_fallback(self, state: dict[str, Any] | None = None) -> str:
        runtime_state = (state or {}).get("runtime_context") or {}
        return runtime_state.get("last_scope") or self._last_context(state or {}).get("scope") or RETRIEVAL_SCOPE_AUTO

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

    @traceable(name="rag_load_context_node")
    def load_context_node(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_context = dict(state.get("runtime_context") or {})
        runtime_context["current_user_id"] = state["user_id"]
        runtime_context["current_session_id"] = state.get("session_id")
        runtime_context.setdefault("active_document_ids", [])
        runtime_context.setdefault("current_session_docs", [])
        return {
            "runtime_context": runtime_context,
            "selected_document_ids": state.get("selected_document_ids") or [],
        }

    @traceable(name="rag_rewrite_detector_node")
    def rewrite_detector_node(self, state: dict[str, Any]) -> dict[str, Any]:
        decision = self.pipeline.rewrite_gate.decide(
            original_query=state["original_query"],
            conversation_state=state.get("runtime_context") or {},
        )
        return {"rewrite_gate": decision.model_dump()}

    @traceable(name="rag_rewrite_query_node")
    def rewrite_query_node(self, state: dict[str, Any]) -> dict[str, Any]:
        rewrite = self.pipeline.query_rewriter.rewrite(
            question=state["original_query"],
            intent_resolution=state.get("intent_resolution") or {},
            scope_resolution=state.get("scope_resolution") or {},
            conversation_state=state.get("runtime_context") or {},
        )
        payload = rewrite.model_dump()
        return {
            "query_rewrite": payload,
            "rewritten_query": rewrite.rewritten_question,
            "final_query": rewrite.rewritten_question,
            "was_rewritten": rewrite.was_rewritten,
        }

    @traceable(name="rag_use_original_query_node")
    def use_original_query_node(self, state: dict[str, Any]) -> dict[str, Any]:
        rewrite = QueryRewrite(
            original_question=state["original_query"],
            rewritten_question=state["original_query"],
            was_rewritten=False,
            reason="Rewrite gate decided query does not need rewrite.",
            stage="pre_intent_gate",
            used_llm=False,
        )
        return {
            "query_rewrite": rewrite.model_dump(),
            "rewritten_query": None,
            "final_query": state["original_query"],
            "was_rewritten": False,
        }

    @traceable(name="rag_intent_router_node")
    def intent_router_node(self, state: dict[str, Any]) -> dict[str, Any]:
        resolution = self.pipeline.intent_router.route(
            question=state["final_query"],
            conversation_state=state.get("runtime_context") or {},
        )
        intent_resolution = resolution.model_dump()
        scope_resolution = {
            "scope": intent_resolution.get("scope"),
            "targets": intent_resolution.get("targets") or [],
            "confidence": intent_resolution.get("confidence", 0.0),
        }
        if intent_resolution.get("action"):
            scope_resolution["action"] = intent_resolution["action"]
        retrieval_plan = dict(state.get("retrieval_plan") or {})
        if scope_resolution.get("action"):
            retrieval_plan["action"] = scope_resolution["action"]
        else:
            retrieval_plan.pop("action", None)
        retrieval_plan["target_scope"] = scope_resolution["scope"]
        retrieval_plan["scope_resolution"] = scope_resolution
        return {
            "intent_resolution": intent_resolution,
            "scope_resolution": scope_resolution,
            "metadata_filter": {},
            "retrieval_plan": retrieval_plan,
        }

    def _reuse_last_scope_resolution(self, state: dict[str, Any]) -> Any:
        from app.rag.graph.scope.scope_analyzer import ScopeResolution

        last_context = self._last_context(state)
        runtime_state = state.get("runtime_context") or {}
        return ScopeResolution(
            scope=last_context.get("scope") or runtime_state.get("last_scope") or RETRIEVAL_SCOPE_AUTO,
            should_reuse_last_filter=True,
            source_type=last_context.get("source_type") or "none",
            procedure_title_hint=last_context.get("procedure_title") or runtime_state.get("last_procedure_title"),
            document_name_hint=last_context.get("filename") or runtime_state.get("last_filename"),
            document_id_hint=last_context.get("document_id"),
            resolution_mode="reuse_last_context",
            confidence=0.95,
            reason="Reused last resolved context filter.",
        )

    @traceable(name="rag_scope_analyzer_node")
    def scope_analyzer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        from app.rag.graph.scope.scope_analyzer import ScopeAnalyzer

        retrieval_plan = dict(state.get("retrieval_plan") or {})
        if self._can_reuse_last_filter(state):
            structured_resolution = self._reuse_last_scope_resolution(state)
        else:
            structured_resolution = ScopeAnalyzer().resolve(state)

        scope_resolution = structured_resolution.model_dump()
        retrieval_plan["target_scope"] = structured_resolution.scope
        retrieval_plan["action"] = scope_resolution.get("action")
        retrieval_plan["scope_resolution"] = scope_resolution
        return {
            "scope_resolution": scope_resolution,
            "metadata_filter": {},
            "retrieval_plan": retrieval_plan,
        }

    @traceable(name="rag_document_resolver_node")
    async def document_resolver_node(self, state: dict[str, Any]) -> dict[str, Any]:
        scope_resolution = state.get("scope_resolution") or {}
        action = scope_resolution.get("action")
        if action == "reuse_last_filter":
            last_context = self._last_context(state)
            selected_ids = [doc_id for doc_id in [last_context.get("document_id")] if doc_id]
            resolution = DocumentResolution(
                metadata_filter=scope_resolution.get("metadata_filter") or state.get("metadata_filter") or {},
                selected_document_ids=selected_ids,
                resolved_documents=[],
                reason="Reused last resolved context.",
            )
        else:
            query = state.get("final_query") or state.get("original_query") or ""
            system_target = self._target_for_source(scope_resolution, "system")
            user_target = self._target_for_source(scope_resolution, "user_upload")
            detected_procedure_title = (
                system_target.get("procedure_title_hint")
                or self._procedure_hint(query)
                or scope_resolution.get("procedure_title_hint")
            )
            detected_filename = user_target.get("document_name_hint") or self._filename_hint(query) or scope_resolution.get("document_name_hint")
            time_hint = user_target.get("time_hint")
            resolution = await self.pipeline.document_resolver.resolve(
                scope=scope_resolution.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION,
                metadata_filter={},
                user_id=state["user_id"],
                session_id=state.get("session_id"),
                detected_filename=detected_filename,
                detected_procedure_title=detected_procedure_title,
                time_hint=time_hint,
                selected_document_ids=state.get("selected_document_ids") or [],
                conversation_state=state.get("runtime_context") or {},
            )
        return {
            "document_resolution": resolution.model_dump(),
            "document_candidates": resolution.resolved_documents,
            "metadata_filter": resolution.metadata_filter,
        }

    @traceable(name="rag_candidate_selector_node")
    def candidate_selector_node(self, state: dict[str, Any]) -> dict[str, Any]:
        candidates = state.get("document_candidates") or []
        document_resolution = dict(state.get("document_resolution") or {})
        scope_resolution = state.get("scope_resolution") or {}
        scope = scope_resolution.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION
        if len(candidates) <= 1:
            selection = CandidateSelection(
                selected_document_ids=document_resolution.get("selected_document_ids", []),
                selected_documents=candidates,
                confident=True,
                needs_clarification=False,
                reason="Zero or one candidate; no disambiguation needed.",
            )
        elif scope in {"current_uploads_only", "past_uploads_only", "user_uploads_all", "mixed"}:
            selection = CandidateSelection(
                selected_document_ids=document_resolution.get("selected_document_ids", []),
                selected_documents=candidates[:10],
                confident=True,
                needs_clarification=False,
                reason="Multiple upload documents matched the requested time/session scope; continue retrieval across them.",
            )
        else:
            selection = CandidateSelection(
                selected_document_ids=document_resolution.get("selected_document_ids", []),
                selected_documents=candidates[:5],
                confident=False,
                needs_clarification=True,
                reason="Multiple candidate documents need user clarification.",
            )
        return {"candidate_selection": selection.model_dump()}

    @traceable(name="rag_build_filter_node")
    def build_filter_node(self, state: dict[str, Any]) -> dict[str, Any]:
        scope_resolution = state.get("scope_resolution") or {}
        document_resolution = dict(state.get("document_resolution") or {})
        selected_document_ids = document_resolution.get("selected_document_ids") or state.get("selected_document_ids") or []
        resolved_documents = document_resolution.get("resolved_documents") or []
        first_document = resolved_documents[0] if resolved_documents else {}
        system_target = self._target_for_source(scope_resolution, "system")
        user_target = self._target_for_source(scope_resolution, "user_upload")
        procedure_title = (
            first_document.get("procedure_title")
            or system_target.get("procedure_title_hint")
            or scope_resolution.get("procedure_title_hint")
        )
        filename = (
            first_document.get("filename")
            or user_target.get("document_name_hint")
            or scope_resolution.get("document_name_hint")
        )

        if scope_resolution.get("action") == "reuse_last_filter":
            last_filter = self._last_context(state).get("filter") or {}
            metadata_filter = last_filter
        else:
            metadata_filter = build_retrieval_filter(
                scope=scope_resolution.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION,
                user_id=state["user_id"],
                session_id=state.get("session_id"),
                selected_document_ids=selected_document_ids,
                procedure_title=procedure_title,
                filename=filename,
            )
        document_resolution["metadata_filter"] = metadata_filter
        return {"metadata_filter": metadata_filter, "document_resolution": document_resolution}

    @traceable(name="rag_retrieval_strategy_node")
    def retrieval_strategy_node(self, state: dict[str, Any]) -> dict[str, Any]:
        scope_resolution = state.get("scope_resolution") or {}
        document_resolution = state.get("document_resolution") or {}
        strategy_plan = self.pipeline.retrieval_strategy.plan(
            rewritten_question=state["final_query"],
            intent_resolution=state.get("intent_resolution") or {},
            scope=scope_resolution.get("scope") or RETRIEVAL_SCOPE_NEED_CLARIFICATION,
            metadata_filter=document_resolution.get("metadata_filter") or state.get("metadata_filter") or {},
        )
        payload = strategy_plan.model_dump()
        payload["target_scope"] = scope_resolution.get("scope")
        return {"retrieval_plan": payload}

    def retrieval_node(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("retrieval_plan") or {}
        branch_results: list[dict[str, Any]] = []
        for branch in plan.get("branches", []):
            contexts = self.pipeline.retriever.retrieve(
                question=branch["query"],
                where_filter=branch["metadata_filter"],
                top_k=branch["top_k"],
            )
            branch_results.append(
                {
                    "name": branch["name"],
                    "metadata_filter": branch["metadata_filter"],
                    "contexts": contexts,
                }
            )
        return {"branch_results": branch_results}

    def evidence_validation_node(self, state: dict[str, Any]) -> dict[str, Any]:
        validation = self.pipeline.context_validator.validate_all(state.get("branch_results") or [])
        warnings = list(validation.warnings)
        plan = state.get("retrieval_plan") or {}
        mixed_warnings: list[str] = []
        if plan.get("mode") == "hybrid_compare":
            branch_names = {result["name"]: result for result in state.get("branch_results", [])}
            system_validation = self.pipeline.context_validator.validate_branch(
                branch_names.get("system_chunks", {}).get("contexts", []),
                branch_names.get("system_chunks", {}).get("metadata_filter", {"source_type": SOURCE_TYPE_SYSTEM}),
            )
            user_validation = self.pipeline.context_validator.validate_branch(
                branch_names.get("user_upload_chunks", {}).get("contexts", []),
                branch_names.get("user_upload_chunks", {}).get("metadata_filter", {"source_type": SOURCE_TYPE_USER_UPLOAD}),
            )
            if not system_validation.should_answer:
                mixed_warnings.append("Chưa tìm thấy thông tin tương ứng trong tài liệu hệ thống.")
            if not user_validation.should_answer:
                mixed_warnings.append("Chưa tìm thấy thông tin tương ứng trong tài liệu bạn upload.")
            warnings.extend(mixed_warnings)
        payload = validation.model_dump()
        payload["warnings"] = warnings
        return {
            "context_validation": payload,
            "raw_contexts": validation.contexts,
            "mixed_branch_warnings": mixed_warnings,
        }

    def _sources_from_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            SourceItem(
                document_id=item["metadata"].get("document_id", ""),
                chunk_id=item["metadata"].get("chunk_id", item.get("id", "")),
                filename=item["metadata"].get("filename", ""),
                source_type=item["metadata"].get("source_type", ""),
                procedure_title=item["metadata"].get("procedure_title"),
                page_number=item["metadata"].get("page_number"),
                page_source=item["metadata"].get("page_source"),
                section_title=item["metadata"].get("section_title"),
                score=item.get("similarity"),
                visibility=item["metadata"].get("visibility"),
                owner_user_id=item["metadata"].get("owner_user_id"),
                session_id=item["metadata"].get("session_id"),
            ).model_dump()
            for item in contexts
        ]

    def answer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        contexts = state.get("raw_contexts") or []
        intent = state.get("intent_resolution") or {}
        answer = self.pipeline.llm.generate_answer(
            question=state["final_query"],
            contexts=contexts,
            answer_style=intent.get("answer_style", "short_answer"),
        )
        if state.get("mixed_branch_warnings"):
            answer = answer + "\n\n" + "\n".join(state["mixed_branch_warnings"])
        sources = self._sources_from_contexts(contexts)
        return {"answer": self.source_formatter.format_answer(answer, sources), "sources": sources}

    @traceable(name="rag_no_context_node")
    def no_context_node(self, state: dict[str, Any]) -> dict[str, Any]:
        answer = (state.get("context_validation") or {}).get("fallback_answer") or FALLBACK_NO_CONTEXT
        if state.get("mixed_branch_warnings"):
            answer = answer + "\n\n" + "\n".join(state["mixed_branch_warnings"])
        return {"answer": answer, "sources": []}

    @traceable(name="rag_direct_answer_node")
    def direct_answer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "Mình là chatbot hỗ trợ hỏi đáp tài liệu hành chính. Bạn có thể hỏi về tài liệu hệ thống hoặc file bạn đã upload.",
            "sources": [],
            "raw_contexts": [],
            "context_validation": {"contexts": [], "should_answer": False, "fallback_answer": None, "warnings": [], "rejected_count": 0},
        }

    @traceable(name="rag_clarification_node")
    def clarification_node(self, state: dict[str, Any]) -> dict[str, Any]:
        candidates = (state.get("candidate_selection") or {}).get("selected_documents") or state.get("document_candidates") or []
        if candidates:
            names = [doc.get("filename") or doc.get("procedure_title") or doc.get("document_id") for doc in candidates[:5]]
            answer = "Mình tìm thấy nhiều tài liệu phù hợp. Bạn muốn hỏi tài liệu nào: " + ", ".join([name for name in names if name])
        else:
            answer = "Mình cần bạn làm rõ tài liệu muốn hỏi: file vừa upload, file cũ, tài liệu hệ thống, hoặc một file cụ thể."
        return {
            "answer": answer,
            "sources": [],
            "raw_contexts": [],
            "context_validation": {"contexts": [], "should_answer": False, "fallback_answer": answer, "warnings": [], "rejected_count": 0},
        }

    @traceable(name="rag_unsupported_node")
    def unsupported_node(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "Mình chưa hỗ trợ yêu cầu này trong pipeline RAG hiện tại.",
            "sources": [],
            "raw_contexts": [],
            "context_validation": {"contexts": [], "should_answer": False, "fallback_answer": None, "warnings": [], "rejected_count": 0},
        }

    @traceable(name="rag_update_state_node")
    def update_state_node(self, state: dict[str, Any]) -> dict[str, Any]:
        return {}

    def route_after_rewrite_gate(self, state: dict[str, Any]) -> str:
        return "rewrite_query" if (state.get("rewrite_gate") or {}).get("needs_rewrite") else "use_original_query"

    def route_after_intent(self, state: dict[str, Any]) -> str:
        intent_resolution = state.get("intent_resolution") or {}
        scope_resolution = state.get("scope_resolution") or {}
        intent = intent_resolution.get("intent")
        action = scope_resolution.get("action") or intent_resolution.get("action")
        scope = scope_resolution.get("scope") or intent_resolution.get("scope")
        if intent == INTENT_UNSUPPORTED:
            return "unsupported"
        if intent == INTENT_NEED_CLARIFICATION or action == "need_clarification" or scope in {"need_clarification", RETRIEVAL_SCOPE_NEED_CLARIFICATION}:
            return "clarification"
        if intent == INTENT_GENERAL_QUERY or not intent_resolution.get("needs_retrieval", True):
            return "direct_answer"
        if action == "reuse_last_filter":
            return "build_filter"
        if intent_resolution.get("is_follow_up"):
            return "rewrite_query"
        return "document_resolver"

    def route_after_scope_resolution(self, state: dict[str, Any]) -> str:
        scope_resolution = state.get("scope_resolution") or {}
        action = scope_resolution.get("action")
        scope = scope_resolution.get("scope")
        if action == "need_clarification" or scope in {"need_clarification", RETRIEVAL_SCOPE_NEED_CLARIFICATION}:
            return "clarification"
        if action == "reuse_last_filter":
            return "build_filter"
        return "document_resolver"

    def route_after_candidate_selector(self, state: dict[str, Any]) -> str:
        selection = state.get("candidate_selection") or {}
        document_resolution = state.get("document_resolution") or {}
        if selection.get("needs_clarification") or document_resolution.get("needs_clarification"):
            return "clarification"
        return "build_filter"

    def route_after_evidence_validation(self, state: dict[str, Any]) -> str:
        if (state.get("context_validation") or {}).get("should_answer"):
            return "answer"
        return "no_context"
