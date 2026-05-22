from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.constants import (
    RETRIEVAL_SCOPE_ALL_USER_UPLOADS,
    RETRIEVAL_SCOPE_AUTO,
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_CURRENT_UPLOAD,
    RETRIEVAL_SCOPE_GENERAL_QUERY,
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    RETRIEVAL_SCOPE_MIXED,
    RETRIEVAL_SCOPE_NEED_CLARIFICATION,
    RETRIEVAL_SCOPE_SYSTEM_DOCS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
    SOURCE_TYPE_SYSTEM,
    SOURCE_TYPE_USER_UPLOAD,
    VISIBILITY_GLOBAL,
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()


def _strip_quotes(value: str) -> str:
    return value.strip(" \t\r\n\"'“”‘’`")


def _and(*conditions: dict[str, Any]) -> dict[str, Any]:
    clean_conditions = [condition for condition in conditions if condition]
    if not clean_conditions:
        return {}
    if len(clean_conditions) == 1:
        return clean_conditions[0]
    return {"$and": clean_conditions}


def _or(*conditions: dict[str, Any]) -> dict[str, Any]:
    clean_conditions = [condition for condition in conditions if condition]
    if not clean_conditions:
        return {}
    if len(clean_conditions) == 1:
        return clean_conditions[0]
    return {"$or": clean_conditions}


@dataclass
class ScopeResolution:
    scope: str
    metadata_filter: dict[str, Any]
    should_retrieve: bool = True
    detected_procedure_title: str | None = None
    detected_filename: str | None = None
    matched_rules: list[str] = field(default_factory=list)
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ScopeResolver:
    _current_upload_patterns = (
        "file nay",
        "tai lieu nay",
        "file vua upload",
        "tai lieu vua upload",
        "file moi upload",
        "tai lieu hien tai",
        "trong session nay",
        "session nay",
    )
    _user_history_patterns = (
        "file cu",
        "file toi upload truoc do",
        "file da upload",
        "tai lieu lan truoc",
        "hom qua toi upload",
        "tai lieu da tai len",
        "file truoc do",
    )
    _compare_patterns = (
        "so sanh",
        "doi chieu",
        "khac nhau",
        "giong nhau",
        "dap ung",
        "doi voi file cua toi",
    )
    _system_general_patterns = (
        "thu tuc",
        "quy trinh",
        "ho so",
        "quy dinh",
        "quy che",
        "thong tu",
        "nghi dinh",
        "quyet dinh",
    )
    _follow_up_patterns = (
        "the",
        "con",
        "vay",
        "thi sao",
        "bao lau",
        "le phi",
        "chi phi",
        "phi",
        "tiep theo",
        "phia tren",
        "no",
        "noi do",
        "muc do",
    )
    _filename_pattern = re.compile(
        r"""(?ix)
        (?:
            (?:file|tai\s+lieu|document)\s*
            (?:nay|nay\s+co|co\s+ten|ten\s+la|la)?\s*
            (?:[:\-]\s*)?
        )?
        ["'“”‘’`]?
        (?P<filename>[a-z0-9][a-z0-9_\-\s().\[\]]*\.(?:pdf|docx?|xlsx?|pptx?|txt|md))
        ["'“”‘’`]?
        """
    )
    _procedure_title_pattern = re.compile(
        r"""(?ix)
        ^\s*
        (?P<title>.+?)
        \s*(?:la\s+gi|nhu\s+the\s+nao|ra\s+sao|bao\s+lau|o\s+dau|can\s+gi|co\s+gi|nao)\s*\??\s*$
        """
    )

    def _contains_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    def _detect_filename(self, question: str) -> str | None:
        match = self._filename_pattern.search(question)
        if not match:
            return None
        filename = _strip_quotes(match.group("filename"))
        filename = re.sub(r"\s+", " ", filename).strip()
        return filename or None

    def _detect_procedure_title(self, question: str) -> str | None:
        normalized = _normalize_text(question)
        if not self._contains_any(normalized, self._system_general_patterns):
            return None

        candidate = question
        for keyword in ("thủ tục", "quy trình", "hồ sơ", "quy định", "procedure"):
            keyword_normalized = _normalize_text(keyword)
            if keyword_normalized in normalized:
                index = normalized.find(keyword_normalized)
                candidate = question[index + len(keyword) :]
                break

        candidate = _strip_quotes(candidate)
        candidate = re.sub(r"^\s*[:\-]\s*", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            return None

        match = self._procedure_title_pattern.match(candidate)
        if match:
            candidate = match.group("title")

        candidate = re.sub(r"\s+", " ", candidate).strip()
        candidate_normalized = _normalize_text(candidate)

        if not candidate or candidate_normalized in {"thu tuc", "quy trinh", "ho so", "quy dinh"}:
            return None
        if candidate_normalized.endswith(("la gi", "nao", "nhu the nao", "ra sao")):
            return None
        return candidate

    def _looks_like_follow_up(self, normalized_question: str) -> bool:
        if len(normalized_question.split()) <= 6:
            return True
        return self._contains_any(normalized_question, self._follow_up_patterns)

    def _build_filter_for_scope(
        self,
        scope: str,
        user_id: str,
        session_id: str | None,
        selected_document_ids: list[str],
        detected_procedure_title: str | None = None,
        detected_filename: str | None = None,
    ) -> dict[str, Any]:
        selected_document_ids = [doc_id for doc_id in selected_document_ids if doc_id]

        if scope == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE:
            base: dict[str, Any] = {"source_type": SOURCE_TYPE_SYSTEM, "visibility": VISIBILITY_GLOBAL}
            if detected_procedure_title:
                base["procedure_title"] = detected_procedure_title
        elif scope in {RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS, RETRIEVAL_SCOPE_CURRENT_UPLOAD}:
            base = {"source_type": SOURCE_TYPE_USER_UPLOAD, "owner_user_id": user_id}
            if session_id:
                base["session_id"] = session_id
        elif scope in {RETRIEVAL_SCOPE_ALL_USER_UPLOADS, RETRIEVAL_SCOPE_USER_ALL_UPLOADS}:
            base = {"source_type": SOURCE_TYPE_USER_UPLOAD, "owner_user_id": user_id}
        elif scope == RETRIEVAL_SCOPE_USER_FILE_NAME:
            base = {"source_type": SOURCE_TYPE_USER_UPLOAD, "owner_user_id": user_id}
            if detected_filename:
                base["filename"] = detected_filename
        elif scope == RETRIEVAL_SCOPE_SYSTEM_DOCS:
            base = {"source_type": SOURCE_TYPE_SYSTEM, "visibility": VISIBILITY_GLOBAL}
        elif scope in {RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER, RETRIEVAL_SCOPE_MIXED}:
            system_filter = {"source_type": SOURCE_TYPE_SYSTEM, "visibility": VISIBILITY_GLOBAL}
            user_filter = {"source_type": SOURCE_TYPE_USER_UPLOAD, "owner_user_id": user_id}
            base = _or(system_filter, user_filter)
        elif scope in {RETRIEVAL_SCOPE_GENERAL_QUERY, RETRIEVAL_SCOPE_NEED_CLARIFICATION}:
            base = {}
        else:
            base = {"source_type": SOURCE_TYPE_SYSTEM, "visibility": VISIBILITY_GLOBAL}

        if selected_document_ids:
            if base:
                return _and(base, {"document_id": {"$in": selected_document_ids}})
            return {"document_id": {"$in": selected_document_ids}}
        return base

    def resolve(
        self,
        question: str,
        user_id: str,
        session_id: str | None = None,
        scope: str = RETRIEVAL_SCOPE_AUTO,
        selected_document_ids: list[str] | None = None,
        conversation_state: dict[str, Any] | None = None,
    ) -> ScopeResolution:
        conversation_state = conversation_state or {}
        selected_document_ids = selected_document_ids or []
        normalized_question = _normalize_text(question)

        last_scope = conversation_state.get("last_scope")
        last_procedure_title = conversation_state.get("last_procedure_title")
        last_filename = conversation_state.get("last_filename")

        detected_filename = self._detect_filename(question)
        detected_procedure_title = self._detect_procedure_title(question)

        matched_rules: list[str] = []

        if scope != RETRIEVAL_SCOPE_AUTO:
            metadata_filter = self._build_filter_for_scope(
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                selected_document_ids=selected_document_ids,
                detected_procedure_title=detected_procedure_title,
                detected_filename=detected_filename,
            )
            return ScopeResolution(
                scope=scope,
                metadata_filter=metadata_filter,
                should_retrieve=scope not in {RETRIEVAL_SCOPE_GENERAL_QUERY, RETRIEVAL_SCOPE_NEED_CLARIFICATION},
                detected_procedure_title=detected_procedure_title,
                detected_filename=detected_filename,
                matched_rules=["explicit_scope"],
                reason="explicit scope provided by request",
            )

        if detected_filename:
            scope_name = RETRIEVAL_SCOPE_USER_FILE_NAME
            matched_rules.append("detected_filename")
        elif detected_procedure_title:
            scope_name = RETRIEVAL_SCOPE_SYSTEM_PROCEDURE
            matched_rules.append("detected_procedure_title")
        elif self._contains_any(normalized_question, self._compare_patterns):
            scope_name = RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER
            matched_rules.append("compare_query")
        elif self._contains_any(normalized_question, self._current_upload_patterns):
            scope_name = RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS
            matched_rules.append("current_session_upload")
        elif self._contains_any(normalized_question, self._user_history_patterns):
            scope_name = RETRIEVAL_SCOPE_USER_ALL_UPLOADS
            matched_rules.append("user_history_upload")
        elif self._contains_any(normalized_question, self._system_general_patterns):
            scope_name = RETRIEVAL_SCOPE_SYSTEM_DOCS
            matched_rules.append("system_general")
        elif last_scope and self._looks_like_follow_up(normalized_question):
            scope_name = last_scope
            matched_rules.append("follow_up")
            detected_procedure_title = detected_procedure_title or last_procedure_title
            detected_filename = detected_filename or last_filename
        elif not self._contains_any(normalized_question, self._system_general_patterns) and not self._contains_any(
            normalized_question,
            self._current_upload_patterns + self._user_history_patterns + self._compare_patterns,
        ):
            scope_name = RETRIEVAL_SCOPE_GENERAL_QUERY
            matched_rules.append("general_query")
        else:
            scope_name = RETRIEVAL_SCOPE_NEED_CLARIFICATION
            matched_rules.append("ambiguous")

        if scope_name == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE and not detected_procedure_title and last_procedure_title:
            detected_procedure_title = last_procedure_title
        if scope_name == RETRIEVAL_SCOPE_USER_FILE_NAME and not detected_filename and last_filename:
            detected_filename = last_filename

        metadata_filter = self._build_filter_for_scope(
            scope=scope_name,
            user_id=user_id,
            session_id=session_id,
            selected_document_ids=selected_document_ids,
            detected_procedure_title=detected_procedure_title,
            detected_filename=detected_filename,
        )

        if scope_name == RETRIEVAL_SCOPE_GENERAL_QUERY:
            reason = "question does not appear to reference any document"
        elif scope_name == RETRIEVAL_SCOPE_NEED_CLARIFICATION:
            reason = "question is ambiguous and needs document clarification"
        elif scope_name == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE and detected_procedure_title:
            reason = f"detected procedure title: {detected_procedure_title}"
        elif scope_name == RETRIEVAL_SCOPE_USER_FILE_NAME and detected_filename:
            reason = f"detected filename: {detected_filename}"
        else:
            reason = "resolved from query heuristics"

        should_retrieve = scope_name not in {RETRIEVAL_SCOPE_GENERAL_QUERY, RETRIEVAL_SCOPE_NEED_CLARIFICATION}
        return ScopeResolution(
            scope=scope_name,
            metadata_filter=metadata_filter,
            should_retrieve=should_retrieve,
            detected_procedure_title=detected_procedure_title,
            detected_filename=detected_filename,
            matched_rules=matched_rules,
            reason=reason,
        )
