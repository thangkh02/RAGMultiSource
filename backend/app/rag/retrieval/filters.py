from app.core.constants import (
    RETRIEVAL_SCOPE_ALL_USER_UPLOADS,
    RETRIEVAL_SCOPE_CURRENT_UPLOAD,
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
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


def _and(*conditions: dict) -> dict:
    clean_conditions = [condition for condition in conditions if condition]
    if not clean_conditions:
        return {}
    if len(clean_conditions) == 1:
        return clean_conditions[0]
    return {"$and": clean_conditions}


def build_retrieval_filter(
    scope: str,
    user_id: str,
    session_id: str | None = None,
    selected_document_ids: list[str] | None = None,
    procedure_title: str | None = None,
    filename: str | None = None,
) -> dict:
    selected_document_ids = selected_document_ids or []

    if scope in {"current_uploads_only", RETRIEVAL_SCOPE_CURRENT_UPLOAD, RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS}:
        conditions = [
            {"source_type": SOURCE_TYPE_USER_UPLOAD},
            {"owner_user_id": user_id},
        ]
        if session_id:
            conditions.append({"session_id": session_id})
        base = _and(*conditions)
    elif scope in {"past_uploads_only", "user_uploads_all", RETRIEVAL_SCOPE_ALL_USER_UPLOADS, RETRIEVAL_SCOPE_USER_ALL_UPLOADS}:
        base = _and({"source_type": SOURCE_TYPE_USER_UPLOAD}, {"owner_user_id": user_id})
    elif scope == RETRIEVAL_SCOPE_USER_FILE_NAME:
        conditions = [
            {"source_type": SOURCE_TYPE_USER_UPLOAD},
            {"owner_user_id": user_id},
        ]
        if filename:
            conditions.append({"filename": filename})
        base = _and(*conditions)
    elif scope in {"system_only", RETRIEVAL_SCOPE_SYSTEM_DOCS, RETRIEVAL_SCOPE_SYSTEM_PROCEDURE}:
        base = _and({"source_type": SOURCE_TYPE_SYSTEM}, {"visibility": VISIBILITY_GLOBAL})
    elif scope == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE:
        conditions = [
            {"source_type": SOURCE_TYPE_SYSTEM},
            {"visibility": VISIBILITY_GLOBAL},
        ]
        if procedure_title:
            conditions.append({"procedure_title": procedure_title})
        base = _and(*conditions)
    elif scope in {"mixed", RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER, RETRIEVAL_SCOPE_MIXED}:
        base = {
            "$or": [
                _and({"source_type": SOURCE_TYPE_SYSTEM}, {"visibility": VISIBILITY_GLOBAL}),
                _and({"source_type": SOURCE_TYPE_USER_UPLOAD}, {"owner_user_id": user_id}),
            ]
        }
    elif scope in {"need_clarification", RETRIEVAL_SCOPE_GENERAL_QUERY, RETRIEVAL_SCOPE_NEED_CLARIFICATION}:
        base = {}
    else:
        base = {
            "$or": [
                _and({"source_type": SOURCE_TYPE_SYSTEM}, {"visibility": VISIBILITY_GLOBAL}),
                _and({"source_type": SOURCE_TYPE_USER_UPLOAD}, {"owner_user_id": user_id}),
            ]
        }

    if selected_document_ids:
        selection_filter = {"document_id": {"$in": selected_document_ids}}
        return _and(base, selection_filter)

    return base
