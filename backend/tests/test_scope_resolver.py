from app.core.constants import (
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_GENERAL_QUERY,
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    RETRIEVAL_SCOPE_SYSTEM_DOCS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
)
from app.rag.retrieval.scope_resolver import ScopeResolver


def test_resolve_system_procedure_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question="Thủ tục công bố cơ sở đủ điều kiện sản xuất chế phẩm diệt côn trùng là gì?",
        user_id="user_1",
    )

    assert resolution.scope == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE
    assert resolution.should_retrieve is True
    assert resolution.metadata_filter["source_type"] == "system"
    assert resolution.metadata_filter["visibility"] == "global"
    assert resolution.metadata_filter["procedure_title"]


def test_resolve_current_session_upload_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question="File này nói gì?",
        user_id="user_1",
        session_id="session_1",
    )

    assert resolution.scope == RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS
    assert resolution.metadata_filter["source_type"] == "user_upload"
    assert resolution.metadata_filter["owner_user_id"] == "user_1"
    assert resolution.metadata_filter["session_id"] == "session_1"


def test_resolve_file_name_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question='File hoc_phi_2024.pdf nói gì?',
        user_id="user_1",
    )

    assert resolution.scope == RETRIEVAL_SCOPE_USER_FILE_NAME
    assert resolution.metadata_filter["source_type"] == "user_upload"
    assert resolution.metadata_filter["owner_user_id"] == "user_1"
    assert resolution.metadata_filter["filename"] == "hoc_phi_2024.pdf"


def test_resolve_compare_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question="So sánh tài liệu hệ thống với file của tôi",
        user_id="user_1",
    )

    assert resolution.scope == RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER
    assert "$or" in resolution.metadata_filter


def test_resolve_general_query_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question="Giải thích khái niệm embedding là gì",
        user_id="user_1",
    )

    assert resolution.scope == RETRIEVAL_SCOPE_GENERAL_QUERY
    assert resolution.should_retrieve is False
    assert resolution.metadata_filter == {}


def test_resolve_follow_up_uses_last_scope():
    resolver = ScopeResolver()

    resolution = resolver.resolve(
        question="Thế thời hạn bao lâu?",
        user_id="user_1",
        conversation_state={
            "last_scope": RETRIEVAL_SCOPE_SYSTEM_DOCS,
            "last_procedure_title": None,
            "last_filename": None,
        },
    )

    assert resolution.scope == RETRIEVAL_SCOPE_SYSTEM_DOCS
    assert resolution.should_retrieve is True
