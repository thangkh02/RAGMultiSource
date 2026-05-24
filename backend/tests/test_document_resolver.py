import asyncio

from app.core.constants import (
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
)
from app.rag.retrieval.resolvers import ConversationStateManager, DocumentResolver


class FakeDocumentRepository:
    async def get_document_by_id(self, document_id: str):
        documents = {
            "doc_1": {
                "_id": "doc_1",
                "filename": "hoc_phi.pdf",
                "source_type": "user_upload",
                "owner_user_id": "user_1",
                "uploaded_in_session_id": "sess_1",
            },
            "doc_other": {
                "_id": "doc_other",
                "filename": "other.pdf",
                "source_type": "user_upload",
                "owner_user_id": "other_user",
                "uploaded_in_session_id": "sess_1",
            },
            "sysdoc_1": {
                "_id": "sysdoc_1",
                "filename": "system.docx",
                "source_type": "system",
                "visibility": "global",
                "procedure_title": "Procedure A",
            },
        }
        return documents.get(document_id)

    async def find_system_documents_by_procedure_title(self, procedure_title: str):
        return [
            {
                "_id": "sysdoc_1",
                "filename": "system.docx",
                "source_type": "system",
                "visibility": "global",
                "procedure_title": procedure_title,
            }
        ]

    async def find_user_documents_by_filename(self, user_id: str, filename: str):
        return [
            {
                "_id": "doc_1",
                "filename": filename,
                "source_type": "user_upload",
                "owner_user_id": user_id,
                "uploaded_in_session_id": "sess_1",
            }
        ]

    async def list_user_documents_by_session(self, user_id: str, session_id: str):
        return [
            {
                "_id": "doc_2",
                "filename": "latest.pdf",
                "source_type": "user_upload",
                "owner_user_id": user_id,
                "uploaded_in_session_id": session_id,
            }
        ]

    async def list_user_ready_documents(self, user_id: str):
        return [
            {
                "_id": "doc_3",
                "filename": "old.pdf",
                "source_type": "user_upload",
                "owner_user_id": user_id,
                "uploaded_in_session_id": "old_sess",
            }
        ]


def test_document_resolver_filters_system_procedure_by_document_id():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
            metadata_filter={"source_type": "system", "procedure_title": "Procedure A"},
            user_id="user_1",
            detected_procedure_title="Procedure A",
        )
    )

    assert resolution.selected_document_ids == ["sysdoc_1"]
    assert resolution.metadata_filter == {
        "$and": [
            {"source_type": "system", "procedure_title": "Procedure A"},
            {"document_id": {"$in": ["sysdoc_1"]}},
        ]
    }


def test_document_resolver_filters_user_file_by_filename():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_USER_FILE_NAME,
            metadata_filter={"source_type": "user_upload", "owner_user_id": "user_1", "filename": "hoc_phi.pdf"},
            user_id="user_1",
            detected_filename="hoc_phi.pdf",
        )
    )

    assert resolution.selected_document_ids == ["doc_1"]
    assert resolution.metadata_filter["$and"][1] == {"document_id": {"$in": ["doc_1"]}}


def test_document_resolver_authorizes_selected_document_ids():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_USER_FILE_NAME,
            metadata_filter={"source_type": "user_upload", "owner_user_id": "user_1"},
            user_id="user_1",
            selected_document_ids=["doc_1", "doc_other"],
        )
    )

    assert resolution.selected_document_ids == ["doc_1"]
    assert resolution.metadata_filter["$and"][1] == {"document_id": {"$in": ["doc_1"]}}


def test_document_resolver_rejects_selected_current_session_doc_from_other_session():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
            metadata_filter={"source_type": "user_upload", "owner_user_id": "user_1", "session_id": "sess_2"},
            user_id="user_1",
            session_id="sess_2",
            selected_document_ids=["doc_1"],
        )
    )

    assert resolution.selected_document_ids == []
    assert resolution.needs_clarification is True


def test_document_resolver_uses_current_session_uploads():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
            metadata_filter={"source_type": "user_upload", "owner_user_id": "user_1", "session_id": "sess_1"},
            user_id="user_1",
            session_id="sess_1",
        )
    )

    assert resolution.selected_document_ids == ["doc_2"]
    assert resolution.metadata_filter["$and"][1] == {"document_id": {"$in": ["doc_2"]}}


def test_document_resolver_uses_last_referenced_doc_for_old_upload_follow_up():
    resolver = DocumentResolver(FakeDocumentRepository())

    resolution = asyncio.run(
        resolver.resolve(
            scope=RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
            metadata_filter={"source_type": "user_upload", "owner_user_id": "user_1"},
            user_id="user_1",
            conversation_state={"last_referenced_doc": {"document_id": "doc_last"}},
        )
    )

    assert resolution.selected_document_ids == ["doc_last"]
    assert resolution.metadata_filter["$and"][1] == {"document_id": {"$in": ["doc_last"]}}


def test_conversation_state_updates_last_referenced_document():
    manager = ConversationStateManager()
    state = manager.load({"conversation_state": {}}, user_id="user_1", session_id="sess_1")

    next_state = manager.update_after_answer(
        state=state,
        intent="ask_question",
        scope=RETRIEVAL_SCOPE_USER_FILE_NAME,
        sources=[
            {
                "document_id": "doc_1",
                "filename": "hoc_phi.pdf",
                "source_type": "user_upload",
                "session_id": "sess_1",
            }
        ],
        selected_document_ids=["doc_1"],
        rewritten_question="File hoc_phi.pdf nói gì?",
        detected_filename="hoc_phi.pdf",
    )

    assert next_state["last_intent"] == "ask_question"
    assert next_state["last_scope"] == RETRIEVAL_SCOPE_USER_FILE_NAME
    assert next_state["last_filename"] == "hoc_phi.pdf"
    assert next_state["last_document_ids"] == ["doc_1"]
    assert next_state["last_rewritten_question"] == "File hoc_phi.pdf nói gì?"
    assert next_state["last_referenced_doc"]["document_id"] == "doc_1"
    assert next_state["current_session_docs"] == ["doc_1"]
