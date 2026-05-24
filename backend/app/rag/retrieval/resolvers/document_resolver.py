from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.constants import (
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_CURRENT_UPLOAD,
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
    RETRIEVAL_SCOPE_USER_FILE_NAME,
    SOURCE_TYPE_SYSTEM,
    SOURCE_TYPE_USER_UPLOAD,
    VISIBILITY_GLOBAL,
)
from app.repositories.document_repository import DocumentRepository


def _and(*conditions: dict[str, Any]) -> dict[str, Any]:
    clean_conditions = [condition for condition in conditions if condition]
    if not clean_conditions:
        return {}
    if len(clean_conditions) == 1:
        return clean_conditions[0]
    return {"$and": clean_conditions}


@dataclass
class DocumentResolution:
    metadata_filter: dict[str, Any]
    selected_document_ids: list[str] = field(default_factory=list)
    resolved_documents: list[dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class DocumentResolver:
    def __init__(self, document_repository: DocumentRepository | None = None) -> None:
        self.document_repository = document_repository or DocumentRepository()

    def _with_document_filter(self, metadata_filter: dict[str, Any], document_ids: list[str]) -> dict[str, Any]:
        document_ids = [document_id for document_id in document_ids if document_id]
        if not document_ids:
            return metadata_filter
        document_filter = {"document_id": {"$in": document_ids}}
        if metadata_filter:
            return _and(metadata_filter, document_filter)
        return document_filter

    def _serialize_docs(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "document_id": doc.get("_id"),
                "filename": doc.get("filename"),
                "source_type": doc.get("source_type"),
                "owner_user_id": doc.get("owner_user_id"),
                "session_id": doc.get("uploaded_in_session_id"),
                "procedure_title": doc.get("procedure_title"),
                "visibility": doc.get("visibility"),
                "created_at": doc.get("created_at"),
            }
            for doc in documents
        ]

    def _is_authorized_selected_document(
        self,
        document: dict[str, Any],
        scope: str,
        user_id: str,
        session_id: str | None,
    ) -> bool:
        source_type = document.get("source_type")
        if source_type == SOURCE_TYPE_SYSTEM:
            return document.get("visibility") == VISIBILITY_GLOBAL

        if source_type != SOURCE_TYPE_USER_UPLOAD or document.get("owner_user_id") != user_id:
            return False

        if scope in {RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS, RETRIEVAL_SCOPE_CURRENT_UPLOAD}:
            return bool(session_id) and document.get("uploaded_in_session_id") == session_id

        return scope in {
            RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
            RETRIEVAL_SCOPE_USER_FILE_NAME,
            RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
        }

    async def _resolve_selected_documents(
        self,
        scope: str,
        user_id: str,
        session_id: str | None,
        selected_document_ids: list[str],
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for document_id in dict.fromkeys([doc_id for doc_id in selected_document_ids if doc_id]):
            document = await self.document_repository.get_document_by_id(document_id)
            if document and self._is_authorized_selected_document(document, scope, user_id, session_id):
                documents.append(document)
        return documents

    async def resolve(
        self,
        scope: str,
        metadata_filter: dict[str, Any],
        user_id: str,
        session_id: str | None = None,
        detected_filename: str | None = None,
        detected_procedure_title: str | None = None,
        selected_document_ids: list[str] | None = None,
        conversation_state: dict[str, Any] | None = None,
    ) -> DocumentResolution:
        conversation_state = conversation_state or {}
        selected_document_ids = selected_document_ids or []

        if selected_document_ids:
            documents = await self._resolve_selected_documents(scope, user_id, session_id, selected_document_ids)
            authorized_document_ids = [doc["_id"] for doc in documents if doc.get("_id")]
            return DocumentResolution(
                metadata_filter=self._with_document_filter(metadata_filter, authorized_document_ids),
                selected_document_ids=authorized_document_ids,
                resolved_documents=self._serialize_docs(documents),
                needs_clarification=not authorized_document_ids,
                reason="explicit selected document ids after authorization check",
            )

        if scope == RETRIEVAL_SCOPE_SYSTEM_PROCEDURE and detected_procedure_title:
            documents = await self.document_repository.find_system_documents_by_procedure_title(detected_procedure_title)
            document_ids = [doc["_id"] for doc in documents if doc.get("_id")]
            return DocumentResolution(
                metadata_filter=self._with_document_filter(metadata_filter, document_ids),
                selected_document_ids=document_ids,
                resolved_documents=self._serialize_docs(documents),
                needs_clarification=len(documents) > 1,
                reason="matched system procedure title",
            )

        if scope == RETRIEVAL_SCOPE_USER_FILE_NAME and detected_filename:
            documents = await self.document_repository.find_user_documents_by_filename(user_id, detected_filename)
            document_ids = [doc["_id"] for doc in documents if doc.get("_id")]
            return DocumentResolution(
                metadata_filter=self._with_document_filter(metadata_filter, document_ids),
                selected_document_ids=document_ids,
                resolved_documents=self._serialize_docs(documents),
                needs_clarification=len(documents) > 1,
                reason="matched uploaded filename",
            )

        if scope in {RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS, RETRIEVAL_SCOPE_CURRENT_UPLOAD} and session_id:
            documents = await self.document_repository.list_user_documents_by_session(user_id, session_id)
            if not documents and conversation_state.get("current_session_docs"):
                document_ids = [doc_id for doc_id in conversation_state["current_session_docs"] if doc_id]
                return DocumentResolution(
                    metadata_filter=self._with_document_filter(metadata_filter, document_ids),
                    selected_document_ids=document_ids,
                    reason="used current session docs from conversation state",
                )
            document_ids = [doc["_id"] for doc in documents if doc.get("_id")]
            return DocumentResolution(
                metadata_filter=self._with_document_filter(metadata_filter, document_ids),
                selected_document_ids=document_ids,
                resolved_documents=self._serialize_docs(documents),
                needs_clarification=len(documents) > 1,
                reason="matched current session uploads",
            )

        if scope == RETRIEVAL_SCOPE_USER_ALL_UPLOADS:
            last_document = conversation_state.get("last_referenced_doc") or {}
            last_document_id = last_document.get("document_id") if isinstance(last_document, dict) else None
            if last_document_id:
                return DocumentResolution(
                    metadata_filter=self._with_document_filter(metadata_filter, [last_document_id]),
                    selected_document_ids=[last_document_id],
                    reason="used last referenced document",
                )
            documents = await self.document_repository.list_user_ready_documents(user_id)
            document_ids = [doc["_id"] for doc in documents[:1] if doc.get("_id")]
            return DocumentResolution(
                metadata_filter=self._with_document_filter(metadata_filter, document_ids),
                selected_document_ids=document_ids,
                resolved_documents=self._serialize_docs(documents[:1]),
                reason="used latest user upload",
            )

        return DocumentResolution(metadata_filter=metadata_filter, reason="scope does not require a specific document")
