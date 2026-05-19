from datetime import datetime
from typing import Any, Optional

from app.db.mongodb import get_database
from app.models.document import DocumentModel


class DocumentRepository:
    collection_name = "documents"

    def _collection(self):
        return get_database()[self.collection_name]

    async def create_document(self, document: DocumentModel) -> str:
        payload = document.model_dump(by_alias=True)
        await self._collection().insert_one(payload)
        return document.id

    async def get_document_by_id(self, document_id: str) -> Optional[dict[str, Any]]:
        return await self._collection().find_one({"_id": document_id})

    async def list_documents(self, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if owner_user_id is not None:
            query["owner_user_id"] = owner_user_id
        cursor = self._collection().find(query)
        return [doc async for doc in cursor]

    async def list_user_documents(self, user_id: str) -> list[dict[str, Any]]:
        return await self.list_documents(owner_user_id=user_id)

    async def update_document_status(self, document_id: str, status: str) -> None:
        await self._collection().update_one(
            {"_id": document_id},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}},
        )

    async def update_markdown_path(self, document_id: str, markdown_path: str) -> None:
        await self._collection().update_one(
            {"_id": document_id},
            {"$set": {"markdown_storage_path": markdown_path, "updated_at": datetime.utcnow()}},
        )

    async def update_document_fields(self, document_id: str, **fields: Any) -> None:
        payload = dict(fields)
        payload["updated_at"] = datetime.utcnow()
        await self._collection().update_one({"_id": document_id}, {"$set": payload})

    async def soft_delete_document(self, document_id: str) -> None:
        await self._collection().update_one(
            {"_id": document_id},
            {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}},
        )
