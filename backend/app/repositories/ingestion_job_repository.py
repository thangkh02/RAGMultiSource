from datetime import datetime
from typing import Any, Optional

from pymongo import ReturnDocument

from app.db.mongodb import get_database
from app.models.ingestion_job import IngestionJobModel


class IngestionJobRepository:
    collection_name = "ingestion_jobs"

    def _collection(self):
        return get_database()[self.collection_name]

    async def create_job(self, job: IngestionJobModel) -> str:
        await self._collection().insert_one(job.model_dump(by_alias=True))
        return job.id

    async def get_job_by_id(self, job_id: str) -> Optional[dict[str, Any]]:
        return await self._collection().find_one({"_id": job_id})

    async def claim_next_queued_job(self) -> Optional[dict[str, Any]]:
        now = datetime.utcnow()
        return await self._collection().find_one_and_update(
            {"status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "current_step": "uploaded",
                    "started_at": now,
                    "updated_at": now,
                    "progress": 5,
                }
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )

    async def list_jobs_by_document_id(self, document_id: str) -> list[dict[str, Any]]:
        cursor = self._collection().find({"document_id": document_id})
        return [job async for job in cursor]

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        current_step: str | None = None,
        error_message: str | None = None,
        progress: int | None = None,
        clear_error: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"status": status, "updated_at": datetime.utcnow()}
        if status == "running":
            payload["started_at"] = payload["updated_at"]
        if status in {"succeeded", "failed", "cancelled"}:
            payload["finished_at"] = payload["updated_at"]
        if current_step is not None:
            payload["current_step"] = current_step
        if error_message is not None:
            payload["error_message"] = error_message
        elif clear_error:
            payload["error_message"] = None
        if progress is not None:
            payload["progress"] = progress
        await self._collection().update_one({"_id": job_id}, {"$set": payload})

    async def delete_jobs_by_document_id(self, document_id: str) -> None:
        await self._collection().delete_many({"document_id": document_id})
