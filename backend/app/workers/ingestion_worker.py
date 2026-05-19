from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from app.core.constants import (
    INGESTION_JOB_STATUS_RUNNING,
    INGESTION_JOB_STATUS_SUCCEEDED,
    INGESTION_JOB_STEP_DONE,
    INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
)
from app.core.logging import configure_logging
from app.models.document import DocumentModel
from app.rag.pipeline.ingestion_pipeline import IngestionPipeline
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository


configure_logging()


class IngestionWorker:
    def __init__(self, poll_interval_seconds: int = 5) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.logger = logging.getLogger(__name__)
        self.ingestion_job_repository = IngestionJobRepository()
        self.document_repository = DocumentRepository()
        self.ingestion_pipeline = IngestionPipeline()

    async def _process_job(self, job: dict) -> None:
        job_id = job["_id"]
        document_id = job["document_id"]
        metadata = job.get("metadata") or {}
        cleanup_profile = metadata.get("cleanup_profile", "default")
        engine = metadata.get("engine", "markitdown")

        document = await self.document_repository.get_document_by_id(document_id)
        if document is None:
            await self.ingestion_job_repository.update_job_status(
                job_id,
                status="failed",
                current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                error_message="Document not found",
                progress=0,
            )
            return

        try:
            await self.ingestion_job_repository.update_job_status(
                job_id,
                status=INGESTION_JOB_STATUS_RUNNING,
                current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                progress=20,
            )
            document_model = DocumentModel.model_validate(document)
            await self.ingestion_pipeline.run(
                document_model,
                cleanup_profile=cleanup_profile,
                engine=engine,
                job_id=job_id,
            )
            await self.ingestion_job_repository.update_job_status(
                job_id,
                status=INGESTION_JOB_STATUS_SUCCEEDED,
                current_step=INGESTION_JOB_STEP_DONE,
                progress=100,
                clear_error=True,
            )
        except Exception as exc:
            self.logger.exception("Failed to process ingestion job %s", job_id)
            _ = exc

    async def run(self) -> None:
        self.logger.info("Ingestion worker started")
        while True:
            job = await self.ingestion_job_repository.claim_next_queued_job()
            if job is None:
                await asyncio.sleep(self.poll_interval_seconds)
                continue
            self.logger.info("Picked ingestion job %s for document %s", job["_id"], job["document_id"])
            await self._process_job(job)


async def run_worker() -> None:
    worker = IngestionWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
