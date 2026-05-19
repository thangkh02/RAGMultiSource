from pathlib import Path
from fastapi import UploadFile

from app.core.config import settings
from app.core.constants import (
    DOCUMENT_STATUS_UPLOADED,
    INGESTION_JOB_STATUS_QUEUED,
    INGESTION_JOB_STEP_UPLOADED,
    SOURCE_TYPE_SYSTEM,
    VISIBILITY_GLOBAL,
)
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.utils.file_utils import save_upload_file
from app.utils.id_utils import generate_id


class SystemDocumentService:
    def __init__(self) -> None:
        self.document_repository = DocumentRepository()
        self.ingestion_job_repository = IngestionJobRepository()

    def _build_raw_path(self, document_id: str, extension: str) -> Path:
        return settings.upload_dir_path / "system" / document_id / f"original.{extension}"

    async def upload_system_document(self, file: UploadFile):
        document_id = generate_id("sysdoc")
        extension = (file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "unknown")
        raw_path = self._build_raw_path(document_id=document_id, extension=extension)
        raw_path_str = raw_path.as_posix()
        await save_upload_file(file, raw_path_str)
        file_size_bytes = raw_path.stat().st_size
        document = DocumentModel(
            id=document_id,
            title=file.filename,
            filename=file.filename,
            file_type=extension,
            mime_type=file.content_type or "application/octet-stream",
            source_type=SOURCE_TYPE_SYSTEM,
            owner_user_id=None,
            uploaded_in_session_id=None,
            visibility=VISIBILITY_GLOBAL,
            raw_storage_path=raw_path_str,
            status=DOCUMENT_STATUS_UPLOADED,
            file_size_bytes=file_size_bytes,
        )
        await self.document_repository.create_document(document)
        job = IngestionJobModel(
            id=generate_id("job"),
            document_id=document_id,
            owner_user_id=None,
            status=INGESTION_JOB_STATUS_QUEUED,
            current_step=INGESTION_JOB_STEP_UPLOADED,
            metadata={
                "source_type": SOURCE_TYPE_SYSTEM,
                "visibility": VISIBILITY_GLOBAL,
                "raw_storage_path": raw_path_str,
                "cleanup_profile": "vi_scientific_paper",
                "engine": "markitdown",
            },
        )
        await self.ingestion_job_repository.create_job(job)
        return document
