import logging
from pathlib import Path
from hashlib import sha256

from fastapi import UploadFile

from app.core.constants import (
    DOCUMENT_STATUS_UPLOADED,
    SOURCE_TYPE_USER_UPLOAD,
    INGESTION_JOB_STATUS_QUEUED,
    INGESTION_JOB_STEP_UPLOADED,
    VISIBILITY_PRIVATE,
)
from app.core.config import settings
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.chunk_repository import ChunkRepository
from app.rag.pipeline.ingestion_pipeline import IngestionPipeline
from app.utils.file_utils import save_upload_file
from app.utils.id_utils import generate_id


class DocumentService:
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        self.document_repository = DocumentRepository()
        self.ingestion_job_repository = IngestionJobRepository()
        self.chunk_repository = ChunkRepository()
        self.ingestion_pipeline = IngestionPipeline()

    def _validate_document_type(self, file: UploadFile) -> str:
        filename = Path(file.filename or "").name
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in {"pdf", "docx"}:
            raise ValueError("Only PDF and DOCX files are supported.")
        return ext

    def _build_raw_path(self, owner_user_id: str, document_id: str, extension: str) -> Path:
        return settings.upload_dir_path / "user_upload" / owner_user_id / document_id / f"original.{extension}"

    async def upload_user_document(self, file: UploadFile, owner_user_id: str, session_id: str | None):
        owner_user_id = owner_user_id.strip()
        if not owner_user_id:
            raise ValueError("owner_user_id is required.")
        extension = self._validate_document_type(file)
        document_id = generate_id("doc")
        raw_path = self._build_raw_path(owner_user_id=owner_user_id, document_id=document_id, extension=extension)
        raw_path_str = raw_path.as_posix()
        await save_upload_file(file, raw_path)
        file_size_bytes = raw_path.stat().st_size
        content_hash = sha256(raw_path.read_bytes()).hexdigest()
        original_filename = Path(file.filename or f"original.{extension}").name
        document = DocumentModel(
            id=document_id,
            title=(Path(original_filename).stem or document_id),
            filename=original_filename,
            file_type=extension,
            mime_type=file.content_type or "application/octet-stream",
            source_type=SOURCE_TYPE_USER_UPLOAD,
            owner_user_id=owner_user_id,
            uploaded_in_session_id=session_id,
            visibility=VISIBILITY_PRIVATE,
            raw_storage_path=raw_path_str,
            status=DOCUMENT_STATUS_UPLOADED,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
        )
        await self.document_repository.create_document(document)

        ingestion_job = IngestionJobModel(
            id=generate_id("job"),
            document_id=document_id,
            owner_user_id=owner_user_id,
            status=INGESTION_JOB_STATUS_QUEUED,
            current_step=INGESTION_JOB_STEP_UPLOADED,
            metadata={
                "source_type": SOURCE_TYPE_USER_UPLOAD,
                "visibility": VISIBILITY_PRIVATE,
                "uploaded_in_session_id": session_id,
                "raw_storage_path": raw_path_str,
                "cleanup_profile": "default",
                "engine": "markitdown",
            },
        )
        job_id = await self.ingestion_job_repository.create_job(ingestion_job)
        return {
            "document": document,
            "job_id": job_id,
        }

    async def list_documents(self, owner_user_id: str | None = None):
        documents = await self.document_repository.list_documents(owner_user_id)
        return [self._serialize_document(document) for document in documents]

    async def get_document(self, document_id: str):
        document = await self.document_repository.get_document_by_id(document_id)
        if document is None:
            return None
        return self._serialize_document(document)

    async def list_ingestion_jobs(self, document_id: str) -> list[dict]:
        jobs = await self.ingestion_job_repository.list_jobs_by_document_id(document_id)
        return [
            {
                "id": job.get("_id"),
                "document_id": job.get("document_id"),
                "owner_user_id": job.get("owner_user_id"),
                "document_version_id": job.get("document_version_id"),
                "job_type": job.get("job_type"),
                "status": job.get("status"),
                "current_step": job.get("current_step"),
                "progress": job.get("progress"),
                "error_message": job.get("error_message"),
                "metadata": job.get("metadata") or {},
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
            }
            for job in jobs
        ]

    async def delete_document(self, document_id: str) -> bool:
        document = await self.document_repository.get_document_by_id(document_id)
        if document is None:
            return False

        raw_storage_path = document.get("raw_storage_path")
        if raw_storage_path:
            from app.utils.file_utils import remove_path

            remove_path(raw_storage_path)
            parent_dir = Path(raw_storage_path).parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()

        markdown_storage_path = document.get("markdown_storage_path")
        if markdown_storage_path:
            from app.utils.file_utils import remove_path

            remove_path(markdown_storage_path)
            parent_dir = Path(markdown_storage_path).parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()

        await self.chunk_repository.delete_chunks_by_document_id(document_id)
        await self.ingestion_job_repository.delete_jobs_by_document_id(document_id)

        await self.document_repository.soft_delete_document(document_id)
        return True

    def _serialize_document(self, document: dict) -> dict:
        return {
            "id": document.get("_id"),
            "title": document.get("title"),
            "filename": document.get("filename"),
            "file_type": document.get("file_type"),
            "mime_type": document.get("mime_type"),
            "source_type": document.get("source_type"),
            "owner_user_id": document.get("owner_user_id"),
            "uploaded_in_session_id": document.get("uploaded_in_session_id"),
            "visibility": document.get("visibility"),
            "raw_storage_path": document.get("raw_storage_path"),
            "markdown_storage_path": document.get("markdown_storage_path"),
            "status": document.get("status"),
            "page_count": document.get("page_count"),
            "chunk_count": document.get("chunk_count"),
            "file_size_bytes": document.get("file_size_bytes"),
            "content_hash": document.get("content_hash"),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
        }
