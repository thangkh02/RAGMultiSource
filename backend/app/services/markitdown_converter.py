from pathlib import Path

from app.core.config import settings
from app.core.constants import (
    DOCUMENT_STATUS_CONVERTED,
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    INGESTION_JOB_STATUS_FAILED,
    INGESTION_JOB_STATUS_RUNNING,
    INGESTION_JOB_STATUS_SUCCEEDED,
    INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
    SOURCE_TYPE_USER_UPLOAD,
)
from app.rag.converter.markitdown_converter import MarkItDownMarkdownConverter
from app.rag.converter.user_upload_converter import UserUploadMarkdownConverter
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository


class MarkItDownConversionService:
    def __init__(self) -> None:
        self.document_repository = DocumentRepository()
        self.ingestion_job_repository = IngestionJobRepository()
        self.system_converter = MarkItDownMarkdownConverter()
        self.user_upload_converter = UserUploadMarkdownConverter()

    def _backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_existing_path(self, path_str: str) -> Path:
        direct = Path(path_str)
        if direct.exists():
            return direct
        backend_candidate = self._backend_root() / path_str
        if backend_candidate.exists():
            return backend_candidate
        repo_candidate = self._repo_root() / path_str
        if repo_candidate.exists():
            return repo_candidate
        return direct

    def _build_markdown_path(self, document: dict) -> str:
        source_type = document.get("source_type") or "user_upload"
        owner_folder = document.get("owner_user_id") or "system"
        document_id = document.get("_id")
        rel_path = settings.markdown_dir_path / source_type / owner_folder / document_id / "document.md"
        return rel_path.as_posix()

    async def convert_document(self, document_id: str, owner_user_id: str | None = None) -> dict:
        document = await self.document_repository.get_document_by_id(document_id)
        if document is None:
            raise ValueError("Document not found")
        if owner_user_id is not None and document.get("owner_user_id") != owner_user_id:
            raise ValueError("Document not found")

        raw_storage_path = document.get("raw_storage_path")
        if not raw_storage_path:
            raise ValueError("Document has no raw_storage_path")
        resolved_raw_path = self._resolve_existing_path(raw_storage_path)
        if not resolved_raw_path.exists():
            raise ValueError(f"Raw file not found: {raw_storage_path}")

        markdown_path = self._build_markdown_path(document)
        resolved_markdown_path = Path(markdown_path) if Path(markdown_path).is_absolute() else self._backend_root() / markdown_path
        jobs = await self.ingestion_job_repository.list_jobs_by_document_id(document_id)
        job_id = jobs[-1]["_id"] if jobs else None

        try:
            await self.document_repository.update_document_status(document_id, DOCUMENT_STATUS_PROCESSING)
            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_RUNNING,
                    current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                    progress=25,
                )

            if document.get("source_type") == SOURCE_TYPE_USER_UPLOAD:
                converted_path = self.user_upload_converter.convert_to_markdown(str(resolved_raw_path), str(resolved_markdown_path))
            else:
                converted_path = self.system_converter.convert_to_markdown(str(resolved_raw_path), str(resolved_markdown_path))
            await self.document_repository.update_markdown_path(document_id, markdown_path)
            await self.document_repository.update_document_status(document_id, DOCUMENT_STATUS_CONVERTED)

            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_SUCCEEDED,
                    current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                    progress=100,
                    clear_error=True,
                )

            updated_document = await self.document_repository.get_document_by_id(document_id)
            if updated_document is None:
                raise ValueError("Document not found after conversion")
            return updated_document
        except Exception as exc:
            await self.document_repository.update_document_status(document_id, DOCUMENT_STATUS_FAILED)
            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_FAILED,
                    current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                    error_message=str(exc),
                    progress=0,
                )
            raise
