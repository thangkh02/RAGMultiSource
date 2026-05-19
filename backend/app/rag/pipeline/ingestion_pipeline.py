from pathlib import Path

from app.core.config import settings
from app.core.constants import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    INGESTION_JOB_STATUS_FAILED,
    INGESTION_JOB_STATUS_RUNNING,
    INGESTION_JOB_STATUS_SUCCEEDED,
    INGESTION_JOB_STEP_CHUNKING,
    INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
    INGESTION_JOB_STEP_EMBEDDING,
    INGESTION_JOB_STEP_DONE,
)
from app.models.document import DocumentModel
from app.rag.chunking.markdown_chunker import MarkdownChunker
from app.rag.embedding.bge_embedding import BGEEmbeddingService
from app.rag.converter.markitdown_converter import MarkItDownMarkdownConverter
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.utils.id_utils import generate_id


class IngestionPipeline:
    def __init__(self) -> None:
        self.converter = MarkItDownMarkdownConverter()
        self.chunker = MarkdownChunker()
        self.embedding_service = BGEEmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
        self.vector_store = ChromaVectorStore()
        self.document_repository = DocumentRepository()
        self.chunk_repository = ChunkRepository()
        self.ingestion_job_repository = IngestionJobRepository()

    def _build_markdown_path(self, document: DocumentModel) -> str:
        source_type = document.source_type or "user_upload"
        owner_folder = document.owner_user_id or "system"
        document_id = document.id
        return (settings.markdown_dir_path / source_type / owner_folder / document_id / "document.md").as_posix()

    async def run(
        self,
        document: DocumentModel,
        cleanup_profile: str = "default",
        engine: str = "markitdown",
        job_id: str | None = None,
    ) -> None:
        if job_id is None:
            jobs = await self.ingestion_job_repository.list_jobs_by_document_id(document.id)
            job_id = jobs[-1]["_id"] if jobs else None
        await self.document_repository.update_document_status(document.id, DOCUMENT_STATUS_PROCESSING)
        try:
            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_RUNNING,
                    current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                    progress=20,
                )

            markdown_path = document.markdown_storage_path or self._build_markdown_path(document)
            markdown_path = self.converter.convert_to_markdown(
                document.raw_storage_path,
                markdown_path,
                cleanup_profile=cleanup_profile,
                engine=engine,
            )
            await self.document_repository.update_markdown_path(document.id, markdown_path)

            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()

            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_RUNNING,
                    current_step=INGESTION_JOB_STEP_CHUNKING,
                    progress=60,
                )

            base_metadata = {
                "document_id": document.id,
                "source_type": document.source_type,
                "owner_user_id": document.owner_user_id,
                "session_id": document.uploaded_in_session_id,
                "filename": document.filename,
                "visibility": document.visibility,
            }
            chunks = self.chunker.chunk(markdown_text, base_metadata)
            chunk_docs: list[dict] = []
            embeddings: list[list[float]] = []
            for chunk in chunks:
                chunk_id = generate_id("chunk")
                vector_metadata = {
                    "chunk_id": chunk_id,
                    "document_id": document.id,
                    "source_type": document.source_type,
                    "owner_user_id": document.owner_user_id,
                    "session_id": document.uploaded_in_session_id,
                    "filename": document.filename,
                    "visibility": document.visibility,
                    "page_number": chunk.get("page_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section_title": chunk.get("section_title"),
                    "heading_path": chunk.get("heading_path", []),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "token_count": chunk.get("token_count", 0),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "contains_table": chunk.get("contains_table", False),
                    "contains_image": chunk.get("contains_image", False),
                }
                embedding = self.embedding_service.embed_text(chunk["content"])
                embeddings.append(embedding)
                metadata = {
                    **vector_metadata,
                    "content_hash": chunk.get("content_hash"),
                }
                chunk_docs.append(
                    {
                        "id": chunk_id,
                        **chunk,
                        "embedding": {
                            "model": settings.EMBEDDING_MODEL_NAME,
                            "dimension": len(embedding),
                            "vector_store": "chroma",
                            "collection_name": settings.CHROMA_COLLECTION_NAME,
                            "vector_id": chunk_id,
                            "embedded_at": None,
                        },
                        "metadata": metadata,
                    }
                )

            await self.chunk_repository.insert_chunks(chunk_docs)
            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_RUNNING,
                    current_step=INGESTION_JOB_STEP_EMBEDDING,
                    progress=80,
                )
            if chunk_docs:
                self.vector_store.add_chunks(chunk_docs, embeddings)
            page_numbers = [chunk.get("page_number") for chunk in chunk_docs if chunk.get("page_number") is not None]
            page_count = len(set(page_numbers)) if page_numbers else None
            await self.document_repository.update_document_fields(
                document.id,
                status=DOCUMENT_STATUS_READY,
                markdown_storage_path=markdown_path,
                chunk_count=len(chunk_docs),
                page_count=page_count,
            )
            document.markdown_storage_path = markdown_path
            document.status = DOCUMENT_STATUS_READY
            document.chunk_count = len(chunk_docs)
            document.page_count = page_count

            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_SUCCEEDED,
                    current_step=INGESTION_JOB_STEP_DONE,
                    progress=100,
                    clear_error=True,
                )
        except Exception as exc:
            await self.document_repository.update_document_status(document.id, DOCUMENT_STATUS_FAILED)
            if job_id:
                await self.ingestion_job_repository.update_job_status(
                    job_id,
                    status=INGESTION_JOB_STATUS_FAILED,
                    current_step=INGESTION_JOB_STEP_CONVERT_TO_MARKDOWN,
                    error_message=str(exc),
                    progress=0,
                )
            raise
