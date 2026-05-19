from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.common_schema import TimestampedItem


class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str | None = None
    filename: str
    status: str
    raw_storage_path: str
    message: Optional[str] = None


class DocumentItem(TimestampedItem):
    id: str
    title: str
    filename: str
    file_type: str
    mime_type: str
    source_type: str
    owner_user_id: Optional[str] = None
    uploaded_in_session_id: Optional[str] = None
    visibility: str
    raw_storage_path: str
    markdown_storage_path: Optional[str] = None
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    content_hash: Optional[str] = None
