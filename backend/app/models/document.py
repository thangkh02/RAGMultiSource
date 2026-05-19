from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from app.models.base import MongoBaseModel


class DocumentModel(MongoBaseModel):
    id: str = Field(alias="_id")
    title: str
    filename: str
    file_type: str
    mime_type: str
    source_type: Literal["system", "user_upload"]
    visibility: Literal["global", "private"]
    owner_user_id: Optional[str] = None
    uploaded_in_session_id: Optional[str] = None
    status: Literal["uploaded", "processing", "converted", "ready", "failed", "deleted"] = "uploaded"
    raw_storage_path: str
    markdown_storage_path: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    content_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
