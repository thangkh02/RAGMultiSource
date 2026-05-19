from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from app.models.base import MongoBaseModel


class DocumentVersionModel(MongoBaseModel):
    id: str = Field(alias="_id")
    document_id: str
    version_number: int
    parser: str = "markitdown"
    parser_version: str = "latest"
    markdown_storage_path: str
    chunking_strategy: str = "markdown_heading_recursive"
    chunk_size: int = 700
    chunk_overlap: int = 120
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    status: Literal["uploaded", "processing", "ready", "failed", "deleted"] = "processing"
    created_at: datetime = Field(default_factory=datetime.utcnow)
