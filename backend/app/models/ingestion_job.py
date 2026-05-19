from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from app.models.base import MongoBaseModel


class IngestionJobModel(MongoBaseModel):
    id: str = Field(alias="_id")
    document_id: str
    owner_user_id: Optional[str] = None
    document_version_id: Optional[str] = None
    job_type: Literal["ingest", "reprocess", "delete"] = "ingest"
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    current_step: Literal["uploaded", "convert_to_markdown", "converting", "chunking", "embedding", "done"] = "uploaded"
    progress: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
