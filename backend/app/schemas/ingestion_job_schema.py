from datetime import datetime
from typing import Optional

from pydantic import Field

from app.schemas.common_schema import TimestampedItem


class IngestionJobItem(TimestampedItem):
    id: str
    document_id: str
    owner_user_id: Optional[str] = None
    document_version_id: Optional[str] = None
    job_type: str
    status: str
    current_step: str
    progress: int = 0
    error_message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
