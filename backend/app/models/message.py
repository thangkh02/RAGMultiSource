from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from app.models.base import MongoBaseModel


class MessageModel(MongoBaseModel):
    id: str = Field(alias="_id")
    session_id: str
    owner_user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    llm_model_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
