from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from app.models.base import MongoBaseModel


class RetrievalLogModel(MongoBaseModel):
    id: str = Field(alias="_id")
    user_id: str
    session_id: Optional[str] = None
    question: str
    resolved_scope: Literal[
        "current_upload",
        "current_session_uploads",
        "all_user_uploads",
        "user_all_uploads",
        "user_file_name",
        "system_docs",
        "system_procedure",
        "hybrid_system_and_user",
        "general_query",
        "need_clarification",
        "mixed",
        "auto",
    ]
    selected_document_ids: list[str] = Field(default_factory=list)
    retrieval_filter: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
