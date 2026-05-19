from app.core.constants import (
    DOCUMENT_STATUS_DELETED,
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_PROCESSING,
    DOCUMENT_STATUS_READY,
    DOCUMENT_STATUS_UPLOADED,
)
from app.db.mongodb import get_database


async def init_mongodb() -> None:
    db = get_database()
    collection_names = [
        "users",
        "sessions",
        "documents",
        "document_versions",
        "chunks",
        "messages",
        "ingestion_jobs",
        "retrieval_logs",
        "feedbacks",
    ]

    existing_collections = await db.list_collection_names()
    for collection_name in collection_names:
        if collection_name not in existing_collections:
            await db.create_collection(collection_name)

    await db.users.create_index([("email", 1)], unique=True)
    await db.sessions.create_index([("owner_user_id", 1), ("updated_at", -1)])
    await db.documents.create_index([("owner_user_id", 1), ("created_at", -1)])
    await db.documents.create_index([("source_type", 1), ("visibility", 1), ("status", 1)])
    await db.documents.create_index([("uploaded_in_session_id", 1)])
    await db.documents.create_index([("content_hash", 1)])
    await db.document_versions.create_index([("document_id", 1), ("version_number", -1)], unique=True)
    await db.chunks.create_index([("document_id", 1), ("chunk_index", 1)], unique=True)
    await db.chunks.create_index([("document_version_id", 1)])
    await db.chunks.create_index([("owner_user_id", 1), ("source_type", 1), ("visibility", 1)])
    await db.messages.create_index([("session_id", 1), ("created_at", 1)])
    await db.messages.create_index([("owner_user_id", 1), ("created_at", -1)])
    await db.ingestion_jobs.create_index([("document_id", 1), ("created_at", -1)])
    await db.ingestion_jobs.create_index([("status", 1), ("current_step", 1), ("created_at", -1)])
    await db.retrieval_logs.create_index([("user_id", 1), ("created_at", -1)])
    await db.retrieval_logs.create_index([("session_id", 1), ("created_at", -1)])
    await db.feedbacks.create_index([("user_id", 1), ("created_at", -1)])
    await db.feedbacks.create_index([("session_id", 1), ("created_at", -1)])
