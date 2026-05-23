import json
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pymongo import MongoClient

from app.core.constants import (
    RETRIEVAL_SCOPE_AUTO,
    RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
    RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
    RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
)
from app.core.config import settings
from app.rag.retrieval.context_validator import ContextValidator
from app.rag.retrieval.filters import build_retrieval_filter
from app.rag.retrieval.resolvers import DocumentResolver, ScopeResolver
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.strategy import RetrievalStrategy
from app.rag.vectorstore.chroma_store import ChromaVectorStore


TEST_PREFIX = "rag_it_metrics"
USER_A = f"{TEST_PREFIX}_user_a"
USER_B = f"{TEST_PREFIX}_user_b"
SESSION_CURRENT = f"{TEST_PREFIX}_session_current"
SESSION_OLD = f"{TEST_PREFIX}_session_old"
SYSTEM_DOC_ID = f"{TEST_PREFIX}_system_doc_enterprise"
USER_CURRENT_DOC_ID = f"{TEST_PREFIX}_user_current_doc"
USER_OLD_DOC_ID = f"{TEST_PREFIX}_user_old_doc"
USER_B_DOC_ID = f"{TEST_PREFIX}_user_b_doc"
PROCEDURE_TITLE = "Dang ky thanh lap doanh nghiep tu nhan"
EMBEDDING_DIMENSION = 1536
METRIC_CASES_PATH = Path(__file__).parent / "fixtures" / "rag_retrieval_metric_cases.json"
HARD_CASES_PATH = Path(__file__).parent / "fixtures" / "rag_retrieval_hard_cases.json"


class StaticEmbeddingService:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed_text(self, text: str) -> list[float]:
        return self.vector


def _vector(primary: float, secondary: float = 0.0) -> list[float]:
    return [primary, secondary, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _metrics_at_k(retrieved: list[dict], expected_chunk_ids: set[str]) -> dict[str, float]:
    retrieved_ids = [item["metadata"].get("chunk_id", item["id"]) for item in retrieved]
    true_positive_count = len([chunk_id for chunk_id in retrieved_ids if chunk_id in expected_chunk_ids])
    first_relevant_rank = next(
        (index + 1 for index, chunk_id in enumerate(retrieved_ids) if chunk_id in expected_chunk_ids),
        None,
    )

    return {
        "precision": true_positive_count / len(retrieved_ids) if retrieved_ids else 0.0,
        "recall": true_positive_count / len(expected_chunk_ids) if expected_chunk_ids else 0.0,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def _build_retriever(query_vector: list[float], vector_store: ChromaVectorStore) -> Retriever:
    retriever = Retriever.__new__(Retriever)
    retriever.embedding_service = StaticEmbeddingService(query_vector)
    retriever.vector_store = vector_store
    return retriever


def _get_sync_database():
    client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=3000)
    return client, client[settings.MONGODB_DB_NAME]


def _cleanup_mongo() -> None:
    client, db = _get_sync_database()
    for collection_name in ("chunks", "documents", "sessions", "users"):
        db[collection_name].delete_many({"_id": {"$regex": f"^{TEST_PREFIX}"}})
    client.close()


def _cleanup_chroma(vector_store: ChromaVectorStore) -> None:
    vector_store.collection.delete(where={"test_run": TEST_PREFIX})


def _seed_mongo() -> None:
    client, db = _get_sync_database()
    now = datetime.now(UTC)

    db.users.insert_many(
        [
            {
                "_id": USER_A,
                "email": "rag-it-user-a@example.test",
                "name": "RAG IT User A",
                "role": "user",
                "created_at": now,
                "updated_at": now,
            },
            {
                "_id": USER_B,
                "email": "rag-it-user-b@example.test",
                "name": "RAG IT User B",
                "role": "user",
                "created_at": now,
                "updated_at": now,
            },
        ]
    )
    db.sessions.insert_many(
        [
            {
                "_id": SESSION_CURRENT,
                "owner_user_id": USER_A,
                "title": "Current upload session",
                "conversation_state": {},
                "status": "active",
                "created_at": now,
                "updated_at": now,
            },
            {
                "_id": SESSION_OLD,
                "owner_user_id": USER_A,
                "title": "Old upload session",
                "conversation_state": {},
                "status": "active",
                "created_at": now,
                "updated_at": now,
            },
        ]
    )
    db.documents.insert_many(
        [
            {
                "_id": SYSTEM_DOC_ID,
                "title": PROCEDURE_TITLE,
                "filename": "system_enterprise.md",
                "file_type": "md",
                "mime_type": "text/markdown",
                "source_type": "system",
                "visibility": "global",
                "procedure_title": PROCEDURE_TITLE,
                "status": "ready",
                "raw_storage_path": "backend/storage/markdown/system/system/sysdoc_b98ec8c6-aa4e-48cf-9d10-2a906ce92dd7/document.md",
                "markdown_storage_path": "backend/storage/markdown/system/system/sysdoc_b98ec8c6-aa4e-48cf-9d10-2a906ce92dd7/document.md",
                "chunk_count": 3,
                "created_at": now,
                "updated_at": now,
            },
            {
                "_id": USER_CURRENT_DOC_ID,
                "title": "Ho so cong ty Alpha",
                "filename": "ho_so_alpha.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "source_type": "user_upload",
                "visibility": "private",
                "owner_user_id": USER_A,
                "uploaded_in_session_id": SESSION_CURRENT,
                "status": "ready",
                "raw_storage_path": "test://ho_so_alpha.pdf",
                "chunk_count": 2,
                "created_at": now,
                "updated_at": now,
            },
            {
                "_id": USER_OLD_DOC_ID,
                "title": "Ho so cong ty Beta",
                "filename": "ho_so_beta.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "source_type": "user_upload",
                "visibility": "private",
                "owner_user_id": USER_A,
                "uploaded_in_session_id": SESSION_OLD,
                "status": "ready",
                "raw_storage_path": "test://ho_so_beta.pdf",
                "chunk_count": 1,
                "created_at": now,
                "updated_at": now,
            },
            {
                "_id": USER_B_DOC_ID,
                "title": "Ho so cong ty Leak",
                "filename": "ho_so_leak.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "source_type": "user_upload",
                "visibility": "private",
                "owner_user_id": USER_B,
                "uploaded_in_session_id": SESSION_CURRENT,
                "status": "ready",
                "raw_storage_path": "test://ho_so_leak.pdf",
                "chunk_count": 1,
                "created_at": now,
                "updated_at": now,
            },
        ]
    )
    client.close()


def _chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    metadata: dict,
    chunk_index: int,
) -> dict:
    full_metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "filename": metadata["filename"],
        "source_type": metadata["source_type"],
        "visibility": metadata["visibility"],
        "test_run": TEST_PREFIX,
        **metadata,
    }
    return {
        "_id": chunk_id,
        "id": chunk_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": content,
        "source_type": full_metadata["source_type"],
        "visibility": full_metadata["visibility"],
        "owner_user_id": full_metadata.get("owner_user_id"),
        "session_id": full_metadata.get("session_id"),
        "filename": full_metadata["filename"],
        "procedure_title": full_metadata.get("procedure_title"),
        "section_title": full_metadata.get("section_title"),
        "token_count": len(content.split()),
        "embedding": {
            "model": "static-test-embedding",
            "dimension": EMBEDDING_DIMENSION,
            "vector_store": "chroma",
            "collection_name": "test",
            "vector_id": chunk_id,
            "embedded_at": datetime.now(UTC),
        },
        "metadata": full_metadata,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


def _seed_chunks(vector_store: ChromaVectorStore) -> list[dict]:
    system_metadata = {
        "source_type": "system",
        "visibility": "global",
        "filename": "system_enterprise.md",
        "procedure_title": PROCEDURE_TITLE,
    }
    user_current_metadata = {
        "source_type": "user_upload",
        "visibility": "private",
        "owner_user_id": USER_A,
        "session_id": SESSION_CURRENT,
        "filename": "ho_so_alpha.pdf",
    }
    user_old_metadata = {
        "source_type": "user_upload",
        "visibility": "private",
        "owner_user_id": USER_A,
        "session_id": SESSION_OLD,
        "filename": "ho_so_beta.pdf",
    }
    user_b_metadata = {
        "source_type": "user_upload",
        "visibility": "private",
        "owner_user_id": USER_B,
        "session_id": SESSION_CURRENT,
        "filename": "ho_so_leak.pdf",
    }
    chunks = [
        _chunk(
            f"{TEST_PREFIX}_sys_fee",
            SYSTEM_DOC_ID,
            "Le phi dang ky doanh nghiep tu nhan gom phi cong bo 100000 dong va le phi 25000 dong.",
            {**system_metadata, "section_title": "Le phi"},
            0,
        ),
        _chunk(
            f"{TEST_PREFIX}_sys_deadline",
            SYSTEM_DOC_ID,
            "Thoi han giai quyet dang ky thanh lap doanh nghiep tu nhan la 03 ngay lam viec khi nhan du ho so hop le.",
            {**system_metadata, "section_title": "Thoi han"},
            1,
        ),
        _chunk(
            f"{TEST_PREFIX}_sys_noise",
            SYSTEM_DOC_ID,
            "Noi dung nhieu ve bao tang ngoai cong lap khong lien quan den dang ky doanh nghiep.",
            {**system_metadata, "section_title": "Noise"},
            2,
        ),
        _chunk(
            f"{TEST_PREFIX}_user_current_fee",
            USER_CURRENT_DOC_ID,
            "Tai lieu upload Alpha ghi le phi noi bo la 75000 dong va can nop tai quay tiep nhan.",
            {**user_current_metadata, "section_title": "Le phi"},
            0,
        ),
        _chunk(
            f"{TEST_PREFIX}_user_current_deadline",
            USER_CURRENT_DOC_ID,
            "Tai lieu upload Alpha ghi thoi han xu ly noi bo la 05 ngay lam viec.",
            {**user_current_metadata, "section_title": "Thoi han"},
            1,
        ),
        _chunk(
            f"{TEST_PREFIX}_user_old_deadline",
            USER_OLD_DOC_ID,
            "Tai lieu upload Beta o session cu ghi thoi han xu ly la 07 ngay lam viec.",
            {**user_old_metadata, "section_title": "Thoi han"},
            0,
        ),
        _chunk(
            f"{TEST_PREFIX}_other_user_leak",
            USER_B_DOC_ID,
            "Tai lieu cua user B co noi dung rat giong: le phi 999999 dong, khong duoc phep lo sang user A.",
            {**user_b_metadata, "section_title": "Security"},
            0,
        ),
    ]
    embeddings = [
        _vector(1.0),
        _vector(0.96, 0.04),
        _vector(0.0, 1.0),
        _vector(1.0),
        _vector(0.95, 0.05),
        _vector(0.94, 0.06),
        _vector(1.0),
    ]
    vector_store.add_chunks(
        [{"id": chunk["_id"], "content": chunk["content"], "metadata": chunk["metadata"]} for chunk in chunks],
        embeddings,
    )
    return chunks


def _seed_chunks_mongo(chunks: list[dict]) -> None:
    client, db = _get_sync_database()
    db.chunks.insert_many(chunks)
    client.close()


def _mongo_is_available() -> bool:
    try:
        client, db = _get_sync_database()
        db.command("ping")
        client.close()
    except Exception:
        return False
    return True


@pytest.fixture()
def seeded_rag_test_data():
    if not _mongo_is_available():
        pytest.skip("MongoDB is not available for direct RAG integration test")

    vector_store = ChromaVectorStore()
    _cleanup_mongo()
    _cleanup_chroma(vector_store)
    _seed_mongo()
    chunks = _seed_chunks(vector_store)
    _seed_chunks_mongo(chunks)

    yield vector_store

    _cleanup_mongo()
    _cleanup_chroma(vector_store)


def test_system_procedure_retrieval_has_multiple_evidence_chunks_and_metrics(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    retriever = _build_retriever(_vector(1.0), vector_store)
    metadata_filter = build_retrieval_filter(
        scope=RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
        user_id=USER_A,
        procedure_title=PROCEDURE_TITLE,
    )

    retrieved = retriever.retrieve(
        question="Dang ky thanh lap doanh nghiep tu nhan le phi va thoi han bao lau?",
        where_filter=metadata_filter,
        top_k=3,
    )
    validation = ContextValidator(min_similarity=0.2).validate_branch(retrieved, metadata_filter)
    evidence_ids = {f"{TEST_PREFIX}_sys_fee", f"{TEST_PREFIX}_sys_deadline"}
    metrics = _metrics_at_k(validation.contexts, evidence_ids)
    retrieved_ids = [item["metadata"]["chunk_id"] for item in validation.contexts]

    assert evidence_ids.issubset(set(retrieved_ids))
    assert len(evidence_ids) >= 2
    assert metrics["precision"] >= 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert all(item["metadata"]["source_type"] == "system" for item in validation.contexts)
    assert all(item["metadata"]["procedure_title"] == PROCEDURE_TITLE for item in validation.contexts)


def test_current_session_upload_retrieval_blocks_other_user_and_old_session(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    retriever = _build_retriever(_vector(1.0), vector_store)
    metadata_filter = build_retrieval_filter(
        scope=RETRIEVAL_SCOPE_CURRENT_SESSION_UPLOADS,
        user_id=USER_A,
        session_id=SESSION_CURRENT,
    )

    retrieved = retriever.retrieve(
        question="File vua upload Alpha noi gi ve le phi va thoi han?",
        where_filter=metadata_filter,
        top_k=5,
    )
    validation = ContextValidator(min_similarity=0.2).validate_branch(retrieved, metadata_filter)
    evidence_ids = {f"{TEST_PREFIX}_user_current_fee", f"{TEST_PREFIX}_user_current_deadline"}
    metrics = _metrics_at_k(validation.contexts, evidence_ids)
    retrieved_ids = [item["metadata"]["chunk_id"] for item in validation.contexts]

    assert evidence_ids.issubset(set(retrieved_ids))
    assert f"{TEST_PREFIX}_other_user_leak" not in retrieved_ids
    assert f"{TEST_PREFIX}_user_old_deadline" not in retrieved_ids
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert all(item["metadata"]["owner_user_id"] == USER_A for item in validation.contexts)
    assert all(item["metadata"]["session_id"] == SESSION_CURRENT for item in validation.contexts)


def test_user_all_uploads_can_retrieve_old_session_without_cross_user_leak(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    retriever = _build_retriever(_vector(1.0), vector_store)
    metadata_filter = build_retrieval_filter(
        scope=RETRIEVAL_SCOPE_USER_ALL_UPLOADS,
        user_id=USER_A,
    )

    retrieved = retriever.retrieve(
        question="Tai lieu da upload truoc do cua toi ghi thoi han bao lau?",
        where_filter=metadata_filter,
        top_k=6,
    )
    validation = ContextValidator(min_similarity=0.2).validate_branch(retrieved, metadata_filter)
    expected_old_chunk = f"{TEST_PREFIX}_user_old_deadline"
    retrieved_ids = [item["metadata"]["chunk_id"] for item in validation.contexts]

    assert expected_old_chunk in retrieved_ids
    assert f"{TEST_PREFIX}_other_user_leak" not in retrieved_ids
    assert all(item["metadata"]["owner_user_id"] == USER_A for item in validation.contexts)


def test_retrieval_metric_json_cases_match_expected_evidence_chunks(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    retriever = _build_retriever(_vector(1.0), vector_store)
    validator = ContextValidator(min_similarity=0.2)
    cases = json.loads(METRIC_CASES_PATH.read_text(encoding="utf-8"))

    for case in cases:
        retrieved = retriever.retrieve(
            question=case["question"],
            where_filter=case["metadata_filter"],
            top_k=case["top_k"],
        )
        validation = validator.validate_branch(retrieved, case["metadata_filter"])
        retrieved_ids = [item["metadata"]["chunk_id"] for item in validation.contexts]
        metrics = _metrics_at_k(validation.contexts, set(case["expected_chunk_ids"]))

        assert validation.should_answer is True, case["id"]
        assert set(case["expected_chunk_ids"]).issubset(set(retrieved_ids)), {
            "case": case["id"],
            "retrieved_ids": retrieved_ids,
            "expected_chunk_ids": case["expected_chunk_ids"],
            "metrics": metrics,
        }
        assert not set(case["blocked_chunk_ids"]).intersection(retrieved_ids), {
            "case": case["id"],
            "retrieved_ids": retrieved_ids,
            "blocked_chunk_ids": case["blocked_chunk_ids"],
        }
        assert metrics["precision"] >= case["min_precision"], {"case": case["id"], "metrics": metrics}
        assert metrics["recall"] >= case["min_recall"], {"case": case["id"], "metrics": metrics}
        assert metrics["mrr"] >= case["min_mrr"], {"case": case["id"], "metrics": metrics}


async def _run_hard_case(case: dict, retriever: Retriever) -> dict:
    scope_resolution = ScopeResolver().resolve(
        question=case["question"],
        user_id=USER_A,
        session_id=SESSION_CURRENT,
        scope=RETRIEVAL_SCOPE_AUTO,
        conversation_state=case.get("conversation_state", {}),
    )
    document_resolution = await DocumentResolver().resolve(
        scope=scope_resolution.scope,
        metadata_filter=scope_resolution.metadata_filter,
        user_id=USER_A,
        session_id=SESSION_CURRENT,
        detected_filename=scope_resolution.detected_filename,
        detected_procedure_title=scope_resolution.detected_procedure_title,
        conversation_state=case.get("conversation_state", {}),
    )
    retrieval_plan = RetrievalStrategy().plan(
        rewritten_question=case["question"],
        intent_resolution={"intent": "compare_documents" if "hybrid" in case["expected_scope"] else "ask_question", "needs_retrieval": True},
        scope=scope_resolution.scope,
        metadata_filter=document_resolution.metadata_filter,
    )
    branch_results = []
    for branch in retrieval_plan.branches:
        branch_results.append(
            {
                "name": branch.name,
                "metadata_filter": branch.metadata_filter,
                "contexts": retriever.retrieve(branch.query, branch.metadata_filter, branch.top_k),
            }
        )
    validation = ContextValidator(min_similarity=0.2).validate_all(branch_results)
    return {
        "scope_resolution": scope_resolution,
        "document_resolution": document_resolution,
        "retrieval_plan": retrieval_plan,
        "validation": validation,
    }


def test_hard_scope_noise_cases_retrieve_expected_evidence_chunks(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    retriever = _build_retriever(_vector(1.0), vector_store)
    cases = json.loads(HARD_CASES_PATH.read_text(encoding="utf-8"))

    async def _run_all_cases() -> list[dict]:
        return [await _run_hard_case(case, retriever) for case in cases]

    results = asyncio.run(_run_all_cases())

    for case, result in zip(cases, results):
        scope_resolution = result["scope_resolution"]
        retrieval_plan = result["retrieval_plan"]
        validation = result["validation"]
        retrieved_ids = [item["metadata"]["chunk_id"] for item in validation.contexts]
        metrics = _metrics_at_k(validation.contexts, set(case["expected_chunk_ids"]))

        assert scope_resolution.scope == case["expected_scope"], {
            "case": case["id"],
            "actual_scope": scope_resolution.scope,
            "matched_rules": scope_resolution.matched_rules,
            "metadata_filter": scope_resolution.metadata_filter,
        }
        assert retrieval_plan.mode == case["expected_mode"], {
            "case": case["id"],
            "actual_mode": retrieval_plan.mode,
            "branches": retrieval_plan.model_dump()["branches"],
        }
        if case.get("expected_branch_names"):
            assert [branch.name for branch in retrieval_plan.branches] == case["expected_branch_names"], case["id"]
        assert set(case["expected_chunk_ids"]).issubset(set(retrieved_ids)), {
            "case": case["id"],
            "retrieved_ids": retrieved_ids,
            "expected_chunk_ids": case["expected_chunk_ids"],
            "metrics": metrics,
        }
        assert not set(case["blocked_chunk_ids"]).intersection(retrieved_ids), {
            "case": case["id"],
            "retrieved_ids": retrieved_ids,
            "blocked_chunk_ids": case["blocked_chunk_ids"],
        }
        assert metrics["precision"] >= case["min_precision"], {"case": case["id"], "metrics": metrics}
        assert metrics["recall"] >= case["min_recall"], {"case": case["id"], "metrics": metrics}
        assert metrics["mrr"] >= case["min_mrr"], {"case": case["id"], "metrics": metrics}
