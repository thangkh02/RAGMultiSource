from pathlib import Path

from fastapi.testclient import TestClient

from app.api import deps
from app.db.mongodb import get_mongo_client
from app.main import create_app
from app.services.chat_service import ChatService
from test_rag_retrieval_integration_metrics import (
    SESSION_CURRENT,
    TEST_PREFIX,
    USER_A,
    _build_retriever,
    _cleanup_mongo,
    _get_sync_database,
    _seed_mongo,
    _vector,
    seeded_rag_test_data,
)


class EvidenceOnlyLLM:
    def generate_answer(self, question: str, contexts: list[dict], answer_style: str = "short_answer") -> str:
        chunk_ids = [item["metadata"].get("chunk_id", item.get("id")) for item in contexts]
        return f"Evidence chunks: {', '.join(chunk_ids)}"


def _reset_motor_client_cache() -> None:
    try:
        get_mongo_client().close()
    except Exception:
        pass
    get_mongo_client.cache_clear()


def test_chat_api_runs_rag_pipeline_and_returns_evidence_sources(seeded_rag_test_data):
    vector_store = seeded_rag_test_data
    app = create_app()

    def _chat_service_override() -> ChatService:
        service = ChatService()
        service.qa_pipeline.retriever = _build_retriever(_vector(1.0), vector_store)
        service.qa_pipeline.llm = EvidenceOnlyLLM()
        return service

    app.dependency_overrides[deps.get_current_user_id] = lambda: USER_A
    app.dependency_overrides[deps.get_chat_service] = _chat_service_override

    _reset_motor_client_cache()
    client = TestClient(app)
    try:
        response = client.post(
            "/chat",
            json={
                "question": "Dung lay tai lieu he thong, chi doc file ho_so_alpha.pdf va cho toi ca le phi voi thoi han",
                "session_id": SESSION_CURRENT,
                "scope": "auto",
            },
        )
    finally:
        client.close()
        _reset_motor_client_cache()

    assert response.status_code == 200
    payload = response.json()
    source_chunk_ids = {source["chunk_id"] for source in payload["sources"]}

    assert f"{TEST_PREFIX}_user_current_fee" in source_chunk_ids
    assert f"{TEST_PREFIX}_user_current_deadline" in source_chunk_ids
    assert f"{TEST_PREFIX}_sys_fee" not in source_chunk_ids
    assert f"{TEST_PREFIX}_other_user_leak" not in source_chunk_ids
    assert f"{TEST_PREFIX}_user_current_fee" in payload["answer"]


def test_upload_document_api_accepts_pdf_and_creates_queued_document():
    app = create_app()
    app.dependency_overrides[deps.get_current_user_id] = lambda: USER_A

    _cleanup_mongo()
    uploaded_document_id = None
    uploaded_job_id = None
    raw_storage_path = None
    try:
        _seed_mongo()
        _reset_motor_client_cache()
        client = TestClient(app)
        try:
            response = client.post(
                "/documents/upload",
                data={"session_id": SESSION_CURRENT},
                files={"file": ("api_test_upload.pdf", b"%PDF-1.4\n% test pdf content", "application/pdf")},
            )
        finally:
            client.close()
            _reset_motor_client_cache()

        assert response.status_code == 200
        payload = response.json()
        assert payload["filename"] == "api_test_upload.pdf"
        assert payload["status"] == "uploaded"
        assert payload["job_id"]
        uploaded_document_id = payload["document_id"]
        uploaded_job_id = payload["job_id"]
        raw_storage_path = payload["raw_storage_path"]
    finally:
        client, db = _get_sync_database()
        if uploaded_document_id:
            db.documents.delete_one({"_id": uploaded_document_id})
        if uploaded_job_id:
            db.ingestion_jobs.delete_one({"_id": uploaded_job_id})
        client.close()
        if raw_storage_path:
            raw_path = Path(raw_storage_path)
            if raw_path.exists():
                raw_path.unlink()
            parent = raw_path.parent
            while parent.exists() and parent.name not in {"user_upload", "raw", "storage"}:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        _cleanup_mongo()
