import asyncio

from app.rag.pipeline.qa_pipeline import QAPipeline
from app.rag.retrieval.resolvers.document_resolver import DocumentResolution


class FakeDocumentResolver:
    async def resolve(self, **kwargs):
        return DocumentResolution(
            metadata_filter={
                "source_type": "system",
                "visibility": "global",
                "procedure_title": "Đăng ký thành lập doanh nghiệp tư nhân",
            },
            selected_document_ids=["sysdoc_1"],
            resolved_documents=[
                {
                    "document_id": "sysdoc_1",
                    "filename": "system.docx",
                    "source_type": "system",
                    "procedure_title": "Đăng ký thành lập doanh nghiệp tư nhân",
                }
            ],
            reason="fake resolved document",
        )


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, question: str, where_filter: dict | None, top_k: int = 5):
        self.calls.append({"question": question, "where_filter": where_filter, "top_k": top_k})
        return [
            {
                "id": "chunk_evidence_1",
                "content": "Thành phần hồ sơ gồm giấy đề nghị đăng ký doanh nghiệp tư nhân.",
                "similarity": 0.91,
                "metadata": {
                    "chunk_id": "chunk_evidence_1",
                    "document_id": "sysdoc_1",
                    "filename": "system.docx",
                    "source_type": "system",
                    "visibility": "global",
                    "procedure_title": "Đăng ký thành lập doanh nghiệp tư nhân",
                    "page_number": 1,
                    "section_title": "Thành phần hồ sơ",
                },
            },
            {
                "id": "chunk_wrong_source",
                "content": "Chunk không được dùng vì sai source_type.",
                "similarity": 0.95,
                "metadata": {
                    "chunk_id": "chunk_wrong_source",
                    "document_id": "doc_user_2",
                    "filename": "other.pdf",
                    "source_type": "user_upload",
                    "owner_user_id": "other_user",
                },
            },
        ]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_answer(self, question: str, contexts: list[dict], answer_style: str = "short_answer"):
        self.calls.append({"question": question, "contexts": contexts, "answer_style": answer_style})
        chunk_ids = [item["metadata"]["chunk_id"] for item in contexts]
        return f"Evidence chunks: {', '.join(chunk_ids)}"


def test_qa_pipeline_retrieves_and_keeps_evidence_chunk():
    pipeline = QAPipeline()
    pipeline.document_resolver = FakeDocumentResolver()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    result = asyncio.run(
        pipeline.run(
            question="Thủ tục đăng ký thành lập doanh nghiệp tư nhân cần hồ sơ gì?",
            user_id="user_1",
            session_id="sess_1",
            scope="auto",
            conversation_state={},
        )
    )

    assert len(pipeline.retriever.calls) == 1
    assert pipeline.retriever.calls[0]["where_filter"]["source_type"] == "system"
    assert pipeline.retriever.calls[0]["where_filter"]["procedure_title"] == "Đăng ký thành lập doanh nghiệp tư nhân"

    evidence_chunk_ids = [item["metadata"]["chunk_id"] for item in result["raw_contexts"]]
    assert evidence_chunk_ids == ["chunk_evidence_1"]
    assert result["sources"][0]["chunk_id"] == "chunk_evidence_1"
    assert result["sources"][0]["source_type"] == "system"
    assert "chunk_evidence_1" in result["answer"]
    assert "chunk_wrong_source" not in result["answer"]
    assert result["context_validation"]["rejected_count"] == 1


def test_qa_pipeline_returns_fallback_when_no_evidence_chunk_matches_filter():
    pipeline = QAPipeline()
    pipeline.document_resolver = FakeDocumentResolver()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()
    pipeline.context_validator.min_similarity = 0.99

    result = asyncio.run(
        pipeline.run(
            question="Thủ tục đăng ký thành lập doanh nghiệp tư nhân cần hồ sơ gì?",
            user_id="user_1",
            session_id="sess_1",
            scope="auto",
            conversation_state={},
        )
    )

    assert result["raw_contexts"] == []
    assert result["sources"] == []
    assert result["answer"] == "Không tìm thấy thông tin này trong tài liệu phù hợp."
    assert pipeline.llm.calls == []
