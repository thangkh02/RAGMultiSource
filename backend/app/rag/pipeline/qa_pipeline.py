from app.core.constants import RETRIEVAL_SCOPE_AUTO
from app.rag.generation.openai_llm import OpenAILLMService
from app.rag.retrieval.filters import build_retrieval_filter
from app.rag.retrieval.query_router import QueryRouter
from app.rag.retrieval.retriever import Retriever
from app.schemas.common_schema import SourceItem
from langsmith import traceable


class QAPipeline:
    def __init__(self) -> None:
        self.router = QueryRouter()
        self.retriever = Retriever()
        self.llm = OpenAILLMService()

    def run(
        self,
        question: str,
        user_id: str,
        session_id: str | None,
        scope: str,
        selected_document_ids: list[str] | None = None,
    ) -> dict:
        return self._run_traced(question, user_id, session_id, scope, selected_document_ids)

    @traceable(name="rag_qa_pipeline")
    def _run_traced(
        self,
        question: str,
        user_id: str,
        session_id: str | None,
        scope: str,
        selected_document_ids: list[str] | None = None,
    ) -> dict:
        resolved_scope = self.router.route(question) if scope == RETRIEVAL_SCOPE_AUTO else scope
        where_filter = build_retrieval_filter(
            scope=resolved_scope,
            user_id=user_id,
            session_id=session_id,
            selected_document_ids=selected_document_ids,
        )
        contexts = self.retriever.retrieve(question=question, where_filter=where_filter)
        answer = self.llm.generate_answer(question=question, contexts=contexts)
        sources = [
            SourceItem(
                document_id=item["metadata"].get("document_id", ""),
                chunk_id=item["metadata"].get("chunk_id", item.get("id", "")),
                filename=item["metadata"].get("filename", ""),
                source_type=item["metadata"].get("source_type", ""),
                procedure_title=item["metadata"].get("procedure_title"),
                page_number=item["metadata"].get("page_number"),
                page_source=item["metadata"].get("page_source"),
                section_title=item["metadata"].get("section_title"),
                score=item.get("similarity"),
                visibility=item["metadata"].get("visibility"),
                owner_user_id=item["metadata"].get("owner_user_id"),
                session_id=item["metadata"].get("session_id"),
            ).model_dump()
            for item in contexts
        ]
        return {"answer": answer, "sources": sources, "raw_contexts": contexts, "scope": resolved_scope}
