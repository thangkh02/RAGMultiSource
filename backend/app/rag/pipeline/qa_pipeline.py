from app.core.constants import RETRIEVAL_SCOPE_AUTO, RETRIEVAL_SCOPE_NEED_CLARIFICATION
from app.rag.generation.openai_llm import OpenAILLMService
from app.rag.generation.source_formatter import SourceFormatter
from app.rag.query import IntentRouter
from app.rag.rewrite import QueryRewriter
from app.rag.retrieval.context_validator import FALLBACK_NO_CONTEXT, ContextValidator
from app.rag.retrieval.resolvers import DocumentResolver, ScopeResolver
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.strategy import RetrievalStrategy
from app.schemas.common_schema import SourceItem
from langsmith import traceable


class QAPipeline:
    def __init__(self) -> None:
        self.intent_router = IntentRouter()
        self.scope_resolver = ScopeResolver()
        self.document_resolver = DocumentResolver()
        self.query_rewriter = QueryRewriter()
        self.retrieval_strategy = RetrievalStrategy()
        self.context_validator = ContextValidator()
        self.retriever = Retriever()
        self.llm = OpenAILLMService()
        self.source_formatter = SourceFormatter()

    async def run(
        self,
        question: str,
        user_id: str,
        session_id: str | None,
        scope: str,
        selected_document_ids: list[str] | None = None,
        conversation_state: dict | None = None,
    ) -> dict:
        return await self._run_traced(question, user_id, session_id, scope, selected_document_ids, conversation_state)

    @traceable(name="rag_qa_pipeline")
    async def _run_traced(
        self,
        question: str,
        user_id: str,
        session_id: str | None,
        scope: str,
        selected_document_ids: list[str] | None = None,
        conversation_state: dict | None = None,
    ) -> dict:
        conversation_state = conversation_state or {}
        intent_resolution = self.intent_router.route(question=question, conversation_state=conversation_state)
        query_rewrite = self.query_rewriter.rewrite_after_intent(
            question=question,
            intent_resolution=intent_resolution.model_dump(),
            conversation_state=conversation_state,
        )
        routing_question = query_rewrite.rewritten_question
        resolution = self.scope_resolver.resolve(
            question=routing_question,
            user_id=user_id,
            session_id=session_id,
            scope=scope if scope != RETRIEVAL_SCOPE_AUTO else RETRIEVAL_SCOPE_AUTO,
            selected_document_ids=selected_document_ids,
            conversation_state=conversation_state,
        )
        document_resolution = await self.document_resolver.resolve(
            scope=resolution.scope,
            metadata_filter=resolution.metadata_filter,
            user_id=user_id,
            session_id=session_id,
            detected_filename=resolution.detected_filename,
            detected_procedure_title=resolution.detected_procedure_title,
            selected_document_ids=selected_document_ids,
            conversation_state=conversation_state,
        )
        retrieval_plan = self.retrieval_strategy.plan(
            rewritten_question=query_rewrite.rewritten_question,
            intent_resolution=intent_resolution.model_dump(),
            scope=resolution.scope,
            metadata_filter=document_resolution.metadata_filter,
        )

        contexts: list[dict] = []
        branch_results: list[dict] = []
        if (
            resolution.scope == RETRIEVAL_SCOPE_NEED_CLARIFICATION
            or document_resolution.needs_clarification
            or not intent_resolution.needs_retrieval
            or not retrieval_plan.should_retrieve
        ):
            answer = (
                "Mình cần bạn làm rõ tài liệu muốn hỏi: file vừa upload, file cũ, tài liệu hệ thống, hoặc một file cụ thể."
                if resolution.scope == RETRIEVAL_SCOPE_NEED_CLARIFICATION or document_resolution.needs_clarification
                else FALLBACK_NO_CONTEXT
            )
            context_validation = self.context_validator.validate_all([])
        else:
            for branch in retrieval_plan.branches:
                branch_contexts = self.retriever.retrieve(
                    question=branch.query,
                    where_filter=branch.metadata_filter,
                    top_k=branch.top_k,
                )
                branch_results.append(
                    {
                        "name": branch.name,
                        "metadata_filter": branch.metadata_filter,
                        "contexts": branch_contexts,
                    }
                )
            context_validation = self.context_validator.validate_all(branch_results)
            contexts = context_validation.contexts
            if not context_validation.should_answer:
                answer = context_validation.fallback_answer or FALLBACK_NO_CONTEXT
            else:
                answer = self.llm.generate_answer(
                    question=query_rewrite.rewritten_question,
                    contexts=contexts,
                    answer_style=intent_resolution.answer_style,
                )

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
        answer = self.source_formatter.format_answer(answer, sources)
        return {
            "answer": answer,
            "sources": sources,
            "raw_contexts": contexts,
            "scope": resolution.scope,
            "intent_resolution": intent_resolution.model_dump(),
            "scope_resolution": resolution.model_dump(),
            "document_resolution": document_resolution.model_dump(),
            "query_rewrite": query_rewrite.model_dump(),
            "retrieval_plan": retrieval_plan.model_dump(),
            "context_validation": context_validation.model_dump(),
            "retrieval_filter": document_resolution.metadata_filter,
        }
