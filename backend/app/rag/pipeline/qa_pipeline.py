from app.core.constants import RETRIEVAL_SCOPE_AUTO, RETRIEVAL_SCOPE_NEED_CLARIFICATION
from app.rag.generation.openai_llm import OpenAILLMService
from app.rag.generation.source_formatter import SourceFormatter
from app.rag.graph import RAGGraphRunner
from app.rag.query import IntentRouter
from app.rag.rewrite import QueryRewriter, RewriteGate
from app.rag.retrieval.context_validator import FALLBACK_NO_CONTEXT, ContextValidator
from app.rag.retrieval.resolvers import DocumentResolver
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.strategy import RetrievalStrategy
from langsmith import traceable


class QAPipeline:
    def __init__(self) -> None:
        self.rewrite_gate = RewriteGate()
        self.intent_router = IntentRouter()
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
        graph_result = await RAGGraphRunner(self).run(
            {
                "original_query": question,
                "user_id": user_id,
                "session_id": session_id,
                "requested_scope": scope if scope != RETRIEVAL_SCOPE_AUTO else RETRIEVAL_SCOPE_AUTO,
                "selected_document_ids": selected_document_ids or [],
                "runtime_context": conversation_state,
            }
        )
        scope_resolution = graph_result.get("scope_resolution") or {}
        document_resolution = graph_result.get("document_resolution") or {}
        return {
            "answer": graph_result.get("answer", FALLBACK_NO_CONTEXT),
            "sources": graph_result.get("sources", []),
            "raw_contexts": graph_result.get("raw_contexts", []),
            "scope": scope_resolution.get("scope", RETRIEVAL_SCOPE_NEED_CLARIFICATION),
            "intent_resolution": graph_result.get("intent_resolution", {}),
            "scope_resolution": scope_resolution,
            "document_resolution": document_resolution,
            "rewrite_gate": graph_result.get("rewrite_gate", {}),
            "query_rewrite": graph_result.get("query_rewrite", {}),
            "retrieval_plan": graph_result.get("retrieval_plan", {}),
            "context_validation": graph_result.get("context_validation", {}),
            "retrieval_filter": graph_result.get("metadata_filter", document_resolution.get("metadata_filter", {})),
        }
