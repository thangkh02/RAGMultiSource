from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.rag.graph.nodes import RAGGraphNodes
from app.rag.graph.state import RAGState


class RAGGraphRunner:
    def __init__(self, pipeline: Any) -> None:
        self.nodes = RAGGraphNodes(pipeline)
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("load_context", self.nodes.load_context_node)
        graph.add_node("rewrite_detector", self.nodes.rewrite_detector_node)
        graph.add_node("rewrite_query", self.nodes.rewrite_query_node)
        graph.add_node("use_original_query", self.nodes.use_original_query_node)
        graph.add_node("intent_router", self.nodes.intent_router_node)
        graph.add_node("retrieval_planner", self.nodes.retrieval_planner_node)
        graph.add_node("scope_resolver", self.nodes.scope_resolver_node)
        graph.add_node("document_resolver", self.nodes.document_resolver_node)
        graph.add_node("candidate_selector", self.nodes.candidate_selector_node)
        graph.add_node("build_filter", self.nodes.build_filter_node)
        graph.add_node("retrieval_strategy", self.nodes.retrieval_strategy_node)
        graph.add_node("retrieval", self.nodes.retrieval_node)
        graph.add_node("evidence_validation", self.nodes.evidence_validation_node)
        graph.add_node("answer", self.nodes.answer_node)
        graph.add_node("no_context", self.nodes.no_context_node)
        graph.add_node("direct_answer", self.nodes.direct_answer_node)
        graph.add_node("clarification", self.nodes.clarification_node)
        graph.add_node("unsupported", self.nodes.unsupported_node)
        graph.add_node("update_state", self.nodes.update_state_node)

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "rewrite_detector")
        graph.add_conditional_edges(
            "rewrite_detector",
            self.nodes.route_after_rewrite_gate,
            {"rewrite_query": "rewrite_query", "use_original_query": "use_original_query"},
        )
        graph.add_edge("rewrite_query", "intent_router")
        graph.add_edge("use_original_query", "intent_router")
        graph.add_conditional_edges(
            "intent_router",
            self.nodes.route_after_intent,
            {
                "direct_answer": "direct_answer",
                "clarification": "clarification",
                "unsupported": "unsupported",
                "retrieval_planner": "retrieval_planner",
            },
        )
        graph.add_conditional_edges(
            "retrieval_planner",
            self.nodes.route_after_planner,
            {
                "direct_answer": "direct_answer",
                "clarification": "clarification",
                "unsupported": "unsupported",
                "scope_resolver": "scope_resolver",
            },
        )
        graph.add_conditional_edges(
            "scope_resolver",
            self.nodes.route_after_scope_resolution,
            {
                "build_filter": "build_filter",
                "direct_answer": "direct_answer",
                "clarification": "clarification",
                "document_resolver": "document_resolver",
            },
        )
        graph.add_edge("document_resolver", "candidate_selector")
        graph.add_conditional_edges(
            "candidate_selector",
            self.nodes.route_after_candidate_selector,
            {"clarification": "clarification", "build_filter": "build_filter"},
        )
        graph.add_edge("build_filter", "retrieval_strategy")
        graph.add_edge("retrieval_strategy", "retrieval")
        graph.add_edge("retrieval", "evidence_validation")
        graph.add_conditional_edges(
            "evidence_validation",
            self.nodes.route_after_evidence_validation,
            {"answer": "answer", "no_context": "no_context"},
        )
        graph.add_edge("answer", "update_state")
        graph.add_edge("no_context", "update_state")
        graph.add_edge("direct_answer", "update_state")
        graph.add_edge("clarification", "update_state")
        graph.add_edge("unsupported", "update_state")
        graph.add_edge("update_state", END)
        return graph

    @traceable(name="rag_langgraph_run")
    async def run(self, initial_state: RAGState) -> RAGState:
        return await self.graph.ainvoke(initial_state)
