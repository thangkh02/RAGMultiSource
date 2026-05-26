from app.core.constants import RETRIEVAL_SCOPE_SYSTEM_PROCEDURE, SOURCE_TYPE_SYSTEM, SOURCE_TYPE_USER_UPLOAD
from app.rag.graph.nodes import RAGGraphNodes
from app.rag.pipeline.qa_pipeline import QAPipeline
from app.rag.query.intent_router import IntentResolution, IntentRouter
from app.rag.retrieval.resolvers.conversation_state import ConversationStateManager
from app.rag.rewrite.rewrite_gate import RewriteGate


def _nodes() -> RAGGraphNodes:
    return RAGGraphNodes(QAPipeline())


def _state_with_last_context(query: str, *, include_filter: bool = True) -> dict:
    last_context = {
        "scope": "system_only",
        "source_type": SOURCE_TYPE_SYSTEM,
        "procedure_title": "dang ky ket hon",
        "document_id": "sysdoc_1",
    }
    if include_filter:
        last_context["filter"] = {"source_type": SOURCE_TYPE_SYSTEM, "procedure_title": "dang ky ket hon"}
    return {
        "original_query": query,
        "final_query": query,
        "was_rewritten": False,
        "retrieval_plan": {"action": "default"},
        "runtime_context": {"last_resolved_context": last_context},
    }


def test_intent_router_routes_document_questions_directly_to_document_resolver():
    nodes = _nodes()

    route = nodes.route_after_intent(
        {
            "intent_resolution": {"intent": "ask_question", "needs_retrieval": True, "is_follow_up": False},
            "scope_resolution": {"action": "resolve_document", "scope": "system_only"},
        }
    )

    assert route == "document_resolver"


def test_intent_router_node_builds_scope_resolution_for_system_query(monkeypatch):
    nodes = _nodes()

    def fake_route(question, conversation_state=None):
        return IntentResolution(
            intent="ask_question",
            answer_style="short_answer",
            needs_retrieval=True,
            is_follow_up=False,
            action="resolve_document",
            scope="system_only",
            targets=[
                {
                    "source_type": SOURCE_TYPE_SYSTEM,
                    "session_scope": None,
                    "procedure_title_hint": "cap lai thong bao van ban buu chinh",
                    "document_name_hint": None,
                    "time_hint": None,
                }
            ],
            confidence=0.92,
        )

    monkeypatch.setattr(nodes.pipeline.intent_router, "route", fake_route)
    result = nodes.intent_router_node({"final_query": "le phi cap lai thong bao la bao nhieu", "runtime_context": {}})

    assert result["scope_resolution"]["scope"] == "system_only"
    assert result["scope_resolution"]["action"] == "resolve_document"
    assert result["retrieval_plan"]["target_scope"] == "system_only"


def test_intent_router_rule_routes_current_upload_query():
    result = IntentRouter()._route_by_rules("file toi vua upload noi gi")

    assert result.scope == "current_uploads_only"
    assert result.targets[0]["source_type"] == SOURCE_TYPE_USER_UPLOAD
    assert result.targets[0]["session_scope"] == "current_session"


def test_intent_router_rule_routes_past_upload_time_hint():
    result = IntentRouter()._route_by_rules("tai lieu toi upload hom qua noi gi")

    assert result.scope == "past_uploads_only"
    assert result.targets[0]["session_scope"] == "past_sessions"
    assert result.targets[0]["time_hint"] == "yesterday"


def test_intent_router_rule_routes_last_month_upload_time_hint():
    result = IntentRouter()._route_by_rules("tai lieu toi upload thang truoc noi gi")

    assert result.scope == "past_uploads_only"
    assert result.targets[0]["time_hint"] == "last_month"


def test_intent_router_rule_routes_mixed_query():
    result = IntentRouter()._route_by_rules("doi chieu file toi upload voi quy dinh he thong")

    assert result.action == "mixed_retrieval"
    assert result.scope == "mixed"
    assert [target["source_type"] for target in result.targets] == [SOURCE_TYPE_SYSTEM, SOURCE_TYPE_USER_UPLOAD]


def test_intent_router_rule_reuses_last_filter_for_safe_follow_up():
    result = IntentRouter()._route_by_rules(
        "le phi bao nhieu",
        {
            "last_resolved_context": {
                "scope": "system_only",
                "source_type": SOURCE_TYPE_SYSTEM,
                "filter": {"source_type": SOURCE_TYPE_SYSTEM, "procedure_title": "dang ky ket hon"},
            }
        },
    )

    assert result.is_follow_up is True
    assert result.action == "reuse_last_filter"
    assert result.targets == []


def test_route_after_intent_sends_follow_up_to_rewrite_query():
    nodes = _nodes()

    route = nodes.route_after_intent(
        {
            "intent_resolution": {"intent": "ask_question", "needs_retrieval": True, "is_follow_up": True},
            "scope_resolution": {"action": "resolve_document", "scope": "system_only"},
        }
    )

    assert route == "rewrite_query"


def test_route_after_intent_reuse_goes_to_build_filter():
    nodes = _nodes()

    route = nodes.route_after_intent(
        {
            "intent_resolution": {"intent": "ask_question", "needs_retrieval": True, "is_follow_up": True},
            "scope_resolution": {"action": "reuse_last_filter", "scope": "system_only"},
        }
    )

    assert route == "build_filter"


def test_route_after_intent_general_query_goes_direct_answer():
    nodes = _nodes()

    route = nodes.route_after_intent(
        {
            "intent_resolution": {"intent": "general_query", "needs_retrieval": False},
            "scope_resolution": {"action": "direct_answer", "scope": "none"},
        }
    )

    assert route == "direct_answer"


def test_route_after_intent_need_clarification_goes_clarification():
    nodes = _nodes()

    route = nodes.route_after_intent(
        {
            "intent_resolution": {"intent": "need_clarification", "needs_retrieval": False},
            "scope_resolution": {"action": "need_clarification", "scope": "need_clarification"},
        }
    )

    assert route == "clarification"


def test_rewrite_query_node_only_rewrites_after_intent_follow_up(monkeypatch):
    nodes = _nodes()

    def fake_rewrite(question, intent_resolution, scope_resolution=None, document_resolution=None, conversation_state=None):
        assert intent_resolution["is_follow_up"] is True
        from app.rag.rewrite import QueryRewrite

        return QueryRewrite(
            original_question=question,
            rewritten_question="Trong thu tuc dang ky ket hon, le phi bao nhieu?",
            was_rewritten=True,
            reason="test",
            stage="post_intent",
            used_llm=False,
        )

    monkeypatch.setattr(nodes.pipeline.query_rewriter, "rewrite", fake_rewrite)
    result = nodes.rewrite_query_node(
        {
            "original_query": "le phi bao nhieu",
            "intent_resolution": {"is_follow_up": True},
            "scope_resolution": {"scope": "system_only"},
            "runtime_context": {},
        }
    )

    assert result["was_rewritten"] is True
    assert result["final_query"] == "Trong thu tuc dang ky ket hon, le phi bao nhieu?"


def test_candidate_selector_clarifies_multiple_system_candidates():
    nodes = _nodes()

    result = nodes.candidate_selector_node(
        {
            "scope_resolution": {"scope": "system_only"},
            "document_candidates": [
                {"document_id": "doc_1", "filename": "tam_tru_1.pdf"},
                {"document_id": "doc_2", "filename": "tam_tru_2.pdf"},
            ],
            "document_resolution": {"selected_document_ids": ["doc_1", "doc_2"]},
        }
    )

    assert result["candidate_selection"]["confident"] is False
    assert result["candidate_selection"]["needs_clarification"] is True


def test_candidate_selector_allows_multiple_upload_documents_for_time_scope():
    nodes = _nodes()

    result = nodes.candidate_selector_node(
        {
            "scope_resolution": {"scope": "past_uploads_only"},
            "document_candidates": [
                {"document_id": "doc_1", "filename": "week_1.pdf"},
                {"document_id": "doc_2", "filename": "week_2.pdf"},
            ],
            "document_resolution": {"selected_document_ids": ["doc_1", "doc_2"]},
        }
    )

    assert result["candidate_selection"]["confident"] is True
    assert result["candidate_selection"]["needs_clarification"] is False
    assert result["candidate_selection"]["selected_document_ids"] == ["doc_1", "doc_2"]


def test_mixed_evidence_validation_reports_missing_system_branch():
    nodes = _nodes()

    result = nodes.evidence_validation_node(
        {
            "retrieval_plan": {"mode": "hybrid_compare"},
            "branch_results": [
                {
                    "name": "system_chunks",
                    "metadata_filter": {"source_type": SOURCE_TYPE_SYSTEM},
                    "contexts": [],
                },
                {
                    "name": "user_upload_chunks",
                    "metadata_filter": {"source_type": SOURCE_TYPE_USER_UPLOAD, "owner_user_id": "user_1"},
                    "contexts": [
                        {
                            "id": "chunk_user_1",
                            "content": "File upload co le phi noi bo la 75.000 dong.",
                            "similarity": 0.9,
                            "metadata": {
                                "chunk_id": "chunk_user_1",
                                "document_id": "doc_1",
                                "source_type": SOURCE_TYPE_USER_UPLOAD,
                                "owner_user_id": "user_1",
                            },
                        }
                    ],
                },
            ],
        }
    )

    assert result["context_validation"]["should_answer"] is True
    assert result["mixed_branch_warnings"]


def test_build_filter_node_builds_deterministic_filter_after_scope_resolution():
    nodes = _nodes()

    result = nodes.build_filter_node(
        {
            "user_id": "user_1",
            "session_id": "sess_1",
            "scope_resolution": {
                "scope": "current_session_uploads",
                "should_reuse_last_filter": False,
            },
            "document_resolution": {"selected_document_ids": ["doc_1"]},
        }
    )

    assert result["metadata_filter"]["$and"][0]["$and"] == [
        {"source_type": SOURCE_TYPE_USER_UPLOAD},
        {"owner_user_id": "user_1"},
        {"session_id": "sess_1"},
    ]
    assert result["metadata_filter"]["$and"][1] == {"document_id": {"$in": ["doc_1"]}}


def test_intent_router_guard_keeps_admin_fee_question_in_rag_path(monkeypatch):
    router = IntentRouter()

    def fake_llm(_question, conversation_state=None):
        return IntentResolution(
            intent="general_query",
            needs_retrieval=False,
            action="direct_answer",
            scope="none",
            confidence=0.82,
            matched_rules=["llm_intent_router"],
            reason="LLM false negative.",
        )

    monkeypatch.setattr(router, "_route_with_llm", fake_llm)

    result = router.route("le phi khi cap lai thong bao van ban buu chinh la bao nhieu")

    assert result.intent == "ask_question"
    assert result.needs_retrieval is True
    assert "llm_false_negative_guard" in result.matched_rules


def test_rewrite_gate_fallback_detects_procedure_follow_up_without_llm(monkeypatch):
    gate = RewriteGate()
    monkeypatch.setattr(gate, "chain", None)

    result = gate.decide(
        "the trinh tu thuc hien nhu nao",
        {
            "last_resolved_context": {
                "scope": RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
                "filter": {"source_type": "system", "procedure_title": "Cap lai van ban xac nhan thong bao hoat dong buu chinh"},
            },
            "last_procedure_title": "Cap lai van ban xac nhan thong bao hoat dong buu chinh",
        },
    )

    assert result.needs_rewrite is True
    assert "fallback_follow_up" in result.matched_rules


def test_intent_router_treats_procedure_sequence_as_follow_up_document_question():
    result = IntentRouter()._route_by_rules(
        "the trinh tu thuc hien nhu nao",
        {
            "last_resolved_context": {
                "scope": "system_only",
                "filter": {"source_type": SOURCE_TYPE_SYSTEM, "procedure_title": "cap lai thong bao"},
            }
        },
    )

    assert result.intent == "ask_question"
    assert result.needs_retrieval is True
    assert result.is_follow_up is True


def test_conversation_state_preserves_last_context_on_general_query():
    manager = ConversationStateManager()
    previous_context = {
        "scope": "system_procedure",
        "source_type": "system",
        "procedure_title": "Cap lai van ban xac nhan thong bao hoat dong buu chinh",
        "document_id": "sysdoc_1",
        "filter": {"source_type": "system", "document_id": {"$in": ["sysdoc_1"]}},
    }

    result = manager.update_after_answer(
        state={"last_resolved_context": previous_context},
        intent="general_query",
        scope="general_query",
        sources=[],
        selected_document_ids=[],
        rewritten_question="the trinh tu thuc hien nhu nao",
        retrieval_filter={},
    )

    assert result["last_resolved_context"] == previous_context
