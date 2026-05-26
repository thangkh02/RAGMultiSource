from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "true")


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.constants import RETRIEVAL_SCOPE_SYSTEM_PROCEDURE
from app.rag.graph.scope.scope_analyzer import ScopeAnalyzer


@dataclass
class ScopeCase:
    name: str
    query: str
    state: dict[str, Any]
    expected_action: str
    expected_scope: str | None


CASES: list[ScopeCase] = [
    ScopeCase(
        name="system_procedure_fee",
        query="le phi khi cap lai thong bao van ban buu chinh la bao nhieu?",
        state={
            "original_query": "le phi khi cap lai thong bao van ban buu chinh la bao nhieu?",
            "final_query": "le phi khi cap lai thong bao van ban buu chinh la bao nhieu?",
            "was_rewritten": False,
            "intent_resolution": {"intent": "find_information", "needs_retrieval": True},
            "retrieval_plan": {"action": "default"},
            "runtime_context": {"last_resolved_context": {}, "current_session_docs": [], "active_document_ids": []},
            "selected_document_ids": [],
            "document_candidates": [],
            "requested_scope": "auto",
        },
        expected_action="resolve_document",
        expected_scope="system_only",
    ),
    ScopeCase(
        name="current_uploads",
        query="file nay toi vua gui co noi ve le phi khong?",
        state={
            "original_query": "file nay toi vua gui co noi ve le phi khong?",
            "final_query": "file nay toi vua gui co noi ve le phi khong?",
            "was_rewritten": False,
            "intent_resolution": {"intent": "find_information", "needs_retrieval": True},
            "retrieval_plan": {"action": "default"},
            "runtime_context": {"last_resolved_context": {}, "current_session_docs": [], "active_document_ids": []},
            "selected_document_ids": [],
            "document_candidates": [],
            "requested_scope": "auto",
        },
        expected_action="resolve_document",
        expected_scope="current_uploads_only",
    ),
    ScopeCase(
        name="past_uploads",
        query="tai lieu toi upload hom qua co noi ve hoc may khong?",
        state={
            "original_query": "tai lieu toi upload hom qua co noi ve hoc may khong?",
            "final_query": "tai lieu toi upload hom qua co noi ve hoc may khong?",
            "was_rewritten": False,
            "intent_resolution": {"intent": "find_information", "needs_retrieval": True},
            "retrieval_plan": {"action": "default"},
            "runtime_context": {"last_resolved_context": {}, "current_session_docs": [], "active_document_ids": []},
            "selected_document_ids": [],
            "document_candidates": [],
            "requested_scope": "auto",
        },
        expected_action="resolve_document",
        expected_scope="past_uploads_only",
    ),
    ScopeCase(
        name="mixed_compare",
        query="doi chieu file toi gui tuan truoc voi quy dinh he thong",
        state={
            "original_query": "doi chieu file toi gui tuan truoc voi quy dinh he thong",
            "final_query": "doi chieu file toi gui tuan truoc voi quy dinh he thong",
            "was_rewritten": False,
            "intent_resolution": {"intent": "compare_documents", "needs_retrieval": True},
            "retrieval_plan": {"action": "default"},
            "runtime_context": {"last_resolved_context": {}, "current_session_docs": [], "active_document_ids": []},
            "selected_document_ids": [],
            "document_candidates": [],
            "requested_scope": "auto",
        },
        expected_action="mixed_retrieval",
        expected_scope="mixed",
    ),
    ScopeCase(
        name="reuse_last_filter",
        query="can chuan bi giay to gi?",
        state={
            "original_query": "can chuan bi giay to gi?",
            "final_query": "can chuan bi giay to gi?",
            "was_rewritten": True,
            "intent_resolution": {"intent": "find_information", "needs_retrieval": True},
            "retrieval_plan": {"action": "reuse_last_filter"},
            "runtime_context": {
                "last_resolved_context": {
                    "filter": {"source_type": "system", "visibility": "global"},
                    "scope": RETRIEVAL_SCOPE_SYSTEM_PROCEDURE,
                    "source_type": "system",
                    "procedure_title": "dang ky ket hon",
                    "filename": None,
                    "document_id": None,
                },
                "current_session_docs": [],
                "active_document_ids": [],
            },
            "selected_document_ids": [],
            "document_candidates": [],
            "requested_scope": "auto",
        },
        expected_action="reuse_last_filter",
        expected_scope=None,
    ),
]


def _prepare_live_llm() -> None:
    if not settings.OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY is not set; live scope test requires OpenRouter.")
    settings.SCOPE_RESOLVER_USE_LLM = True


def run_live_scope_cases() -> dict[str, Any]:
    _prepare_live_llm()
    analyzer = ScopeAnalyzer()
    if analyzer.chain is None:
        raise AssertionError("ScopeAnalyzer did not initialize the LLM chain.")

    start_total = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case in CASES:
        started = time.perf_counter()
        resolution = analyzer.resolve(case.state)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = resolution.model_dump()
        results.append(
            {
                "case": case.name,
                "query": case.query,
                "elapsed_ms": round(elapsed_ms, 2),
                "chain_is_none": analyzer.chain is None,
                "used_llm": resolution.used_llm,
                "llm_status": resolution.llm_status,
                "llm_failure_stage": resolution.llm_failure_stage,
                "llm_failure_detail": resolution.llm_failure_detail,
                "llm_used_recovered_json": resolution.llm_used_recovered_json,
                "llm_raw_preview": resolution.llm_raw_preview,
                "action": payload["action"],
                "expected_action": case.expected_action,
                "action_ok": payload["action"] == case.expected_action,
                "scope": payload["scope"],
                "expected_scope": case.expected_scope,
                "scope_ok": case.expected_scope is None or payload["scope"] == case.expected_scope,
                "targets": payload["targets"],
                "confidence": payload["confidence"],
            }
        )

    total_ms = round((time.perf_counter() - start_total) * 1000, 2)
    return {
        "results": results,
        "total_ms": total_ms,
        "all_action_ok": all(item["action_ok"] for item in results),
        "all_scope_ok": all(item["scope_ok"] for item in results),
    }


def test_scope_analyzer_live_openrouter() -> None:
    report = run_live_scope_cases()
    assert report["all_action_ok"], json.dumps(report, ensure_ascii=True, indent=2)
    assert report["all_scope_ok"], json.dumps(report, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    report = run_live_scope_cases()
    print(json.dumps(report, ensure_ascii=True, indent=2))
