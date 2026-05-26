from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "true")


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.rag.graph.scope.scope_analyzer import ScopeAnalyzer
from tests.test_scope_analyzer_live import CASES


MODELS = (
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-3.1-flash-lite-preview",
)
TOKEN_BUDGETS = (64, 48)
WARMUP_RUNS = 1
MEASURED_RUNS = 5


def _ensure_live_llm() -> None:
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for live scope benchmark.")
    settings.SCOPE_RESOLVER_USE_LLM = True


def _run_case_set(analyzer: ScopeAnalyzer) -> dict[str, Any]:
    started_total = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case in CASES:
        started = time.perf_counter()
        resolution = analyzer.resolve(case.state)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = resolution.model_dump()
        results.append(
            {
                "case": case.name,
                "elapsed_ms": round(elapsed_ms, 2),
                "used_llm": resolution.used_llm,
                "action": payload["action"],
                "scope": payload["scope"],
                "confidence": payload["confidence"],
                "action_ok": payload["action"] == case.expected_action,
                "scope_ok": case.expected_scope is None or payload["scope"] == case.expected_scope,
            }
        )
    total_ms = round((time.perf_counter() - started_total) * 1000, 2)
    return {
        "results": results,
        "total_ms": total_ms,
        "all_action_ok": all(item["action_ok"] for item in results),
        "all_scope_ok": all(item["scope_ok"] for item in results),
        "all_used_llm": all(item["used_llm"] for item in results),
    }


def benchmark_scope_models() -> dict[str, Any]:
    _ensure_live_llm()
    original_model = settings.OPENROUTER_SCOPE_MODEL
    original_max_tokens = settings.OPENROUTER_SCOPE_MAX_TOKENS
    benchmark_runs: list[dict[str, Any]] = []
    try:
        for model in MODELS:
            for max_tokens in TOKEN_BUDGETS:
                settings.OPENROUTER_SCOPE_MODEL = model
                settings.OPENROUTER_SCOPE_MAX_TOKENS = max_tokens

                warmups: list[dict[str, Any]] = []
                for _ in range(WARMUP_RUNS):
                    analyzer = ScopeAnalyzer()
                    warmups.append(_run_case_set(analyzer))

                measured: list[dict[str, Any]] = []
                for _ in range(MEASURED_RUNS):
                    analyzer = ScopeAnalyzer()
                    measured.append(_run_case_set(analyzer))

                totals = [run["total_ms"] for run in measured]
                median_total_ms = round(statistics.median(totals), 2)
                p95_total_ms = round(max(totals), 2)
                per_case: dict[str, list[float]] = {case.name: [] for case in CASES}
                for run in measured:
                    for item in run["results"]:
                        per_case[item["case"]].append(item["elapsed_ms"])
                per_case_medians = {
                    case_name: round(statistics.median(samples), 2) if samples else None
                    for case_name, samples in per_case.items()
                }
                median_case_ms = round(statistics.median([value for value in per_case_medians.values() if value is not None]), 2)
                benchmark_runs.append(
                    {
                        "model": model,
                        "max_tokens": max_tokens,
                        "langsmith_tracing": os.environ.get("LANGSMITH_TRACING"),
                        "langchain_tracing_v2": os.environ.get("LANGCHAIN_TRACING_V2"),
                        "warmups": warmups,
                        "measured_runs": measured,
                        "per_case_medians_ms": per_case_medians,
                        "median_total_ms": median_total_ms,
                        "median_case_ms": median_case_ms,
                        "p95_total_ms": p95_total_ms,
                        "pass": all(
                            run["all_action_ok"] and run["all_scope_ok"] and run["all_used_llm"] for run in measured
                        )
                        and all(value is not None and value < 800 for value in per_case_medians.values()),
                    }
                )
    finally:
        settings.OPENROUTER_SCOPE_MODEL = original_model
        settings.OPENROUTER_SCOPE_MAX_TOKENS = original_max_tokens

    passing = [run for run in benchmark_runs if run["pass"]]
    fastest = min(passing, key=lambda run: run["median_case_ms"]) if passing else None
    best_effort = min(
        benchmark_runs,
        key=lambda run: (
            not all(
                measured["all_action_ok"] and measured["all_scope_ok"] and measured["all_used_llm"]
                for measured in run["measured_runs"]
            ),
            run["median_case_ms"],
        ),
    ) if benchmark_runs else None
    return {
        "criteria": {"per_case_median_ms_lt": 800, "all_action_ok": True, "all_scope_ok": True, "all_used_llm": True},
        "runs": benchmark_runs,
        "recommended": fastest,
        "best_effort": best_effort,
    }


if __name__ == "__main__":
    report = benchmark_scope_models()
    print(json.dumps(report, ensure_ascii=True, indent=2))
