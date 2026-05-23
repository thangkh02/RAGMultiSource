import json
from pathlib import Path

from app.rag.rewrite import QueryRewriter


CASES_PATH = Path(__file__).parent / "fixtures" / "query_rewrite_cases.json"


def test_query_rewriter_json_cases():
    rewriter = QueryRewriter()
    rewriter.chain = None

    for case in json.loads(CASES_PATH.read_text(encoding="utf-8")):
        result = rewriter.rewrite(
            question=case["question"],
            intent_resolution=case["intent_resolution"],
            scope_resolution=case["scope_resolution"],
            document_resolution=case["document_resolution"],
            conversation_state=case["conversation_state"],
        )

        assert result.was_rewritten is case["expected_was_rewritten"], case["id"]
        if "expected_contains" in case:
            assert case["expected_contains"] in result.rewritten_question, case["id"]
        if "expected_rewritten_question" in case:
            assert result.rewritten_question == case["expected_rewritten_question"], case["id"]
