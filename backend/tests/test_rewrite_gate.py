from app.rag.rewrite import RewriteGate


class FakeGateChain:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def invoke(self, payload: dict) -> str:
        self.calls.append(payload)
        return self.response


def test_rewrite_gate_uses_llm_and_triggers_for_follow_up_with_history_and_active_document():
    gate = RewriteGate()
    gate.chain = FakeGateChain(
        '{"needs_rewrite": true, "reason": "follow-up needs previous file", "matched_rules": ["follow_up", "active_document"]}'
    )

    decision = gate.decide(
        original_query="còn lệ phí thì sao?",
        conversation_state={
            "last_filename": "ho_so_alpha.pdf",
            "recent_chat_history": [
                {"role": "user", "content": "File ho_so_alpha.pdf nói gì?"},
                {"role": "assistant", "content": "File nói về hồ sơ Alpha."},
            ],
        },
    )

    assert decision.needs_rewrite is True
    assert decision.used_llm is True
    assert "active_document" in decision.matched_rules
    assert gate.chain.calls[0]["active_document"] == "ho_so_alpha.pdf"


def test_rewrite_gate_defaults_to_no_rewrite_when_llm_unavailable():
    gate = RewriteGate()
    gate.chain = None
    decision = gate.decide(original_query="còn lệ phí thì sao?", conversation_state={"last_filename": "ho_so_alpha.pdf"})

    assert decision.needs_rewrite is True
    assert decision.used_llm is False
