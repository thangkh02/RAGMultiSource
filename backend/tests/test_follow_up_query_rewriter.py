from app.rag.rewrite import QueryRewriter


def test_follow_up_rewrite_uses_history_and_fallback_when_llm_not_configured(monkeypatch):
    rewriter = QueryRewriter()
    monkeypatch.setattr(rewriter, "chain", None)

    result = rewriter.rewrite_after_intent(
        question="còn lệ phí thì sao?",
        intent_resolution={"intent": "follow_up", "is_follow_up": True},
        conversation_state={
            "last_filename": "ho_so_alpha.pdf",
            "recent_chat_history": [
                {"role": "user", "content": "File ho_so_alpha.pdf nói gì?"},
                {"role": "assistant", "content": "Tài liệu nói về hồ sơ Alpha."},
            ],
        },
    )

    assert result.was_rewritten is True
    assert result.stage == "post_intent"
    assert result.used_llm is False
    assert "ho_so_alpha.pdf" in result.rewritten_question


def test_follow_up_rewrite_passes_through_without_chat_history():
    result = QueryRewriter().rewrite_after_intent(
        question="còn lệ phí thì sao?",
        intent_resolution={"intent": "follow_up", "is_follow_up": True},
        conversation_state={"last_filename": "ho_so_alpha.pdf"},
    )

    assert result.was_rewritten is False
    assert result.rewritten_question == "còn lệ phí thì sao?"


def test_query_rewrite_does_not_run_for_non_follow_up():
    question = "Thủ tục đăng ký doanh nghiệp tư nhân cần hồ sơ gì?"
    result = QueryRewriter().rewrite_after_intent(
        question=question,
        intent_resolution={"intent": "ask_question", "is_follow_up": False},
        conversation_state={
            "recent_chat_history": [
                {"role": "user", "content": "Câu trước"},
            ]
        },
    )

    assert result.was_rewritten is False
    assert result.rewritten_question == question
