from __future__ import annotations

from tools.rag.retrieval_judge import _parse_judge_json, judge_retrieval_context


def test_parse_judge_json_strips_fence():
    raw = '```json\n{"score":8,"sufficient":true,"reason":"ok","web_queries":[]}\n```'
    d = _parse_judge_json(raw)
    assert d is not None
    assert d.get("sufficient") is True
    assert d.get("web_queries") == []


def test_parse_judge_json_invalid():
    assert _parse_judge_json("not json") is None


def test_judge_empty_question_or_context_skips_llm():
    class _NoCallLLM:
        def invoke(self, *_a, **_k):
            raise AssertionError("LLM should not be called")

    out = judge_retrieval_context(_NoCallLLM(), "", [])
    assert out["sufficient"] is True
    assert out["should_supplement_web"] is False

    out2 = judge_retrieval_context(_NoCallLLM(), "hello?", [])
    assert out2["sufficient"] is True
