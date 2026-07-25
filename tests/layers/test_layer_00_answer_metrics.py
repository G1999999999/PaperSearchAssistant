from __future__ import annotations

from eval.answer_metrics import keyword_recall


def test_keyword_recall_all_hit():
    score, missing = keyword_recall("Attention Is All You Need", ["attention", "need"])
    assert score == 1.0
    assert missing == []


def test_keyword_recall_partial():
    score, missing = keyword_recall("Hello", ["hello", "world"])
    assert score == 0.5
    assert "world" in missing
