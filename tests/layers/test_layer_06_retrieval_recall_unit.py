from __future__ import annotations

from types import SimpleNamespace

from eval.retrieval_recall_eval import RetrievalHitSpec, hit_satisfied_by_any_doc, recall_hit_ratio


def test_hit_satisfied_text_any():
    docs = [
        SimpleNamespace(page_content="We propose Multi-Head Attention.", metadata={}),
    ]
    hit = RetrievalHitSpec(match_any_text=["multi-head attention", "foo"])
    assert hit_satisfied_by_any_doc(docs, hit) is True


def test_hit_requires_metadata_when_given():
    docs = [
        SimpleNamespace(page_content="Multi-Head Attention", metadata={"arxiv_id": "9999.99999"}),
        SimpleNamespace(page_content="Multi-Head Attention", metadata={"arxiv_id": "1706.03762"}),
    ]
    hit = RetrievalHitSpec(
        match_any_text=["multi-head"],
        metadata_contains={"arxiv_id": "1706.03762"},
    )
    assert hit_satisfied_by_any_doc(docs, hit) is True


def test_hit_fails_when_no_substring():
    docs = [SimpleNamespace(page_content="unrelated", metadata={})]
    hit = RetrievalHitSpec(match_any_text=["nope"])
    assert hit_satisfied_by_any_doc(docs, hit) is False


def test_recall_hit_ratio_partial():
    docs = [SimpleNamespace(page_content="alpha", metadata={})]
    hits = [
        RetrievalHitSpec(match_any_text=["alpha"]),
        RetrievalHitSpec(match_any_text=["missing"]),
    ]
    ratio, sat, total = recall_hit_ratio(docs, hits)
    assert total == 2
    assert sat == 1
    assert ratio == 0.5


def test_empty_hits_is_full_recall():
    docs = [SimpleNamespace(page_content="x", metadata={})]
    ratio, sat, total = recall_hit_ratio(docs, [])
    assert ratio == 1.0 and sat == 0 and total == 0
