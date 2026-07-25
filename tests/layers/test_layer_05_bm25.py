from __future__ import annotations

import pytest

from tools.rag.bm25 import build_bm25_index, bm25_top_k, default_tokenizer


pytest.importorskip("rank_bm25")


def test_default_tokenizer_mixed():
    toks = default_tokenizer("Hello 世界")
    assert "hello" in toks
    assert "世" in toks


def test_bm25_ranking():
    docs = ["cat sits on mat", "dog runs fast", "the mat is wool"]
    bm25, doc_list = build_bm25_index(docs)
    assert bm25 is not None
    hits = bm25_top_k(bm25, doc_list, "mat", k=2)
    assert len(hits) == 2
    texts = [h[0] for h in hits]
    assert any("mat" in str(t) for t in texts)
