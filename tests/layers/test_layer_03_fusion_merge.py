from __future__ import annotations

from tools.rag.math_utils import merge_ranked_lists
from tools.rag.bm25 import rrf_fusion
from tools.retrieval.fusion import weighted_rrf_fusion


def test_merge_ranked_lists_dedup_min_score():
    merged = merge_ranked_lists(
        [
            [("a", 0.5), ("b", 0.3)],
            [("b", 0.2), ("c", 0.4)],
        ]
    )
    # 同一 item 保留全局最小 score；最终按 score 升序：b(0.2) < c(0.4) < a(0.5)
    keys = [x[0] for x in merged]
    assert keys == ["b", "c", "a"]
    assert dict(merged)["b"] == 0.2


def test_rrf_fusion_order():
    a, b, c = "doc_a", "doc_b", "doc_c"
    # list1: a best; list2: b best — RRF should still rank both
    fused = rrf_fusion(
        [
            [(a, 1.0), (b, 0.5)],
            [(b, 1.0), (c, 0.5)],
        ],
        k=60,
    )
    assert len(fused) == 3
    # Winners appear with highest RRF contribution
    top_keys = {fused[0][0], fused[1][0]}
    assert top_keys <= {a, b}


def test_weighted_rrf_channel_weight():
    d1, d2 = "x1", "x2"
    # Stronger channel should pull its top doc higher
    out = weighted_rrf_fusion(
        [
            ([(d1, 0.0)], 2.0),
            ([(d2, 0.0)], 0.5),
        ],
        k=60,
    )
    assert out[0][0] == d1
