from __future__ import annotations

import os

import pytest

from eval.retrieval_recall_eval import load_retrieval_eval_cases, run_all_retrieval_cases


@pytest.mark.integration
def test_chroma_retrieval_recall_golden():
    if os.environ.get("RUN_LAYER_INTEGRATION", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("Set RUN_LAYER_INTEGRATION=1 to run Chroma recall tests")

    _, cases = load_retrieval_eval_cases()
    if not cases:
        pytest.skip(
            "No enabled cases in eval/fixtures/golden_retrieval_chunk_ids.json — "
            "copy from golden_retrieval_chunk_ids.example.json and set enabled=true"
        )

    report, _ = run_all_retrieval_cases()
    failures = [r for r in report if not r["ok"]]
    assert not failures, (
        "检索 recall 未达标，详情: "
        + str(failures)
        + "；可调大 k、检查 namespace/metadata 或改写 hits 短语。"
    )
