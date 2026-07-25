from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_placeholder_integration_gated():
    """占位：连接 Chroma/PG、调用 rerank 或 RAGAgent.answer 的实集成测试写在这里。"""
    if os.environ.get("RUN_LAYER_INTEGRATION", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("Set RUN_LAYER_INTEGRATION=1 to run L6+ integration tests")
