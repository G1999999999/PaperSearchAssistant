from __future__ import annotations

import pytest

from eval.routing_golden import (
    assert_case_langgraph_route,
    assert_case_query_route,
    load_golden_cases,
)


@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda c: c.id)
def test_golden_query_route(case):
    assert_case_query_route(case)


@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda c: c.id)
def test_golden_langgraph_selected_path(case):
    assert_case_langgraph_route(case)


def test_extract_arxiv_id_layer():
    from tools.agent.router import extract_arxiv_id

    assert extract_arxiv_id("read 1706.03762 for details") == "1706.03762"
    assert extract_arxiv_id("old id cs.CL/0001015") is not None
