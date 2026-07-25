"""PaperSearchAssistant2 离线评测：路由 / LangGraph 路径 / 可选回答关键词。

无需调用 LLM 即可回归核心路由逻辑；扩写黄金用例见 ``eval/fixtures/``。
"""

from eval.routing_golden import (
    EvalCase,
    assert_case_query_route,
    assert_case_langgraph_route,
    load_golden_cases,
    run_all_cases,
)

__all__ = [
    "EvalCase",
    "assert_case_query_route",
    "assert_case_langgraph_route",
    "load_golden_cases",
    "run_all_cases",
]
