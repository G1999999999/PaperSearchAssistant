from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.retrieval.query_router import build_query_route
from tools.agent.langgraph_orchestrator import _route_node


@dataclass
class EvalCase:
    id: str
    question: str
    use_tools: bool = False
    user_image_paths: list[str] | None = None
    session_id: str | None = None
    namespace: str = "default"
    expect_route: dict[str, Any] | None = None
    expect_selected_path: str | None = None


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "golden_routing.json"


def load_golden_cases(path: Path | None = None) -> list[EvalCase]:
    p = path or _fixture_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw.get("cases", []):
        cases.append(
            EvalCase(
                id=str(item["id"]),
                question=str(item["question"]),
                use_tools=bool(item.get("use_tools", False)),
                user_image_paths=list(item["user_image_paths"])
                if item.get("user_image_paths")
                else None,
                session_id=item.get("session_id"),
                namespace=str(item.get("namespace", "default")),
                expect_route=item.get("expect_route"),
                expect_selected_path=item.get("expect_selected_path"),
            )
        )
    return cases


def assert_case_query_route(case: EvalCase) -> None:
    if not case.expect_route:
        return
    d = build_query_route(case.question)
    exp = case.expect_route
    if "intent" in exp:
        assert d.intent == exp["intent"], f"[{case.id}] intent: {d.intent!r} != {exp['intent']!r}"
    if "sub_intent" in exp:
        assert d.sub_intent == exp["sub_intent"], (
            f"[{case.id}] sub_intent: {d.sub_intent!r} != {exp['sub_intent']!r}"
        )
    if "route" in exp:
        got_route = getattr(d.route, "value", d.route)
        assert str(got_route) == str(exp["route"]), (
            f"[{case.id}] route: {got_route!r} != {exp['route']!r}"
        )
    if "source_preference" in exp:
        assert d.source_preference == exp["source_preference"], (
            f"[{case.id}] source_preference mismatch"
        )
    if "needs_web" in exp:
        assert bool(d.needs_web) == bool(exp["needs_web"]), f"[{case.id}] needs_web mismatch"


def _route_node_state(case: EvalCase) -> dict[str, Any]:
    return {
        "question": case.question,
        "use_tools": case.use_tools,
        "user_image_paths": case.user_image_paths or [],
        "session_id": case.session_id,
        "namespace": case.namespace,
        "_agent": None,
    }


def assert_case_langgraph_route(case: EvalCase) -> None:
    if case.expect_selected_path is None:
        return
    out = _route_node(_route_node_state(case))
    got = str(out.get("selected_path") or "")
    assert got == case.expect_selected_path, (
        f"[{case.id}] selected_path: {got!r} != {case.expect_selected_path!r} "
        f"(paper_intent={out.get('paper_intent')}, route={out.get('route')})"
    )


def run_all_cases(cases: list[EvalCase] | None = None, fixture: Path | None = None) -> list[str]:
    """运行全部断言；返回每条用例的简短通过信息。"""
    resolved = cases if cases is not None else load_golden_cases(fixture)
    lines: list[str] = []
    for c in resolved:
        assert_case_query_route(c)
        assert_case_langgraph_route(c)
        lines.append(f"ok {c.id}")
    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="运行 golden 路由评测（无 LLM）")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="黄金 JSON 路径，默认 eval/fixtures/golden_routing.json",
    )
    args = parser.parse_args()
    for line in run_all_cases(fixture=args.fixture):
        print(line)
    print(f"pass {len(load_golden_cases(args.fixture))} cases")


if __name__ == "__main__":
    main()
