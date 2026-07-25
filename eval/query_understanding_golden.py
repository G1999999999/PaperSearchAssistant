from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.retrieval.query_understanding import analyze_paper_query, subquestions_for_decomposition


@dataclass
class QUCase:
    id: str
    question: str
    expect: dict[str, Any] | None
    expect_subquestions_min: int | None


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "golden_query_understanding.json"


def load_qu_cases(path: Path | None = None) -> list[QUCase]:
    p = path or _fixture_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: list[QUCase] = []
    for item in raw.get("cases", []):
        out.append(
            QUCase(
                id=str(item["id"]),
                question=str(item["question"]),
                expect=item.get("expect"),
                expect_subquestions_min=item.get("expect_subquestions_min"),
            )
        )
    return out


def assert_qu_case(case: QUCase) -> None:
    if case.expect:
        u = analyze_paper_query(case.question)
        exp = case.expect
        if "intent" in exp:
            assert u.intent == exp["intent"], f"[{case.id}] intent {u.intent!r}"
        if "wants_table" in exp:
            assert u.wants_table == exp["wants_table"], f"[{case.id}] wants_table"
        if "wants_figure" in exp:
            assert u.wants_figure == exp["wants_figure"], f"[{case.id}] wants_figure"
    if case.expect_subquestions_min is not None:
        sq = subquestions_for_decomposition(case.question)
        assert len(sq) >= int(case.expect_subquestions_min), (
            f"[{case.id}] subquestions {sq!r} len {len(sq)}"
        )
