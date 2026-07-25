from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import PaperSectionSummary, PaperSummaryView


def get_paper_summary_bundle(paper_id: int) -> dict[str, Any] | None:
    factory = get_session_factory()
    if factory is None:
        return None
    session = factory()
    try:
        row = session.scalar(select(PaperSummaryView).where(PaperSummaryView.paper_id == int(paper_id)))
        if row is None:
            return None
        return {
            "paper_id": int(row.paper_id),
            "abstract_summary": row.abstract_summary,
            "intro_summary": row.intro_summary,
            "method_summary": row.method_summary,
            "result_summary": row.result_summary,
            "conclusion_summary": row.conclusion_summary,
        }
    finally:
        session.close()


def list_section_summaries(paper_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperSectionSummary)
            .where(PaperSectionSummary.paper_id == int(paper_id))
            .order_by(PaperSectionSummary.section_id.asc())
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "section_id": int(r.section_id),
            "paper_id": int(r.paper_id),
            "section_role": r.section_role,
            "summary_text": r.summary_text,
            "keywords_json": list(r.keywords_json or []),
        }
        for r in rows
    ]

