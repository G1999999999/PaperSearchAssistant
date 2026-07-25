from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import Paper, PaperFigure, PaperSection, PaperTable


def get_paper_by_id(paper_id: int) -> dict[str, Any] | None:
    factory = get_session_factory()
    if factory is None:
        return None
    session = factory()
    try:
        p = session.scalar(select(Paper).where(Paper.id == int(paper_id)))
        if p is None:
            return None
        return {
            "id": p.id,
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "year": p.year,
            "pdf_path": p.pdf_path,
        }
    finally:
        session.close()


def list_sections(paper_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperSection)
            .where(PaperSection.paper_id == int(paper_id))
            .order_by(PaperSection.order_index.asc())
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": r.id,
            "title": r.title,
            "section_number": r.section_number,
            "level": r.section_level,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "order_index": r.order_index,
            "parent_section_id": r.parent_section_id,
        }
        for r in rows
    ]


def list_tables(paper_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = select(PaperTable).where(PaperTable.paper_id == int(paper_id)).order_by(PaperTable.id.asc())
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": r.id,
            "page_no": r.page_no,
            "table_number": r.table_number,
            "title": r.title,
            "caption_text": r.caption_text,
            "summary_text": r.summary_text,
            "markdown_text": r.markdown_text,
        }
        for r in rows
    ]


def list_figures(paper_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperFigure).where(PaperFigure.paper_id == int(paper_id)).order_by(PaperFigure.id.asc())
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": r.id,
            "page_no": r.page_no,
            "figure_number": r.figure_number,
            "caption_text": r.caption_text,
            "ocr_text": r.ocr_text,
            "vision_summary": r.vision_summary,
            "image_path": r.image_path,
        }
        for r in rows
    ]
