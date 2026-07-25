from __future__ import annotations

from typing import Any

from sqlalchemy import and_, or_, select

from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import PaperChunk
from tools.storage.repos.section_repo import get_paper_by_arxiv_id


def paper_id_from_rag_namespace(namespace: str) -> int | None:
    """从 ``paper:<arxiv_id>:...`` 解析论文主键；非论文 namespace 返回 None。"""
    ns = (namespace or "").strip()
    if not ns.startswith("paper:"):
        return None
    parts = ns.split(":")
    arxiv_id = parts[1].strip() if len(parts) > 1 else ""
    if not arxiv_id:
        return None
    row = get_paper_by_arxiv_id(arxiv_id)
    if not row:
        return None
    try:
        return int(row.get("id"))
    except Exception:
        return None


def list_chunks_for_paper_on_pages(
    paper_id: int,
    pages: list[int],
    *,
    exclude_roles: frozenset[str] | None = None,
    limit_total: int = 24,
) -> list[dict[str, Any]]:
    """取与给定页码有重叠的正文类 chunk（用于插图同页文字上下文）。"""
    pset = sorted({int(p) for p in (pages or []) if int(p) > 0})
    if not pset or paper_id <= 0:
        return []
    factory = get_session_factory()
    if factory is None:
        return []
    excl = exclude_roles or frozenset({"figure", "table"})
    overlap_conds = []
    for p in pset:
        overlap_conds.append(
            and_(
                PaperChunk.page_from.isnot(None),
                PaperChunk.page_from <= p,
                or_(PaperChunk.page_to.is_(None), PaperChunk.page_to >= p),
            )
        )
    if not overlap_conds:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperChunk)
            .where(
                PaperChunk.paper_id == int(paper_id),
                or_(*overlap_conds),
                PaperChunk.chunk_role.notin_(list(excl)),
            )
            .order_by(PaperChunk.chunk_index.asc())
            .limit(max(1, min(200, int(limit_total))))
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": int(r.id),
            "paper_id": int(r.paper_id),
            "section_id": int(r.section_id) if r.section_id is not None else None,
            "chunk_index": int(r.chunk_index),
            "chunk_role": r.chunk_role,
            "content": r.content,
            "chroma_doc_id": r.chroma_doc_id,
            "page_from": r.page_from,
            "page_to": r.page_to,
        }
        for r in rows
    ]


def list_chunks_by_section_ids(
    section_ids: list[int],
    *,
    limit: int = 120,
) -> list[dict[str, Any]]:
    if not section_ids:
        return []
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperChunk)
            .where(PaperChunk.section_id.in_([int(x) for x in section_ids[:200]]))
            .order_by(PaperChunk.chunk_index.asc())
            .limit(max(1, min(500, int(limit))))
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": int(r.id),
            "paper_id": int(r.paper_id),
            "section_id": int(r.section_id) if r.section_id is not None else None,
            "chunk_index": int(r.chunk_index),
            "chunk_role": r.chunk_role,
            "content": r.content,
            "chroma_doc_id": r.chroma_doc_id,
            "prev_chunk_id": r.prev_chunk_id,
            "next_chunk_id": r.next_chunk_id,
        }
        for r in rows
    ]


def list_table_role_chunks_for_paper(
    paper_id: int,
    *,
    limit: int = 48,
) -> list[dict[str, Any]]:
    """取该论文在 PG 中标记为 ``chunk_role=table`` 的块（入库管线写入；与 ``paper_tables`` 可并存）。"""
    if paper_id <= 0:
        return []
    factory = get_session_factory()
    if factory is None:
        return []
    lim = max(1, min(120, int(limit)))
    session = factory()
    try:
        stmt = (
            select(PaperChunk)
            .where(
                PaperChunk.paper_id == int(paper_id),
                PaperChunk.chunk_role == "table",
            )
            .order_by(PaperChunk.chunk_index.asc())
            .limit(lim)
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": int(r.id),
            "paper_id": int(r.paper_id),
            "chunk_index": int(r.chunk_index),
            "chunk_role": r.chunk_role,
            "content": r.content,
            "page_from": int(r.page_from) if r.page_from is not None else None,
            "page_to": int(r.page_to) if r.page_to is not None else None,
        }
        for r in rows
    ]


def get_chunk_neighbors(chunk_id: int) -> dict[str, Any] | None:
    factory = get_session_factory()
    if factory is None:
        return None
    session = factory()
    try:
        c = session.scalar(select(PaperChunk).where(PaperChunk.id == int(chunk_id)))
        if c is None:
            return None
        return {
            "chunk_id": int(c.id),
            "prev_chunk_id": c.prev_chunk_id,
            "next_chunk_id": c.next_chunk_id,
            "section_id": c.section_id,
            "paper_id": c.paper_id,
        }
    finally:
        session.close()

