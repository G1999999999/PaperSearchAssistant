from __future__ import annotations

from typing import Any

from sqlalchemy import text

from tools.storage.sql.db import get_session_factory


def fts_search_chunks(
    query: str,
    *,
    paper_id: int | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """PostgreSQL 全文检索 paper_chunks.content_norm；无表或失败时返回空。"""
    if not (query or "").strip():
        return []
    lim = max(1, min(200, int(limit)))
    pid_clause = "AND paper_id = :pid" if paper_id is not None else ""
    sql = text(
        f"""
        SELECT id, paper_id, section_id, chunk_role, content, chroma_doc_id,
               ts_rank(to_tsvector('simple', content_norm),
                       plainto_tsquery('simple', :q)) AS rnk
        FROM paper_chunks
        WHERE to_tsvector('simple', content_norm) @@ plainto_tsquery('simple', :q)
        {pid_clause}
        ORDER BY rnk DESC
        LIMIT :lim
        """
    )
    params: dict[str, Any] = {"q": query.strip(), "lim": lim}
    if paper_id is not None:
        params["pid"] = int(paper_id)
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        try:
            result = session.execute(sql, params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]
        except Exception:
            return []
    finally:
        session.close()


def hydrate_chunk_contents(chunk_ids: list[int]) -> dict[int, str]:
    """按主键拉取 chunk 正文（用于向量命中后从 PG 补水）。"""
    if not chunk_ids:
        return {}
    factory = get_session_factory()
    if factory is None:
        return {}
    from sqlalchemy import select

    from tools.storage.sql.models import PaperChunk

    session = factory()
    try:
        try:
            stmt = select(PaperChunk).where(
                PaperChunk.id.in_([int(x) for x in chunk_ids[:200]])
            )
            rows = list(session.scalars(stmt).all())
            return {r.id: r.content or "" for r in rows}
        except Exception:
            return {}
    finally:
        session.close()
