"""论文入库时同步 PostgreSQL：papers / paper_chunks / paper_figures / paper_tables（可选，需 DATABASE_URL）。"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, text

from config import DATABASE_URL, RAG_PG_SYNC_ON_INGEST
from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import (
    Paper,
    PaperChunk,
    PaperFigure,
    PaperSection,
    PaperSectionSummary,
    PaperSummaryView,
    PaperTable,
)


def pg_sync_enabled() -> bool:
    return bool((DATABASE_URL or "").strip()) and bool(RAG_PG_SYNC_ON_INGEST)


def ensure_ingest_performance_indexes() -> bool:
    """为入库/检索热点列补齐索引（幂等，不改变检索逻辑）。"""
    if not pg_sync_enabled():
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paper_chunks_paper_id "
                "ON paper_chunks (paper_id)"
            )
        )
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paper_chunks_paper_page_range "
                "ON paper_chunks (paper_id, page_from, page_to)"
            )
        )
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paper_chunks_fts_simple "
                "ON paper_chunks USING GIN (to_tsvector('simple', content_norm))"
            )
        )
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _parse_year(published: str | None) -> int | None:
    if not published:
        return None
    m = re.search(r"(19|20)\d{2}", str(published))
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


def _parse_published_at(published: str | None) -> datetime | None:
    if not published:
        return None
    raw = str(published).strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def upsert_paper_row_for_ingest(
    *,
    arxiv_id: str,
    title: str | None,
    abstract: str | None,
    authors: list[str],
    pdf_path: str,
    source_url: str | None,
    published: str | None,
    page_count: int | None = None,
) -> int | None:
    """插入或更新 ``papers``，返回主键 id；未配置 PG 或失败时返回 None。"""
    if not pg_sync_enabled():
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    aid = (arxiv_id or "").strip()
    if not aid:
        return None
    title_f = (title or "").strip() or f"arXiv:{aid}"
    pdf = (pdf_path or "").strip()
    if not pdf:
        return None
    try:
        digest = _sha256_file(pdf)
    except OSError:
        digest = ""
    if not digest:
        digest = hashlib.sha256(pdf.encode("utf-8", errors="ignore")).hexdigest()

    authors_list: list[str] = [str(a) for a in (authors or []) if str(a).strip()]
    year = _parse_year(published)
    pub_at = _parse_published_at(published)

    session = factory()
    try:
        row = session.scalar(select(Paper).where(Paper.arxiv_id == aid))
        if row is None:
            row = Paper(
                arxiv_id=aid,
                title=title_f,
                title_norm=_norm_title(title_f),
                abstract=(abstract or "").strip() or None,
                authors_json=authors_list,
                pdf_path=pdf,
                pdf_sha256=digest,
                source_url=(source_url or "").strip() or None,
                published_at=pub_at,
                year=year,
                page_count=page_count,
                ingest_status="ready",
                parse_status="basic",
            )
            session.add(row)
        else:
            row.title = title_f
            row.title_norm = _norm_title(title_f)
            row.abstract = (abstract or "").strip() or None
            row.authors_json = authors_list
            row.pdf_path = pdf
            row.pdf_sha256 = digest
            row.source_url = (source_url or "").strip() or row.source_url
            row.published_at = pub_at or row.published_at
            row.year = year if year is not None else row.year
            if page_count is not None:
                row.page_count = page_count
            row.ingest_status = "ready"
        session.commit()
        session.refresh(row)
        return int(row.id)
    except Exception:
        session.rollback()
        return None
    finally:
        session.close()


def _token_count_estimate(text: str) -> int:
    return max(1, len((text or "").split()))


def replace_paper_chunks(
    paper_id: int,
    items: list[dict[str, Any]],
) -> bool:
    """删除该论文在 PG 中的旧 chunks，批量插入新块，并串 prev/next。``items`` 可为空（仅清空）。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        session.execute(delete(PaperChunk).where(PaperChunk.paper_id == int(paper_id)))
        if not items:
            session.commit()
            return True
        seen_ids: set[str] = set()
        ordered_rows: list[dict[str, Any]] = []
        for it in sorted(items, key=lambda x: int(x.get("chunk_index", 0))):
            cid = str(it.get("chroma_doc_id") or "").strip()
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            body = str(it.get("content") or "")
            if not body:
                continue
            norm = body.casefold()
            role = str(it.get("chunk_role") or "generic").strip()[:32] or "generic"
            ordered_rows.append(
                {
                    "paper_id": int(paper_id),
                    "section_id": (
                        int(it.get("section_id"))
                        if str(it.get("section_id") or "").strip().isdigit()
                        else None
                    ),
                    "chunk_index": int(it.get("chunk_index", 0)),
                    "chunk_role": role,
                    "content": body,
                    "content_norm": norm,
                    "summary_text": (str(it.get("summary_text")).strip() or None)
                    if it.get("summary_text")
                    else None,
                    "token_count": int(it.get("token_count") or _token_count_estimate(body)),
                    "page_from": it.get("page_from"),
                    "page_to": it.get("page_to"),
                    "has_table": bool(it.get("has_table")),
                    "has_figure": bool(it.get("has_figure")),
                    "chroma_doc_id": cid[:128],
                }
            )
        if not ordered_rows:
            session.commit()
            return True
        insert_stmt = (
            PaperChunk.__table__
            .insert()
            .returning(PaperChunk.id, PaperChunk.chunk_index)
        )
        inserted = list(session.execute(insert_stmt, ordered_rows).all())
        if inserted:
            idx_to_id = {int(r.chunk_index): int(r.id) for r in inserted}
            update_rows: list[dict[str, Any]] = []
            for row in ordered_rows:
                cur_idx = int(row["chunk_index"])
                cur_id = idx_to_id.get(cur_idx)
                if not cur_id:
                    continue
                update_rows.append(
                    {
                        "id": cur_id,
                        "prev_chunk_id": idx_to_id.get(cur_idx - 1),
                        "next_chunk_id": idx_to_id.get(cur_idx + 1),
                    }
                )
            if update_rows:
                session.bulk_update_mappings(PaperChunk, update_rows)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def replace_paper_figures(
    paper_id: int,
    items: list[dict[str, Any]],
) -> bool:
    """替换 ``paper_figures``（插图元数据，与向量中的 figure chunk 对应）。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        session.execute(delete(PaperFigure).where(PaperFigure.paper_id == int(paper_id)))
        rows: list[dict[str, Any]] = []
        for it in items:
            p = str(it.get("image_path") or "").strip()
            if not p:
                continue
            rows.append(
                {
                    "paper_id": int(paper_id),
                    "section_id": None,
                    "page_no": int(it.get("page_no") or 0),
                    "figure_number": str(it.get("figure_number") or "")[:64] or None,
                    "caption_text": (str(it.get("caption_text") or "").strip() or None),
                    "ocr_text": None,
                    "vision_summary": (str(it.get("vision_summary") or "").strip() or None),
                    "image_path": p[:2000],
                }
            )
        if rows:
            session.bulk_insert_mappings(PaperFigure, rows)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def replace_paper_tables(
    paper_id: int,
    items: list[dict[str, Any]],
) -> bool:
    """替换 ``paper_tables``（表格结构化内容与 caption，与向量中 table chunk 对应）。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        session.execute(delete(PaperTable).where(PaperTable.paper_id == int(paper_id)))
        rows: list[dict[str, Any]] = []
        for it in items:
            rows.append(
                {
                    "paper_id": int(paper_id),
                    "section_id": None,
                    "page_no": int(it.get("page_no") or 0),
                    "table_number": str(it.get("table_number") or "").strip()[:64] or None,
                    "title": (str(it.get("title") or "").strip() or None),
                    "caption_text": (str(it.get("caption_text") or "").strip() or None),
                    "summary_text": (str(it.get("summary_text") or "").strip() or None),
                    "markdown_text": (str(it.get("markdown_text") or "").strip() or None),
                }
            )
        if rows:
            session.bulk_insert_mappings(PaperTable, rows)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def clear_paper_chunks_only(paper_id: int) -> None:
    """仅清空 chunks（例如在重写向量前）。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return
    factory = get_session_factory()
    if factory is None:
        return
    session = factory()
    try:
        session.execute(delete(PaperChunk).where(PaperChunk.paper_id == int(paper_id)))
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def replace_paper_sections(
    paper_id: int,
    items: list[dict[str, Any]],
) -> dict[int, int]:
    """替换 paper_sections，返回 order_index -> section_id 映射。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return {}
    factory = get_session_factory()
    if factory is None:
        return {}
    session = factory()
    out: dict[int, int] = {}
    try:
        session.execute(delete(PaperSection).where(PaperSection.paper_id == int(paper_id)))
        if not items:
            session.commit()
            return {}
        rows: list[PaperSection] = []
        for it in sorted(items, key=lambda x: int(x.get("order_index", 0))):
            title = str(it.get("title") or "").strip()
            if not title:
                continue
            row = PaperSection(
                paper_id=int(paper_id),
                parent_section_id=None,
                section_level=max(1, int(it.get("section_level") or 1)),
                section_number=(str(it.get("section_number") or "").strip() or None),
                title=title,
                title_norm=_norm_title(title),
                page_start=it.get("page_start"),
                page_end=it.get("page_end"),
                order_index=max(0, int(it.get("order_index") or 0)),
            )
            rows.append(row)
            session.add(row)
        session.flush()
        for r in rows:
            out[int(r.order_index)] = int(r.id)
        session.commit()
        return out
    except Exception:
        session.rollback()
        return {}
    finally:
        session.close()


def bind_chunk_sections_by_index(
    paper_id: int,
    chunk_index_to_section_id: dict[int, int],
) -> bool:
    """按 chunk_index 回填 chunk.section_id。"""
    if not pg_sync_enabled() or paper_id <= 0 or not chunk_index_to_section_id:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        stmt = select(PaperChunk).where(PaperChunk.paper_id == int(paper_id))
        rows = list(session.scalars(stmt).all())
        for r in rows:
            sid = chunk_index_to_section_id.get(int(r.chunk_index))
            if sid:
                r.section_id = int(sid)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def replace_section_summaries(
    paper_id: int,
    items: list[dict[str, Any]],
) -> bool:
    """替换 paper_section_summaries。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        session.execute(delete(PaperSectionSummary).where(PaperSectionSummary.paper_id == int(paper_id)))
        rows: list[dict[str, Any]] = []
        for it in items:
            sid = int(it.get("section_id") or 0)
            if sid <= 0:
                continue
            rows.append(
                {
                    "section_id": sid,
                    "paper_id": int(paper_id),
                    "section_role": (str(it.get("section_role") or "").strip() or None),
                    "summary_text": (str(it.get("summary_text") or "").strip() or None),
                    "keywords_json": list(it.get("keywords_json") or []),
                }
            )
        if rows:
            session.bulk_insert_mappings(PaperSectionSummary, rows)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()


def upsert_paper_summary_view(
    paper_id: int,
    bundle: dict[str, Any],
) -> bool:
    """写入 paper_summary_views（论文级摘要包）。"""
    if not pg_sync_enabled() or paper_id <= 0:
        return False
    factory = get_session_factory()
    if factory is None:
        return False
    session = factory()
    try:
        row = session.scalar(select(PaperSummaryView).where(PaperSummaryView.paper_id == int(paper_id)))
        if row is None:
            row = PaperSummaryView(paper_id=int(paper_id))
            session.add(row)
        row.abstract_summary = (str(bundle.get("abstract_summary") or "").strip() or None)
        row.intro_summary = (str(bundle.get("intro_summary") or "").strip() or None)
        row.method_summary = (str(bundle.get("method_summary") or "").strip() or None)
        row.result_summary = (str(bundle.get("result_summary") or "").strip() or None)
        row.conclusion_summary = (str(bundle.get("conclusion_summary") or "").strip() or None)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()
