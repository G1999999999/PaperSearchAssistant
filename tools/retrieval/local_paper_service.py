from __future__ import annotations

from typing import Any

from tools.storage.papers_db import list_papers as db_list_papers
from tools.retrieval.local_paper_matcher import match_local_paper


def search_local_papers(query_keywords: list[str], limit: int = 8) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    keys = [str(k or "").strip() for k in (query_keywords or []) if str(k or "").strip()]
    if not keys:
        keys = [""]
    for kw in keys[:8]:
        rows = db_list_papers(keyword=kw or None, limit=max(10, int(limit) * 2), offset=0)
        for r in rows or []:
            rid = str(r.get("arxiv_id") or r.get("title") or "").strip().lower()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            merged.append(
                {
                    "paper_id": r.get("arxiv_id"),
                    "arxiv_id": r.get("arxiv_id"),
                    "title": r.get("title"),
                    "authors": list(r.get("authors") or []),
                    "summary": r.get("summary"),
                    "published": r.get("published"),
                    "pdf_path": r.get("pdf_path"),
                    "indexed": True,
                    "source_type": "local",
                }
            )
            if len(merged) >= max(1, int(limit)):
                return merged
    return merged


def bind_local_paper_if_mentioned(question: str) -> dict[str, Any] | None:
    m = match_local_paper(question)
    if not m.matched or not m.paper:
        return None
    p = dict(m.paper)
    return {
        "paper_id": p.get("arxiv_id"),
        "arxiv_id": p.get("arxiv_id"),
        "title": p.get("title"),
        "pdf_path": p.get("pdf_path"),
        "indexed": True,
        "source_type": "local",
        "match_score": m.score,
        "match_reason": m.reason,
    }

