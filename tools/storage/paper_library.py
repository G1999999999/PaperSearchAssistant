from __future__ import annotations

"""
Backward-compatible paper library API.

Historically this project stored local paper metadata in `data/papers/index.json`.
Now we prefer SQLite (`data/papers/papers.db`) via `tools/papers_db.py`.

This module keeps the old import paths stable (`LocalPaper`, `upsert_paper`, etc.)
and performs best-effort migration from index.json into SQLite when needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.storage.papers_db import (
    PaperRow,
    get_paper as db_get_paper,
    list_papers as db_list_papers,
    reconcile_from_disk as db_reconcile_from_disk,
    upsert_paper as db_upsert_paper,
)


PAPERS_DIR = Path("data/papers")
PAPERS_INDEX_PATH = PAPERS_DIR / "index.json"


@dataclass
class LocalPaper:
    arxiv_id: str
    pdf_path: str
    title: str | None = None
    authors: list[str] | None = None
    published: str | None = None
    url: str | None = None
    added_at: str | None = None
    namespaces: list[str] | None = None
    summary: str | None = None

    def get(self, key: str, default=None):
        """兼容历史代码把 LocalPaper 当 dict 使用的 `.get(...)` 访问。"""
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "pdf_path": self.pdf_path,
            "title": self.title,
            "authors": list(self.authors or []),
            "published": self.published,
            "url": self.url,
            "added_at": self.added_at,
            "namespaces": list(self.namespaces or []),
            "summary": self.summary,
        }


def _migrate_index_json_to_sqlite() -> int:
    if not PAPERS_INDEX_PATH.exists():
        return 0
    try:
        idx = json.loads(PAPERS_INDEX_PATH.read_text(encoding="utf-8"))
        papers = (idx.get("papers") or {}) if isinstance(idx, dict) else {}
    except Exception:
        return 0
    n = 0
    for arxiv_id, rec in papers.items():
        if not isinstance(rec, dict):
            continue
        try:
            db_upsert_paper(
                PaperRow(
                    arxiv_id=str(rec.get("arxiv_id") or arxiv_id),
                    title=rec.get("title"),
                    authors=list(rec.get("authors") or []) if rec.get("authors") is not None else None,
                    summary=rec.get("summary"),
                    published=rec.get("published"),
                    url=rec.get("url"),
                    pdf_path=rec.get("pdf_path"),
                    namespaces=list(rec.get("namespaces") or []) if rec.get("namespaces") is not None else None,
                    added_at=rec.get("added_at"),
                )
            )
            n += 1
        except Exception:
            continue
    return n


def upsert_paper(paper: LocalPaper) -> LocalPaper:
    _migrate_index_json_to_sqlite()
    rec = db_upsert_paper(
        PaperRow(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            summary=paper.summary,
            published=paper.published,
            url=paper.url,
            pdf_path=paper.pdf_path,
            namespaces=paper.namespaces,
            added_at=paper.added_at,
        )
    )
    return LocalPaper(
        arxiv_id=rec.get("arxiv_id", paper.arxiv_id),
        pdf_path=rec.get("pdf_path") or paper.pdf_path,
        title=rec.get("title"),
        authors=rec.get("authors") or [],
        summary=rec.get("summary"),
        published=rec.get("published"),
        url=rec.get("url"),
        added_at=rec.get("added_at"),
        namespaces=rec.get("namespaces") or [],
    )


def get_paper(arxiv_id: str) -> Optional[dict]:
    _migrate_index_json_to_sqlite()
    return db_get_paper(arxiv_id)


def list_papers() -> list[dict]:
    _migrate_index_json_to_sqlite()
    return db_list_papers(limit=200, offset=0)


def reconcile_index_with_disk() -> dict:
    _migrate_index_json_to_sqlite()
    db_reconcile_from_disk(str(PAPERS_DIR))
    return {"migrated": True}

