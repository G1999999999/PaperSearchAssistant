from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import sqlite3  # type: ignore
    _SQLITE_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on environment
    # Some Python distributions (e.g. mismatched conda/system) can fail to import
    # the built-in sqlite3 extension at runtime (missing symbols).
    # In that case we fall back to JSON-based storage so the CLI can still run.
    sqlite3 = None  # type: ignore
    _SQLITE_AVAILABLE = False
    _SQLITE_IMPORT_ERROR = e


DB_PATH = Path("data/papers/papers.db")
PAPERS_INDEX_PATH = Path("data/papers/index.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
              arxiv_id TEXT PRIMARY KEY,
              title TEXT,
              authors_json TEXT NOT NULL DEFAULT '[]',
              summary TEXT,
              published TEXT,
              url TEXT,
              pdf_path TEXT,
              namespaces_json TEXT NOT NULL DEFAULT '[]',
              added_at TEXT,
              updated_at TEXT
            )
            """
        )
        # FTS5 是可选的；如果环境不支持，这条建表语句会失败并被忽略。
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                  arxiv_id UNINDEXED,
                  title,
                  authors,
                  summary
                )
                """
            )
        except sqlite3.OperationalError:
            pass


def _norm_arxiv_id(arxiv_id: str) -> str:
    return (arxiv_id or "").strip().replace("arXiv:", "").replace("ARXIV:", "")


@dataclass
class PaperRow:
    arxiv_id: str
    title: str | None = None
    authors: list[str] | None = None
    summary: str | None = None
    published: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    namespaces: list[str] | None = None
    added_at: str | None = None
    updated_at: str | None = None


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "arxiv_id": r["arxiv_id"],
        "title": r["title"],
        "authors": json.loads(r["authors_json"] or "[]"),
        "summary": r["summary"],
        "published": r["published"],
        "url": r["url"],
        "pdf_path": r["pdf_path"],
        "namespaces": json.loads(r["namespaces_json"] or "[]"),
        "added_at": r["added_at"],
        "updated_at": r["updated_at"],
    }


def upsert_paper(p: PaperRow) -> dict:
    init_db()
    arxiv_id = _norm_arxiv_id(p.arxiv_id)
    if not arxiv_id:
        raise ValueError("arxiv_id is required")
    now = _now_iso()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?",
            (arxiv_id,),
        ).fetchone()
        existing_authors = json.loads(existing["authors_json"] or "[]") if existing else []
        existing_namespaces = json.loads(existing["namespaces_json"] or "[]") if existing else []

        authors = existing_authors
        if p.authors is not None:
            authors = p.authors
        namespaces = set(existing_namespaces)
        if p.namespaces:
            namespaces.update(p.namespaces)

        added_at = (existing["added_at"] if existing else None) or p.added_at or now
        updated_at = now

        conn.execute(
            """
            INSERT INTO papers (
              arxiv_id, title, authors_json, summary, published, url, pdf_path, namespaces_json, added_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
              title = COALESCE(excluded.title, papers.title),
              authors_json = excluded.authors_json,
              summary = COALESCE(excluded.summary, papers.summary),
              published = COALESCE(excluded.published, papers.published),
              url = COALESCE(excluded.url, papers.url),
              pdf_path = COALESCE(excluded.pdf_path, papers.pdf_path),
              namespaces_json = excluded.namespaces_json,
              added_at = papers.added_at,
              updated_at = excluded.updated_at
            """,
            (
                arxiv_id,
                p.title,
                json.dumps(authors, ensure_ascii=False),
                p.summary,
                p.published,
                p.url,
                p.pdf_path,
                json.dumps(sorted(namespaces), ensure_ascii=False),
                added_at,
                updated_at,
            ),
        )
        try:
            conn.execute(
                "INSERT INTO papers_fts(arxiv_id, title, authors, summary) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(arxiv_id) DO UPDATE SET title=excluded.title, authors=excluded.authors, summary=excluded.summary",
                (
                    arxiv_id,
                    p.title or (existing["title"] if existing else None) or "",
                    ", ".join(authors),
                    p.summary or (existing["summary"] if existing else None) or "",
                ),
            )
        except sqlite3.OperationalError:
            pass
        row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return _row_to_dict(row) if row else {"arxiv_id": arxiv_id}


def get_paper(arxiv_id: str) -> Optional[dict]:
    init_db()
    arxiv_id = _norm_arxiv_id(arxiv_id)
    if not arxiv_id:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return _row_to_dict(row) if row else None


def _sanitize_fts5_match_query(keyword: str) -> str:
    """避免 FTS5 MATCH 将用户串里的 `foo:bar` 解析成「列 foo」。

    典型误伤：`Gaussian Grouping（arXiv: 2312.xxxx）的方法部分` 会报
    ``no such column: 的方法部分（arXiv``。
    """
    s = (keyword or "").strip()
    if not s:
        return ""
    s = re.sub(
        r"\(?\s*arxiv\s*[:：]\s*[\d]{4}\.[\d]{4,5}v?\d*\s*\)?",
        " ",
        s,
        flags=re.I,
    )
    # 列限定语法 column : token；统一去掉冒号，只做「全文」检索
    s = re.sub(r"[：:]+", " ", s)
    s = re.sub(r"[\*\"]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def list_papers(
    *,
    title: str | None = None,
    author: str | None = None,
    keyword: str | None = None,
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    init_db()
    where = []
    params: list[Any] = []

    def _like(field: str, value: str):
        where.append(f"LOWER({field}) LIKE ?")
        params.append(f"%{value.lower()}%")

    if title:
        _like("title", title)

    if author:
        # authors 以 json 形式存储：这里用 json 文本上的 LIKE 做一个简单最小可行实现
        _like("authors_json", author)

    # 年份过滤作用在 published 字符串上（ISO 格式）
    if year is not None:
        where.append("substr(published, 1, 4) = ?")
        params.append(f"{int(year):04d}")
    if year_from is not None:
        where.append("substr(published, 1, 4) >= ?")
        params.append(f"{int(year_from):04d}")
    if year_to is not None:
        where.append("substr(published, 1, 4) <= ?")
        params.append(f"{int(year_to):04d}")

    base_sql = "SELECT * FROM papers"

    if keyword:
        kw_raw = (keyword or "").strip()
        kw_fts = _sanitize_fts5_match_query(kw_raw)
        use_fts = False
        with _connect() as conn:
            try:
                conn.execute("SELECT 1 FROM papers_fts LIMIT 1")
                use_fts = True
            except sqlite3.OperationalError:
                use_fts = False
        if use_fts and kw_fts:
            base_sql = (
                "SELECT p.* FROM papers p JOIN papers_fts f ON p.arxiv_id = f.arxiv_id"
            )
            where.append("f MATCH ?")
            params.append(kw_fts)
        else:
            where.append(
                "(LOWER(COALESCE(title,'')) LIKE ? OR LOWER(COALESCE(summary,'')) LIKE ?)"
            )
            kw = f"%{kw_raw.lower()}%"
            params.extend([kw, kw])

    if where:
        base_sql += " WHERE " + " AND ".join(where)

    base_sql += " ORDER BY COALESCE(added_at, updated_at, '') DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])

    with _connect() as conn:
        try:
            rows = conn.execute(base_sql, tuple(params)).fetchall()
            return [_row_to_dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS MATCH 语法极个别仍失败时：同一组 title/author/year 条件 + keyword 走 LIKE
            if not keyword:
                return []
            w_fb: list[str] = []
            p_fb: list[Any] = []
            if title:
                w_fb.append("LOWER(title) LIKE ?")
                p_fb.append(f"%{str(title).lower()}%")
            if author:
                w_fb.append("LOWER(authors_json) LIKE ?")
                p_fb.append(f"%{str(author).lower()}%")
            if year is not None:
                w_fb.append("substr(published, 1, 4) = ?")
                p_fb.append(f"{int(year):04d}")
            if year_from is not None:
                w_fb.append("substr(published, 1, 4) >= ?")
                p_fb.append(f"{int(year_from):04d}")
            if year_to is not None:
                w_fb.append("substr(published, 1, 4) <= ?")
                p_fb.append(f"{int(year_to):04d}")
            kw_raw = (keyword or "").strip()
            kw_like = f"%{kw_raw.lower()}%"
            w_fb.append(
                "(LOWER(COALESCE(title,'')) LIKE ? OR LOWER(COALESCE(summary,'')) LIKE ?)"
            )
            p_fb.extend([kw_like, kw_like])
            fb_sql = "SELECT * FROM papers WHERE " + " AND ".join(w_fb)
            fb_sql += (
                " ORDER BY COALESCE(added_at, updated_at, '') DESC LIMIT ? OFFSET ?"
            )
            p_fb.extend([int(limit), int(offset)])
            rows = conn.execute(fb_sql, tuple(p_fb)).fetchall()
            return [_row_to_dict(r) for r in rows]


def reconcile_from_disk(papers_dir: str = "data/papers") -> int:
    """确保 `papers_dir` 下的 *.pdf 文件都在 DB 中存在；使用文件名去后缀作为 arXiv ID。"""
    init_db()
    dir_path = Path(papers_dir)
    if not dir_path.exists():
        return 0
    count = 0
    for p in dir_path.glob("*.pdf"):
        arxiv_id = _norm_arxiv_id(p.stem)
        if not arxiv_id:
            continue
        if get_paper(arxiv_id) is None:
            upsert_paper(
                PaperRow(
                    arxiv_id=arxiv_id,
                    pdf_path=str(p.as_posix()),
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                )
            )
            count += 1
    return count


# ----------------------------
# JSON fallback (no sqlite)
# ----------------------------


def _json_load_index() -> dict[str, Any]:
    if not PAPERS_INDEX_PATH.exists():
        return {"papers": {}}
    try:
        idx = json.loads(PAPERS_INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(idx, dict) and isinstance(idx.get("papers"), dict):
            return idx
    except Exception:
        pass
    return {"papers": {}}


def _json_save_index(index: dict[str, Any]) -> None:
    PAPERS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPERS_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def init_db_json() -> None:
    # Best-effort: ensure index.json exists with expected shape.
    _json_save_index(_json_load_index())


def _paper_to_json_record(p: PaperRow) -> dict[str, Any]:
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "authors": p.authors if p.authors is not None else [],
        "summary": p.summary,
        "published": p.published,
        "url": p.url,
        "pdf_path": p.pdf_path,
        "namespaces": p.namespaces if p.namespaces is not None else [],
        "added_at": p.added_at,
        "updated_at": p.updated_at,
    }


def upsert_paper_json(p: PaperRow) -> dict:
    init_db_json()
    arxiv_id = _norm_arxiv_id(p.arxiv_id)
    if not arxiv_id:
        raise ValueError("arxiv_id is required")

    index = _json_load_index()
    papers: dict[str, Any] = index.setdefault("papers", {})

    existing = papers.get(arxiv_id) or {}
    existing_authors = list(existing.get("authors") or [])
    existing_namespaces = set(existing.get("namespaces") or [])

    authors = existing_authors if p.authors is None else p.authors
    namespaces = existing_namespaces
    if p.namespaces is not None:
        namespaces.update(p.namespaces)

    now = _now_iso()
    added_at = existing.get("added_at") or p.added_at or now
    updated_at = now

    record = _paper_to_json_record(
        PaperRow(
            arxiv_id=arxiv_id,
            title=p.title if p.title is not None else existing.get("title"),
            authors=authors,
            summary=p.summary if p.summary is not None else existing.get("summary"),
            published=p.published if p.published is not None else existing.get("published"),
            url=p.url if p.url is not None else existing.get("url"),
            pdf_path=p.pdf_path if p.pdf_path is not None else existing.get("pdf_path"),
            namespaces=sorted(namespaces),
            added_at=added_at,
            updated_at=updated_at,
        )
    )

    papers[arxiv_id] = record
    _json_save_index(index)
    return record


def get_paper_json(arxiv_id: str) -> Optional[dict]:
    init_db_json()
    arxiv_id = _norm_arxiv_id(arxiv_id)
    if not arxiv_id:
        return None
    index = _json_load_index()
    rec = index.get("papers", {}).get(arxiv_id)
    if not isinstance(rec, dict):
        return None

    # Match the sqlite row->dict shape used elsewhere.
    return {
        "arxiv_id": rec.get("arxiv_id", arxiv_id),
        "title": rec.get("title"),
        "authors": list(rec.get("authors") or []),
        "summary": rec.get("summary"),
        "published": rec.get("published"),
        "url": rec.get("url"),
        "pdf_path": rec.get("pdf_path"),
        "namespaces": list(rec.get("namespaces") or []),
        "added_at": rec.get("added_at"),
        "updated_at": rec.get("updated_at"),
    }


def list_papers_json(
    *,
    title: str | None = None,
    author: str | None = None,
    keyword: str | None = None,
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    init_db_json()
    index = _json_load_index()
    papers = index.get("papers", {})
    rows: list[dict] = [get_paper_json(arxiv_id) for arxiv_id in papers.keys()]  # type: ignore[misc]
    rows = [r for r in rows if r is not None]

    def _lower(s: Any) -> str:
        return str(s or "").lower()

    if title:
        t = title.lower()
        rows = [r for r in rows if t in _lower(r.get("title"))]

    if author:
        a = author.lower()
        rows = [r for r in rows if a in _lower(", ".join(r.get("authors") or []))]

    if keyword:
        k = keyword.lower()

        def _row_text(r: dict) -> str:
            return " ".join(
                [
                    _lower(r.get("title")),
                    _lower(r.get("summary")),
                    _lower(", ".join(r.get("authors") or [])),
                ]
            )

        rows = [r for r in rows if k in _row_text(r)]

    def _year_of_published(published: Any) -> str:
        s = str(published or "")
        return s[:4]

    if year is not None:
        y = f"{int(year):04d}"
        rows = [r for r in rows if _year_of_published(r.get("published")) == y]
    if year_from is not None:
        y = f"{int(year_from):04d}"
        rows = [r for r in rows if _year_of_published(r.get("published")) >= y]
    if year_to is not None:
        y = f"{int(year_to):04d}"
        rows = [r for r in rows if _year_of_published(r.get("published")) <= y]

    def _sort_key(r: dict) -> str:
        return r.get("added_at") or r.get("updated_at") or ""

    rows.sort(key=_sort_key, reverse=True)
    return rows[int(offset) : int(offset) + int(limit)]


if not _SQLITE_AVAILABLE:
    init_db = init_db_json  # type: ignore[assignment]
    upsert_paper = upsert_paper_json  # type: ignore[assignment]
    get_paper = get_paper_json  # type: ignore[assignment]
    list_papers = list_papers_json  # type: ignore[assignment]

