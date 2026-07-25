#!/usr/bin/env python3
"""查看论文相关存储：SQLite 本地库、PostgreSQL（若配置）、Chroma 向量条数。

请在项目根目录执行（与 main.py 同级）::

    cd PaperSearchAssistant2
    python scripts/inspect_db.py
    python scripts/inspect_db.py --arxiv 2312.00732
    python scripts/inspect_db.py --arxiv 2312.00732 --chroma
    python scripts/inspect_db.py --sqlite-sql "SELECT arxiv_id, title FROM papers LIMIT 5;"

纯 SQLite 也可直接用系统命令::

    sqlite3 data/papers/papers.db ".schema papers"
    sqlite3 data/papers/papers.db "SELECT arxiv_id, title, namespaces_json FROM papers;"
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _chdir_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _print_sqlite_overview(*, limit: int) -> None:
    from tools.storage import papers_db

    papers_db.init_db()
    rows = papers_db.list_papers(limit=limit, offset=0)
    print(f"\n=== SQLite 本地库 ({papers_db.DB_PATH}) 前 {len(rows)} 条（limit={limit}）===\n")
    for r in rows:
        aid = r.get("arxiv_id")
        title = (r.get("title") or "")[:80]
        ns = r.get("namespaces") or []
        pdf = r.get("pdf_path") or ""
        print(f"  {aid}\t{title}")
        print(f"           namespaces: {ns}")
        print(f"           pdf: {pdf}\n")


def _run_sqlite_sql(sql: str) -> None:
    from tools.storage import papers_db

    papers_db.init_db()
    conn = papers_db._connect()  # noqa: SLF001
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        print(json.dumps({"columns": cols}, ensure_ascii=False))
        batch = cur.fetchmany(500)
        while batch:
            for row in batch:
                print(json.dumps(list(row), ensure_ascii=False))
            batch = cur.fetchmany(500)
    finally:
        conn.close()


def _print_postgres_overview(*, arxiv: str | None, sample_chunks: int) -> None:
    from sqlalchemy import func, select

    from tools.storage.sql.db import get_session_factory, is_database_configured
    from tools.storage.sql.models import Paper, PaperChunk

    if not is_database_configured():
        print("\n=== PostgreSQL：未配置 DATABASE_URL，跳过 ===\n")
        return

    factory = get_session_factory()
    if factory is None:
        print("\n=== PostgreSQL：无法创建会话，跳过 ===\n")
        return

    sess = factory()
    try:
        total = sess.scalar(select(func.count()).select_from(Paper))
        print(f"\n=== PostgreSQL papers 表：约 {total} 行 ===\n")
        if not arxiv:
            stmt = (
                select(Paper.id, Paper.arxiv_id, Paper.title)
                .order_by(Paper.id.desc())
                .limit(15)
            )
            for p in sess.execute(stmt).all():
                print(f"  id={p.id}\tarxiv={p.arxiv_id}\t{(p.title or '')[:72]}")
            return

        aid = arxiv.strip()
        p = sess.scalar(select(Paper).where(Paper.arxiv_id == aid))
        if p is None:
            print(f"  （未找到 arxiv_id={aid} 的 PostgreSQL 记录；可能未开 RAG_PG_SYNC_ON_INGEST）\n")
            return

        print(f"  paper.id={p.id}  title={(p.title or '')[:100]}\n")
        role_stmt = (
            select(PaperChunk.chunk_role, func.count())
            .where(PaperChunk.paper_id == int(p.id))
            .group_by(PaperChunk.chunk_role)
        )
        print("  paper_chunks 按 chunk_role 计数：")
        for role, c in sess.execute(role_stmt).all():
            print(f"    {role}: {c}")

        samp = sess.scalars(
            select(PaperChunk)
            .where(PaperChunk.paper_id == int(p.id))
            .order_by(PaperChunk.chunk_index.asc())
            .limit(sample_chunks)
        ).all()
        print(f"\n  前 {len(samp)} 条分块预览（chunk_index / role / content[:240]）：")
        for ch in samp:
            body = (ch.content or "").replace("\n", " ")[:240]
            print(f"    [{ch.chunk_index}] role={ch.chunk_role}  {body}…")
    finally:
        sess.close()


def _chroma_count(namespace: str) -> int | None:
    from tools.rag.knowledge import CHROMA_PERSIST_DIR, vector_store

    ns = (namespace or "").strip()
    if not ns:
        return None
    try:
        store = vector_store._get_or_create_store(ns)  # noqa: SLF001
        coll = getattr(store, "_collection", None)
        if coll is None:
            return None
        return int(coll.count())
    except Exception as e:
        print(f"  Chroma 读取失败 namespace={ns!r} persist={CHROMA_PERSIST_DIR}: {e}")
        return None


def main() -> None:
    _chdir_project_root()

    ap = argparse.ArgumentParser(description="查看 SQLite / PostgreSQL / Chroma 中的论文数据")
    ap.add_argument("--arxiv", type=str, default=None, help="指定 arXiv ID，打印 PostgreSQL 分块统计与预览")
    ap.add_argument("--limit", type=int, default=30, help="SQLite list_papers 条数上限")
    ap.add_argument("--no-sqlite", action="store_true", help="跳过 SQLite 概览")
    ap.add_argument(
        "--sqlite-sql",
        type=str,
        default=None,
        help="对 data/papers/papers.db 执行只读 SQL（结果 JSON 行输出）",
    )
    ap.add_argument("--chroma", action="store_true", help="若提供 --arxiv，额外打印 paper:<id>:full 的 Chroma 条数")
    ap.add_argument(
        "--chroma-namespace",
        type=str,
        default=None,
        help="直接查看某 namespace 的 Chroma 文档条数（优先于根据 --arxiv 推导）",
    )
    ap.add_argument("--sample-chunks", type=int, default=4, help="PostgreSQL 分块预览条数")
    args = ap.parse_args()

    if args.sqlite_sql:
        _run_sqlite_sql(args.sqlite_sql)
        return

    if not args.no_sqlite:
        _print_sqlite_overview(limit=max(1, min(args.limit, 500)))

    _print_postgres_overview(arxiv=args.arxiv, sample_chunks=max(1, min(args.sample_chunks, 50)))

    chroma_ns = (args.chroma_namespace or "").strip()
    if not chroma_ns and args.arxiv and args.chroma:
        v = (args.arxiv or "").strip()
        if v:
            chroma_ns = f"paper:{v}:full"

    if chroma_ns:
        n = _chroma_count(chroma_ns)
        print(f"\n=== Chroma namespace={chroma_ns!r} 文档条数: {n} ===\n")
    elif args.chroma:
        print("\n=== Chroma：请附带 --arxiv ID 或 --chroma-namespace ===\n")


if __name__ == "__main__":
    main()
