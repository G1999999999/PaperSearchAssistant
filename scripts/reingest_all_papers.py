#!/usr/bin/env python3
"""清空论文相关数据后，批量将 data/papers 下 PDF 重新入库（PostgreSQL + SQLite + Chroma）。

用法（在 PaperSearchAssistant2 目录下）::

    set -a && source .env.runtime
    export RAG_INGEST_SECTION_MODE=section_aware   # 可选：按章节重切分入库
    python3 scripts/reingest_all_papers.py --yes

默认只清空「论文」相关表（papers / chunks / sections 等），**不删** chat_sessions。
若要连会话表一起清空，加 ``--full-db``（危险）。

说明：
- 需要可用的 DATABASE_URL（PostgreSQL）才会清 PG；否则仅清 SQLite 并尝试入库。
- Chroma 依赖本机 sqlite + chromadb；若向量入库失败，脚本仍会打印每篇的错误信息。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env.runtime"))
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass


_ARXIV_STEM = re.compile(r"^(\d{4}\.\d{4,5}(?:v\d+)?)$", re.I)


def _arxiv_id_from_filename(stem: str) -> str | None:
    s = (stem or "").strip()
    if not s:
        return None
    if _ARXIV_STEM.match(s):
        return s.lower().replace("v", "v")  # 保留 v1 等形式
    # 宽松：去掉扩展名后只保留数字点形式
    m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", s, re.I)
    return m.group(1).lower() if m else None


def _clear_postgresql(*, full_db: bool) -> None:
    from sqlalchemy import text

    from tools.storage.sql.db import get_session_factory

    factory = get_session_factory()
    if factory is None:
        print("PostgreSQL: skip — DATABASE_URL 未配置或无法连接")
        return

    session = factory()
    try:
        if full_db:
            session.execute(
                text(
                    "TRUNCATE TABLE chat_attachments, chat_messages, chat_sessions "
                    "RESTART IDENTITY CASCADE"
                )
            )
            print("PostgreSQL: 已 TRUNCATE chat_*")
        session.execute(text("TRUNCATE TABLE paper_ingest_jobs RESTART IDENTITY"))
        session.execute(text("TRUNCATE TABLE papers RESTART IDENTITY CASCADE"))
        session.commit()
        print("PostgreSQL: 已 TRUNCATE paper_ingest_jobs + papers CASCADE（含 chunks/sections/figures/...）")
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"PostgreSQL 清空失败: {e}") from e
    finally:
        session.close()


def _clear_sqlite_papers() -> None:
    from pathlib import Path

    import sqlite3

    db_path = Path(_ROOT) / "data" / "papers" / "papers.db"
    if not db_path.is_file():
        print(f"SQLite: skip — 不存在 {db_path}")
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM papers")
        try:
            conn.execute("DELETE FROM papers_fts")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        print(f"SQLite: 已清空 {db_path} 中 papers（及 FTS 若存在）")
    finally:
        conn.close()


def _chroma_clear_paper_namespaces(arxiv_ids: list[str]) -> None:
    try:
        from tools.rag.knowledge import vector_store
    except Exception as e:
        print(f"Chroma: skip import — {e}")
        return
    for aid in arxiv_ids:
        ns = f"paper:{aid}:full"
        try:
            vector_store.clear_namespace(ns)
            print(f"Chroma: 已清空 namespace {ns}")
        except Exception as e:
            print(f"Chroma: 清空 {ns} 失败 — {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="清空论文库数据并批量重新入库 PDF")
    parser.add_argument(
        "--papers-dir",
        default=os.path.join(_ROOT, "data", "papers"),
        help="PDF 目录（默认: data/papers）",
    )
    parser.add_argument(
        "--full-db",
        action="store_true",
        help="同时 TRUNCATE chat_sessions / chat_messages / chat_attachments（危险）",
    )
    parser.add_argument(
        "--skip-pg",
        action="store_true",
        help="不清空 PostgreSQL",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="不清空 SQLite 本地 papers.db",
    )
    parser.add_argument(
        "--chroma-clear",
        action="store_true",
        help="入库前对每个 arXiv id 执行 clear_namespace(paper:<id>:full)，避免旧向量残留",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="跳过交互确认（批量脚本必加）",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "fast", "full"],
        default=(os.getenv("RAG_INGEST_PROFILE", "auto") or "auto").strip().lower(),
        help="入库策略：auto(批量自动fast) / fast(极速正文) / full(完整图表)",
    )
    args = parser.parse_args()

    # 批量重入库场景：标记 bulk，供 ingest 策略自动选择。
    os.environ["RAG_INGEST_BULK"] = "1"
    os.environ["RAG_INGEST_PROFILE"] = str(args.profile)

    papers_dir = os.path.abspath(args.papers_dir)
    pattern = os.path.join(papers_dir, "*.pdf")
    pdfs = sorted(glob.glob(pattern))
    if not pdfs:
        print(f"未找到 PDF: {pattern}")
        return 1

    stems: list[str] = []
    seen: set[str] = set()
    for p in pdfs:
        base = os.path.splitext(os.path.basename(p))[0]
        aid = _arxiv_id_from_filename(base)
        if not aid:
            print(f"跳过（无法从文件名解析 arXiv id）: {p}")
            continue
        if aid in seen:
            continue
        seen.add(aid)
        stems.append(aid)

    if not stems:
        print("没有可入库的 PDF（文件名需类似 2401.12345.pdf）")
        return 1

    print("将处理以下 arXiv id:", ", ".join(stems))
    print(f"RAG_INGEST_PROFILE={os.getenv('RAG_INGEST_PROFILE', 'auto')}")
    print(f"RAG_INGEST_BULK={os.getenv('RAG_INGEST_BULK', '0')}")

    if not args.yes:
        try:
            ans = input("确认清空数据库并重新入库? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("已取消")
            return 2

    if not args.skip_pg:
        _clear_postgresql(full_db=bool(args.full_db))
    else:
        print("PostgreSQL: skip (--skip-pg)")

    if not args.skip_sqlite:
        _clear_sqlite_papers()
    else:
        print("SQLite: skip (--skip-sqlite)")

    if args.chroma_clear:
        _chroma_clear_paper_namespaces(stems)

    from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline

    ok = 0
    failed: list[tuple[str, str]] = []
    for aid in stems:
        print(f"\n=== 入库 {aid} ===")
        try:
            msg = ingest_arxiv_paper_full_pipeline(aid, embed_full_text=True)
            print(msg)
            if "但全文向量入库失败" in msg or "向量入库失败" in msg:
                failed.append((aid, msg))
            else:
                ok += 1
        except Exception as e:
            failed.append((aid, str(e)))
            print(f"异常: {e}")

    print(f"\n完成: 成功约 {ok}/{len(stems)}，失败 {len(failed)}")
    for aid, err in failed:
        print(f"  - {aid}: {err[:300]}...")
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
