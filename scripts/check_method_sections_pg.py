#!/usr/bin/env python3
"""
检查 PostgreSQL 中是否存在“method 角色”的 paper_sections。

逻辑：
1) 读取 papers / paper_sections（title_norm/title）
2) 通过 tools.storage.repos.section_repo.infer_section_role() 推断每个 section 的 role
3) 汇总每篇论文是否存在 role=method 的 section
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any

from sqlalchemy import select

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PG method-role paper_sections")
    parser.add_argument("--arxiv-id", type=str, default="", help="仅检查某个 arXiv id（如 2603.26665）")
    parser.add_argument("--top", type=int, default=10, help="展示前 N 篇命中 method 的论文")
    parser.add_argument("--print-all", action="store_true", help="打印所有 method section 标题")
    args = parser.parse_args()

    from tools.storage.sql.db import get_session_factory, is_database_configured
    from tools.storage.sql.models import Paper, PaperSection
    from tools.storage.repos.section_repo import infer_section_role

    if not is_database_configured():
        print("PostgreSQL: skip — DATABASE_URL 未设置或无法连接")
        return 1

    factory = get_session_factory()
    if factory is None:
        print("PostgreSQL: skip — get_session_factory() 返回 None")
        return 1

    session = factory()
    try:
        stmt = (
            select(
                Paper.id.label("paper_id"),
                Paper.arxiv_id.label("arxiv_id"),
                Paper.title.label("paper_title"),
                PaperSection.id.label("section_id"),
                PaperSection.title_norm.label("section_title_norm"),
                PaperSection.title.label("section_title"),
                PaperSection.section_number,
                PaperSection.order_index,
            )
            .select_from(Paper)
            .join(PaperSection, PaperSection.paper_id == Paper.id)
        )

        if args.arxiv_id.strip():
            aid = args.arxiv_id.strip()
            stmt = stmt.where(Paper.arxiv_id == aid)

        rows = list(session.execute(stmt).mappings().all())
    finally:
        session.close()

    if not rows:
        if args.arxiv_id.strip():
            print(f"arXiv {args.arxiv_id}: no paper_sections rows")
        else:
            print("No paper_sections rows found.")
        return 0

    per_paper: dict[int, dict[str, Any]] = {}
    method_sections_by_paper: dict[int, list[dict[str, Any]]] = defaultdict(list)

    total_sections = 0
    for r in rows:
        total_sections += 1
        pid = int(r["paper_id"])
        arxiv_id = r.get("arxiv_id")
        if pid not in per_paper:
            per_paper[pid] = {
                "paper_id": pid,
                "arxiv_id": arxiv_id,
                "title": r.get("paper_title") or "",
            }

        sec_title_norm = str(r.get("section_title_norm") or "").strip()
        sec_title = str(r.get("section_title") or "").strip()
        probe = sec_title_norm or sec_title
        role = infer_section_role(probe)
        if role == "method":
            method_sections_by_paper[pid].append(
                {
                    "section_id": int(r.get("section_id") or 0),
                    "title": sec_title,
                    "title_norm": sec_title_norm,
                    "section_number": r.get("section_number"),
                    "order_index": r.get("order_index"),
                }
            )

    total_papers = len(per_paper)
    hit_papers = [pid for pid in per_paper.keys() if method_sections_by_paper.get(pid)]

    print(f"Total papers: {total_papers}")
    print(f"Total paper_sections rows scanned: {total_sections}")
    print(f"Papers with role=method: {len(hit_papers)}")

    hit_papers_sorted = sorted(
        hit_papers,
        key=lambda pid: len(method_sections_by_paper.get(pid) or []),
        reverse=True,
    )
    show_n = max(1, int(args.top or 10))
    for pid in hit_papers_sorted[:show_n]:
        info = per_paper.get(pid) or {}
        ms = method_sections_by_paper.get(pid) or []
        print(
            f"\n- arXiv: {info.get('arxiv_id')} | paper_id={pid} | method_sections={len(ms)}"
        )
        if not (args.print_all or len(ms) <= 8):
            print("  (method section titles truncated; use --print-all to show all)")
            ms = sorted(ms, key=lambda x: x.get("order_index") or 0)[:8]
        for s in sorted(ms, key=lambda x: x.get("order_index") or 0):
            sec_no = s.get("section_number") or ""
            oi = s.get("order_index") or ""
            print(f"  * [{sec_no}] order={oi} title={s.get('title')}")

    if not hit_papers_sorted:
        print("\nNo method-role sections detected. This usually means:")
        print(" - paper_sections extraction stored only generic headings")
        print(" - or title/title_norm missing standard method keywords")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

