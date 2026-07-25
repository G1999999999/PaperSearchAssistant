#!/usr/bin/env python3
"""
检查 PostgreSQL 中 paper_sections 在四个 focus 角色上的命中质量：
abstract / introduction / method / result
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

FOCUS_ROLES = ("abstract", "introduction", "method", "result")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PG focus section roles")
    parser.add_argument("--arxiv-id", type=str, default="", help="仅检查某个 arXiv id")
    parser.add_argument("--top", type=int, default=10, help="每个角色最多展示前 N 篇论文")
    parser.add_argument(
        "--sample-per-paper",
        type=int,
        default=4,
        help="每篇论文每个角色展示多少个 section 标题样本",
    )
    parser.add_argument("--print-all", action="store_true", help="打印全部标题（不截断）")
    args = parser.parse_args()

    from tools.storage.sql.db import get_session_factory, is_database_configured
    from tools.storage.sql.models import Paper, PaperSection
    from tools.storage.repos.section_repo import infer_section_role, looks_like_section_heading

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
            .order_by(Paper.id.asc(), PaperSection.order_index.asc())
        )
        if args.arxiv_id.strip():
            stmt = stmt.where(Paper.arxiv_id == args.arxiv_id.strip())
        rows = list(session.execute(stmt).mappings().all())
    finally:
        session.close()

    if not rows:
        print("No paper_sections rows found.")
        return 0

    papers: dict[int, dict[str, Any]] = {}
    total_sections = 0
    # paper_id -> role -> list[section-dict]
    hits: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {r: [] for r in FOCUS_ROLES}
    )
    non_heading_rows = 0

    for r in rows:
        total_sections += 1
        pid = int(r["paper_id"])
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "arxiv_id": r.get("arxiv_id"),
                "title": r.get("paper_title") or "",
            }
        title_norm = str(r.get("section_title_norm") or "").strip()
        title = str(r.get("section_title") or "").strip()
        probe = title_norm or title
        if not looks_like_section_heading(probe):
            non_heading_rows += 1
            continue
        role = infer_section_role(probe)
        if role not in FOCUS_ROLES:
            continue
        hits[pid][role].append(
            {
                "section_id": int(r.get("section_id") or 0),
                "title": title,
                "title_norm": title_norm,
                "section_number": r.get("section_number"),
                "order_index": int(r.get("order_index") or 0),
            }
        )

    total_papers = len(papers)
    print(f"Total papers: {total_papers}")
    print(f"Total paper_sections rows scanned: {total_sections}")
    print(f"Rows filtered as non-heading: {non_heading_rows}")

    # role-wise summary
    role_to_papers: dict[str, list[int]] = {r: [] for r in FOCUS_ROLES}
    for pid in papers:
        for role in FOCUS_ROLES:
            if hits[pid][role]:
                role_to_papers[role].append(pid)

    print("\nRole coverage:")
    for role in FOCUS_ROLES:
        print(f"- {role}: papers_with_hits={len(role_to_papers[role])}")

    top_n = max(1, int(args.top or 10))
    sample_n = max(1, int(args.sample_per_paper or 4))
    for role in FOCUS_ROLES:
        pids = sorted(
            role_to_papers[role],
            key=lambda pid: len(hits[pid][role]),
            reverse=True,
        )
        print(f"\n=== {role.upper()} ===")
        if not pids:
            print("  (no hits)")
            continue
        for pid in pids[:top_n]:
            info = papers[pid]
            sec_rows = sorted(hits[pid][role], key=lambda x: x["order_index"])
            print(
                f"- arXiv: {info.get('arxiv_id')} | paper_id={pid} | {role}_sections={len(sec_rows)}"
            )
            show_rows = sec_rows
            if not args.print_all and len(sec_rows) > sample_n:
                show_rows = sec_rows[:sample_n]
                print(
                    f"  (titles truncated; use --print-all to show all {len(sec_rows)} rows)"
                )
            for s in show_rows:
                sec_no = s.get("section_number") or ""
                print(
                    f"  * [{sec_no}] order={s.get('order_index')} title={s.get('title')}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

