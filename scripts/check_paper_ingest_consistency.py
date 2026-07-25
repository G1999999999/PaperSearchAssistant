#!/usr/bin/env python3
"""检查单篇论文在 SQLite / PostgreSQL / Chroma 的入库一致性。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import func, select

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

from tools.storage.paper_library import get_paper
from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import (
    Paper,
    PaperChunk,
    PaperFigure,
    PaperSection,
    PaperSummaryView,
    PaperTable,
)


def _check_sqlite(arxiv_id: str) -> tuple[bool, dict]:
    rec = get_paper(arxiv_id)
    if not rec:
        return False, {"exists": False}
    return True, {
        "exists": True,
        "pdf_path": rec.get("pdf_path"),
        "title": rec.get("title"),
        "namespaces": rec.get("namespaces") or [],
        "updated_at": rec.get("updated_at"),
    }


def _check_pg(arxiv_id: str) -> tuple[bool, dict]:
    factory = get_session_factory()
    if factory is None:
        return False, {"enabled": False, "reason": "DATABASE_URL unavailable"}

    session = factory()
    try:
        p = session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if p is None:
            return False, {"enabled": True, "exists": False}
        pid = int(p.id)
        chunks = int(
            session.scalar(select(func.count()).select_from(PaperChunk).where(PaperChunk.paper_id == pid))
            or 0
        )
        sections = int(
            session.scalar(select(func.count()).select_from(PaperSection).where(PaperSection.paper_id == pid))
            or 0
        )
        figures = int(
            session.scalar(select(func.count()).select_from(PaperFigure).where(PaperFigure.paper_id == pid))
            or 0
        )
        tables = int(
            session.scalar(select(func.count()).select_from(PaperTable).where(PaperTable.paper_id == pid))
            or 0
        )
        summary = int(
            session.scalar(select(func.count()).select_from(PaperSummaryView).where(PaperSummaryView.paper_id == pid))
            or 0
        )
        return True, {
            "enabled": True,
            "exists": True,
            "paper_id": pid,
            "title": p.title,
            "chunks": chunks,
            "sections": sections,
            "figures": figures,
            "tables": tables,
            "summary_view": summary,
        }
    finally:
        session.close()


def _check_chroma(arxiv_id: str, probe_query: str, top_k: int) -> tuple[bool, dict]:
    namespace = f"paper:{arxiv_id}:full"
    try:
        from tools.rag.knowledge import vector_store
    except Exception as e:  # pragma: no cover - env specific
        return False, {"enabled": False, "reason": f"import failed: {e}"}

    try:
        hits = vector_store.retrieve(
            queries=[probe_query],
            namespace=namespace,
            k=max(1, int(top_k)),
            score_threshold=2.0,
            # 一致性自检避免拉取 CrossEncoder（可能需要联网），用纯混合检索即可
            strategy="hybrid",
        )
    except Exception as e:  # pragma: no cover - env specific
        return False, {"enabled": True, "namespace": namespace, "error": str(e)}

    first_meta = {}
    if hits:
        d, _s = hits[0]
        md = getattr(d, "metadata", {}) or {}
        if isinstance(md, dict):
            first_meta = {
                "type": md.get("type"),
                "chunk_role": md.get("chunk_role"),
                "source": md.get("source"),
                "arxiv_id": md.get("arxiv_id"),
            }
    ok = len(hits) > 0
    return ok, {
        "enabled": True,
        "namespace": namespace,
        "probe_query": probe_query,
        "hits": len(hits),
        "first_hit_meta": first_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ingest consistency across SQLite/PG/Chroma")
    ap.add_argument("--arxiv-id", required=True, help="Paper arXiv id, e.g. 2312.00732")
    ap.add_argument("--probe-query", default="summary", help="Probe query for vector retrieval")
    ap.add_argument("--top-k", type=int, default=6, help="Probe top-k for vector retrieval")
    ap.add_argument(
        "--auto-repair",
        action="store_true",
        help="When PG is OK but Chroma MISS, auto-run full ingest repair for this paper",
    )
    args = ap.parse_args()

    arxiv_id = str(args.arxiv_id).strip()
    if not arxiv_id:
        print("arxiv_id is required")
        return 2

    s_ok, s = _check_sqlite(arxiv_id)
    p_ok, p = _check_pg(arxiv_id)
    c_ok, c = _check_chroma(arxiv_id, str(args.probe_query), int(args.top_k))

    print(f"[SQLite] {'OK' if s_ok else 'MISS'}: {s}")
    print(f"[PostgreSQL] {'OK' if p_ok else 'MISS'}: {p}")
    print(f"[Chroma] {'OK' if c_ok else 'MISS'}: {c}")

    issues: list[str] = []
    if not s_ok:
        issues.append("SQLite lacks this paper metadata")
    if not p_ok and p.get("enabled", True):
        issues.append("PostgreSQL lacks this paper row")
    if not c_ok:
        issues.append("Chroma has no retrievable vectors for paper namespace")
    if p_ok and p.get("chunks", 0) == 0:
        issues.append("PostgreSQL paper_chunks count is 0")

    pg_ok_with_chunks = bool(p_ok and int(p.get("chunks", 0)) > 0)
    chroma_miss = not c_ok
    if args.auto_repair and pg_ok_with_chunks and chroma_miss:
        print("\nAuto-repair triggered: PG looks healthy but Chroma MISS.")
        try:
            from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline

            msg = ingest_arxiv_paper_full_pipeline(arxiv_id, embed_full_text=True)
            print("\nRepair run output:")
            print(msg)
        except Exception as e:
            print(f"\nAuto-repair failed: {e}")
            return 2

        # Re-check only Chroma after repair
        c_ok2, c2 = _check_chroma(arxiv_id, str(args.probe_query), int(args.top_k))
        print(f"\n[Chroma recheck] {'OK' if c_ok2 else 'MISS'}: {c2}")
        if c_ok2:
            print("\nConsistency check passed after auto-repair.")
            return 0
        print("\nAuto-repair executed, but Chroma is still MISS.")
        print("Please inspect Chroma runtime/storage config and embedding pipeline logs.")
        return 1

    if issues:
        print("\nDetected issues:")
        for x in issues:
            print(f"- {x}")
        print("\nSuggested action:")
        print(
            f"python3 -c \"from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline; "
            f"print(ingest_arxiv_paper_full_pipeline('{arxiv_id}', embed_full_text=True))\""
        )
        return 1

    print("\nConsistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

