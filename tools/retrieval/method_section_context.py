"""论文章节定向问答：从 PostgreSQL 拉取目标章节全部正文 chunk，合并后注入 RAG 提示词。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from config import RAG_PAPER_METHOD_INJECT_PG_FULL_SECTION, RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS
from tools.retrieval.query_understanding import analyze_paper_query
from tools.retrieval.table_retriever import extract_table_numbers_from_user_question
from tools.storage.repos.chunk_repo import (
    list_chunks_by_section_ids,
    list_table_role_chunks_for_paper,
)
from tools.storage.repos.paper_repo import list_tables
from tools.storage.repos.section_repo import (
    get_paper_by_arxiv_id,
    infer_section_role,
    list_sections_for_paper,
    looks_like_section_heading,
)
from tools.agent.middleware import trace_event

_Q_ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"(abstract|摘要|摘要部分)", re.I)),
    ("introduction", re.compile(r"(introduction|引言|背景|相关工作)", re.I)),
    ("method", re.compile(r"(method|approach|architecture|model|方法|方法部分|方法论|技术路线|模型结构|第\s*3\s*节|第三节)", re.I)),
    ("result", re.compile(r"(experiment|results?|evaluation|ablation|实验|结果|实验部分|性能)", re.I)),
]
_EXP_SETUP_Q = re.compile(r"(实验.*怎么做|怎么做.*实验|实验设置|实验方案|evaluation setup|experimental setup)", re.I)


def _target_roles_for_question(question: str) -> list[str]:
    q = question or ""
    qu = analyze_paper_query(q)
    intent = (qu.intent or "").strip()
    # 明确“实验怎么做/实验设置”时，优先拉 result/experiments，避免延续上一轮 method 注入。
    if _EXP_SETUP_Q.search(q):
        return ["result"]
    if intent == "paper_method":
        return ["method"]
    if intent in ("paper_result", "table_lookup"):
        return ["result"]
    # 指定问 abstract/introduction 时优先注入对应章节；paper_summary 不强制注入
    out: list[str] = []
    for role, pat in _Q_ROLE_PATTERNS:
        if pat.search(q):
            out.append(role)
    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _looks_focus_chunk_text(role: str, body: str) -> bool:
    txt = (body or "").strip()
    if not txt:
        return False
    head = txt[:600]
    if role == "method":
        return bool(re.search(r"(we propose|our method|pipeline|architecture|algorithm|loss|optimization|方法|提出|模型由)", head, re.I))
    if role == "introduction":
        return bool(re.search(r"(in this paper|we address|motivation|background|challenge|本文|我们提出|背景)", head, re.I))
    if role == "abstract":
        return bool(re.search(r"(we present|we propose|this paper|摘要|本文提出)", head, re.I))
    if role == "result":
        return bool(re.search(r"(results?|evaluation|ablation|performance|benchmark|实验结果|性能|对比)", head, re.I))
    return True


def build_focus_section_merged_document(namespace: str, role: str) -> Document | None:
    """从 PG 合并该论文某角色章节 chunk 正文（排除摘要/图/表）。"""
    target_role = (role or "").strip().lower()
    if target_role not in ("abstract", "introduction", "method", "result"):
        return None
    ns = (namespace or "").strip()
    if not ns.startswith("paper:"):
        return None
    arxiv_raw = ns.split(":", 1)[1].strip()
    if not arxiv_raw:
        return None
    aid = arxiv_raw.split(":", 1)[0].strip()
    row = get_paper_by_arxiv_id(aid)
    if not row:
        return None
    paper_id = int(row.get("id") or 0)
    if paper_id <= 0:
        return None
    sec_ids: list[int] = []
    for sec in list_sections_for_paper(paper_id):
        sid = int(sec.get("id") or 0)
        title = str(sec.get("title_norm") or sec.get("title") or "").strip()
        if sid <= 0 or not title:
            continue
        # 二次过滤：标题必须像章节标题，且角色匹配
        if not looks_like_section_heading(title):
            continue
        if infer_section_role(title) == target_role:
            sec_ids.append(sid)
    if not sec_ids:
        return None
    rows: list[dict[str, Any]] = list_chunks_by_section_ids(sec_ids, limit=600)
    if not rows:
        return None
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r.get("chunk_index") or 0),
            int(r.get("id") or 0),
        ),
    )
    parts: list[str] = []
    for r in rows_sorted:
        chunk_role = str(r.get("chunk_role") or "").lower().strip()
        if chunk_role in ("figure", "paper_summary", "table"):
            continue
        body = str(r.get("content") or "").strip()
        if body and _looks_focus_chunk_text(role=target_role, body=body):
            parts.append(body)
    text = "\n\n".join(parts).strip()
    if not text:
        return None
    cap = int(RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS or 200_000)
    suffix = ""
    if cap > 0 and len(text) > cap:
        text = text[:cap].rstrip()
        suffix = "\n\n...（章节合并文本已触达 RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS 上限）"
    meta = {
        "source": f"postgresql_{target_role}_sections_full",
        "type": f"paper_focus_pg_full_{target_role}",
        "arxiv_id": aid,
        "paper_id": paper_id,
        "focus_role": target_role,
        "focus_section_chunk_count": len(parts),
    }
    return Document(page_content=text + suffix, metadata=meta)


def _normalize_table_number_label(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"^0+(?=\d)", "", t)
    return t


def _pg_chunk_body_matches_table_numbers(content: str, nums: list[str]) -> bool:
    if not nums:
        return False
    blob = (content or "")[:12000].lower()
    for n in nums:
        n = str(n).strip()
        if not n:
            return False
        pats = [
            rf"\btable\s*{re.escape(n)}\b",
            rf"\btab\.\s*{re.escape(n)}\b",
            rf"table{re.escape(n)}\b",
            rf"表\s*{re.escape(n)}\b",
            rf"表格\s*{re.escape(n)}\b",
        ]
        if not any(re.search(p, blob) for p in pats):
            return False
    return True


def _select_table_chunks_for_numbers(
    chunks: list[dict[str, Any]], nums: list[str]
) -> list[dict[str, Any]]:
    """先按正文是否出现 Table N / 表N 匹配；否则按 chunk 顺序将第 i 个 table 块当作 Table i。"""
    if not chunks or not nums:
        return []
    hit = [
        c
        for c in chunks
        if _pg_chunk_body_matches_table_numbers(str(c.get("content") or ""), nums)
    ]
    if hit:
        return hit
    ordered = sorted(
        chunks,
        key=lambda x: (int(x.get("chunk_index") or 0), int(x.get("id") or 0)),
    )
    if len(nums) == 1:
        try:
            i = int(str(nums[0]).strip()) - 1
        except ValueError:
            i = -1
        if 0 <= i < len(ordered):
            return [ordered[i]]
    return []


def _prepend_paper_tables_pg_evidence(
    grouped: list[tuple[object, float]],
    *,
    namespace: str,
    question: str,
) -> list[tuple[object, float]]:
    """若 PostgreSQL 中已有 ``paper_tables`` 行，按问句中的 Table N 将整表注入上下文（不依赖向量排序）。"""
    if not RAG_PAPER_METHOD_INJECT_PG_FULL_SECTION:
        return grouped
    ns = (namespace or "").strip()
    if not ns.startswith("paper:"):
        return grouped
    q = question or ""
    qu = analyze_paper_query(q)
    nums = extract_table_numbers_from_user_question(q)
    if qu.intent != "table_lookup" and not nums:
        return grouped
    if not nums:
        return grouped

    arxiv_raw = ns.split(":", 1)[1].strip()
    if not arxiv_raw:
        return grouped
    aid = arxiv_raw.split(":", 1)[0].strip()
    row = get_paper_by_arxiv_id(aid)
    if not row:
        return grouped
    paper_id = int(row.get("id") or 0)
    if paper_id <= 0:
        return grouped

    tbl_rows = list_tables(paper_id)
    role_chunks = list_table_role_chunks_for_paper(paper_id)
    inject: list[tuple[object, float]] = []
    cap_body = 18_000

    if tbl_rows:
        want_set = {_normalize_table_number_label(n) for n in nums}
        picked_rows: list[dict[str, Any]] = []
        for r in tbl_rows:
            tn = r.get("table_number")
            if tn is None or str(tn).strip() == "":
                continue
            if _normalize_table_number_label(str(tn)) in want_set:
                picked_rows.append(r)

        if not picked_rows:
            blob_keys = ("caption_text", "summary_text", "markdown_text")
            for r in tbl_rows:
                blob = " ".join(str(r.get(k) or "") for k in blob_keys).lower()
                if not blob.strip():
                    continue
                if all(
                    (f"table {n}" in blob or f"table{n}" in blob or f"tab. {n}" in blob)
                    for n in nums
                ):
                    picked_rows.append(r)
                    break

        for i, r in enumerate(picked_rows):
            md = str(r.get("markdown_text") or "").strip()
            cap = str(r.get("caption_text") or "").strip()
            tnum = r.get("table_number")
            page_no = int(r.get("page_no") or 0)
            head = f"[PostgreSQL paper_tables"
            if tnum is not None and str(tnum).strip():
                head += f" Table {tnum}"
            if page_no:
                head += f" page {page_no}"
            head += "]"
            cap_line = f"\nCaption: {cap}" if cap else ""
            body = f"{head}{cap_line}\n\n{md}".strip()[:cap_body]
            if len(body) < 24:
                continue
            doc = Document(
                page_content=body,
                metadata={
                    "source": "postgresql_paper_tables",
                    "type": "paper_table_pg",
                    "arxiv_id": aid,
                    "paper_id": paper_id,
                    "chunk_role": "table",
                    "table_number": str(tnum).strip() if tnum is not None else None,
                    "page": page_no or None,
                    "has_table": True,
                },
            )
            inject.append((doc, -0.02 - float(i) * 1e-6))

    if not inject and role_chunks:
        selected = _select_table_chunks_for_numbers(role_chunks, nums)
        for i, c in enumerate(selected):
            content = str(c.get("content") or "").strip()
            body = content[:cap_body]
            if len(body) < 12:
                continue
            cid = c.get("id")
            pg0 = c.get("page_from")
            head = f"[PostgreSQL paper_chunks chunk_role=table chunk_id={cid}"
            if pg0 is not None:
                head += f" page~{pg0}"
            head += "]"
            doc = Document(
                page_content=f"{head}\n\n{body}",
                metadata={
                    "source": "postgresql_paper_chunks_table",
                    "type": "paper_table_pg",
                    "arxiv_id": aid,
                    "paper_id": paper_id,
                    "chunk_role": "table",
                    "paper_chunk_id": cid,
                    "has_table": True,
                },
            )
            inject.append((doc, -0.019 - float(i) * 1e-6))
        if inject:
            trace_event(
                "paper_table_pg_chunk_fallback_inject",
                {
                    "arxiv_id": aid,
                    "paper_id": paper_id,
                    "count": len(inject),
                    "chunk_ids": [c.get("id") for c in selected],
                    "requested_numbers": list(nums),
                    "question": q[:220],
                },
            )

    if not inject:
        trace_event(
            "paper_table_pg_inject_skip",
            {
                "reason": "no_paper_tables_match_and_no_table_chunk_fallback",
                "arxiv_id": aid,
                "paper_id": paper_id,
                "paper_tables_rows": len(tbl_rows),
                "table_role_chunks": len(role_chunks),
                "requested_numbers": nums,
                "question": q[:220],
            },
        )
        return grouped

    if any(
        isinstance(getattr(d, "metadata", None), dict)
        and (getattr(d, "metadata", {}) or {}).get("source") == "postgresql_paper_tables"
        for d, _ in inject
    ):
        trace_event(
            "paper_table_pg_inject",
            {
                "arxiv_id": aid,
                "paper_id": paper_id,
                "count": len(inject),
                "table_numbers": list(nums),
                "question": q[:220],
            },
        )
    return inject + list(grouped)


def prepend_method_section_full_context(
    grouped: list[tuple[object, float]],
    *,
    namespace: str,
    question: str,
) -> list[tuple[object, float]]:
    # 兼容函数名：现已支持 method/introduction/abstract/result 多角色注入。
    if not RAG_PAPER_METHOD_INJECT_PG_FULL_SECTION:
        return grouped
    grouped = _prepend_paper_tables_pg_evidence(
        grouped, namespace=namespace, question=question
    )
    roles = _target_roles_for_question(question)
    if not roles:
        return grouped
    target = set(roles)
    # 保险：先清理历史残留的 focus 注入块（跨轮切换 method/result 时避免污染）
    cleaned: list[tuple[object, float]] = []
    removed_roles: set[str] = set()
    for d0, sc0 in grouped:
        m0 = getattr(d0, "metadata", None) or {}
        if not isinstance(m0, dict):
            cleaned.append((d0, sc0))
            continue
        tp = str(m0.get("type") or "")
        if not tp.startswith("paper_focus_pg_full_"):
            cleaned.append((d0, sc0))
            continue
        role0 = str(m0.get("focus_role") or "").strip().lower()
        if role0 in target:
            cleaned.append((d0, sc0))
        else:
            if role0:
                removed_roles.add(role0)
    # 若已存在本轮目标角色注入块，不重复插入
    existing_roles: set[str] = set()
    for d0, _ in cleaned[:8]:
        m0 = getattr(d0, "metadata", None) or {}
        if isinstance(m0, dict) and str(m0.get("type") or "").startswith("paper_focus_pg_full_"):
            role0 = str(m0.get("focus_role") or "").strip().lower()
            if role0:
                existing_roles.add(role0)
    inject_docs: list[tuple[object, float]] = []
    for idx, r in enumerate(roles):
        if r in existing_roles:
            continue
        doc = build_focus_section_merged_document(namespace, r)
        if doc is not None:
            inject_docs.append((doc, float(idx) * 1e-6))
    if inject_docs:
        injected_roles: list[str] = []
        injected_chunks: list[int] = []
        for doc0, _sc in inject_docs:
            m1 = getattr(doc0, "metadata", None) or {}
            if isinstance(m1, dict):
                injected_roles.append(str(m1.get("focus_role") or "").strip().lower())
                injected_chunks.append(int(m1.get("focus_section_chunk_count") or 0))
        trace_event(
            "paper_focus_context_reset",
            {
                "target_roles": sorted(list(target)),
                "removed_focus_roles": sorted(list(removed_roles)),
                "kept_grouped_after_clean": len(cleaned),
                "injected_focus_roles": injected_roles,
                "injected_focus_chunk_counts": injected_chunks,
                "question": (question or "")[:220],
            },
        )
        return inject_docs + cleaned

    trace_event(
        "paper_focus_context_reset",
        {
            "target_roles": sorted(list(target)),
            "removed_focus_roles": sorted(list(removed_roles)),
            "kept_grouped_after_clean": len(cleaned),
            "injected_focus_roles": [],
            "injected_focus_chunk_counts": [],
            "question": (question or "")[:220],
        },
    )
    return cleaned
