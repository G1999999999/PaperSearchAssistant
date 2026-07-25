from __future__ import annotations

import re
from typing import Any

from config import RAG_PAPER_TABLE_TOP_K
from tools.rag.retrieval_merge import retrieve_with_public_merge

# 紧贴中文时也需匹配到，故不用 \b 锚在左侧（例如「分析Table1」）
_TABLE_NUM_EN = re.compile(
    r"(?:Table|TAB|Tab\.)\s*([0-9]+(?:\.[0-9]+)?[A-Za-z]?)",
    re.IGNORECASE,
)
# 「表1」「表 1」与「表格1」「表 格 1」等（「表格1」中间有「格」，不能只用 表\s*数字）
_TABLE_NUM_ZH = re.compile(r"(?:表格|表)\s*(\d+)", re.IGNORECASE)


def extract_table_numbers_from_user_question(question: str) -> list[str]:
    """从用户问句中提取 Table/Tab./表/表格 编号（去重保序）。"""
    q = question or ""
    seen: set[str] = set()
    out: list[str] = []
    for pat in (_TABLE_NUM_EN, _TABLE_NUM_ZH):
        for m in pat.finditer(q):
            raw = str(m.group(1) or "").strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            out.append(raw)
    return out


def retrieve_table_evidence(
    store: Any,
    *,
    question: str,
    namespace: str,
    strategy: str,
    score_threshold: float,
    session_ingest_ids: list[str] | None,
    top_k: int | None = None,
) -> list[tuple[object, float]]:
    """表格通道：caption/metric 强化查询 + table 元数据过滤。"""
    k = int(top_k or RAG_PAPER_TABLE_TOP_K)
    q = (question or "").strip()
    if not q:
        return []
    tnums = extract_table_numbers_from_user_question(q)
    queries = [
        q,
        f"{q} table results metrics",
        f"{q} 表 指标 对比 结果",
    ]
    for n in tnums:
        queries.extend(
            [
                f"Table {n} Tab. {n} caption",
                f"Table {n} benchmark metrics comparison results",
                f"table number {n} pdf_table",
            ]
        )
    filt = {
        "$or": [
            {"chunk_role": "table"},
            {"has_table": True},
            {"type": "table_summary"},
            {"type": "pdf_table"},
        ]
    }
    hits = retrieve_with_public_merge(
        store,
        queries=queries,
        namespace=namespace,
        k=max(3, k),
        score_threshold=score_threshold,
        strategy=strategy,
        session_ingest_ids=session_ingest_ids,
        extra_chroma_filter=filt,
    )
    return hits[:k]

