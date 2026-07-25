from __future__ import annotations

import re
from typing import Any

from config import RAG_PAPER_FIGURE_TOP_K
from tools.agent.middleware import trace_event
from tools.rag.retrieval_merge import retrieve_with_public_merge

_FIG_NUM_IN_QUESTION_PAT = re.compile(
    r"\b(?:Figure|Fig\.|FIG|图)\s*(?:[:\-–—]?\s*)?([0-9]+(?:\.[0-9]+)?[A-Za-z]?)\b",
    re.IGNORECASE,
)


def parse_figure_numbers_from_question(question: str) -> list[str]:
    """从用户问题中解析 Figure / 图 编号（去重保序）。"""
    out: list[str] = []
    seen: set[str] = set()
    if not (question or "").strip():
        return out
    for m in _FIG_NUM_IN_QUESTION_PAT.finditer(question):
        n = str(m.group(1) or "").strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _base_figure_chroma_filter() -> dict[str, Any]:
    return {
        "$or": [
            {"chunk_role": "figure"},
            {"has_figure": True},
            {"type": "pdf_figure"},
            {"type": "figure_summary"},
        ]
    }


def retrieve_figure_evidence(
    store: Any,
    *,
    question: str,
    namespace: str,
    strategy: str,
    score_threshold: float,
    session_ingest_ids: list[str] | None,
    top_k: int | None = None,
) -> list[tuple[object, float]]:
    """图片/图表通道：caption / figure_number 过滤 + 语义检索。

    若问题中出现 ``Figure N`` 等编号，优先按 metadata ``figure_number`` 精确过滤；
    若无命中则回退为仅按 figure 类型过滤（兼容旧向量）。
    """
    k = int(top_k or RAG_PAPER_FIGURE_TOP_K)
    q = (question or "").strip()
    if not q:
        return []
    queries = [
        q,
        f"{q} figure caption architecture diagram ocr",
        f"{q} 图 结构图 流程图 图注 OCR",
    ]
    base = _base_figure_chroma_filter()
    nums = parse_figure_numbers_from_question(q)
    strict: dict[str, Any] | None = None
    if nums:
        strict = {"$and": [base, {"$or": [{"figure_number": n} for n in nums]}]}

    hits = retrieve_with_public_merge(
        store,
        queries=queries,
        namespace=namespace,
        k=max(3, k),
        score_threshold=score_threshold,
        strategy=strategy,
        session_ingest_ids=session_ingest_ids,
        extra_chroma_filter=strict or base,
    )
    if strict and not hits:
        trace_event(
            "figure_number_filter_fallback",
            {"namespace": namespace, "requested_numbers": ",".join(nums)},
        )
        hits = retrieve_with_public_merge(
            store,
            queries=queries,
            namespace=namespace,
            k=max(3, k),
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=base,
        )
    elif strict:
        trace_event(
            "figure_number_filter_applied",
            {"namespace": namespace, "numbers": ",".join(nums), "hits": str(len(hits))},
        )
    return hits[:k]
