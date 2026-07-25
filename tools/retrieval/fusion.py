from __future__ import annotations

import re
from typing import Callable, List, Tuple

from config import (
    RAG_PAPER_METHOD_DOWNRANK_TABLE_CHUNKS,
    RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY,
    RRF_K,
)


def _doc_key(doc: object) -> str:
    if hasattr(doc, "page_content"):
        return str(getattr(doc, "page_content", "") or "")
    return str(doc)


def weighted_rrf_fusion(
    ranked_lists: List[Tuple[List[Tuple[object, float]], float]],
    *,
    k: int = RRF_K,
    intent_boost_fn: Callable[[object], float] | None = None,
) -> List[Tuple[object, float]]:
    """带权重的 RRF：每路列表乘以 channel 权重；可选对单条文档按意图再乘 boost。"""
    if not ranked_lists:
        return []
    rrf_scores: dict[str, tuple[object, float]] = {}
    for lst, weight in ranked_lists:
        if weight <= 0 or not lst:
            continue
        for rank, (doc, _) in enumerate(lst, start=1):
            key = _doc_key(doc)
            if not key:
                continue
            inc = float(weight) * (1.0 / (k + rank))
            if intent_boost_fn is not None:
                try:
                    inc *= float(intent_boost_fn(doc))
                except Exception:
                    pass
            if key not in rrf_scores:
                rrf_scores[key] = (doc, inc)
            else:
                prev_doc, prev_s = rrf_scores[key]
                rrf_scores[key] = (prev_doc, prev_s + inc)
    # 转为“距离型”分数：名次越靠前 score 越小（与 cosine distance 排序约定一致）
    ranked = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    return [(d, float(i) * 1e-4) for i, (d, _) in enumerate(ranked)]


_TABLE_LINE_PAT = re.compile(
    r"^\s*((table|tab\.?)\s*\d+|表\s*[\d一二三四五六七八九十]+)",
    re.I | re.M,
)


def looks_like_table_evidence(doc: object) -> bool:
    """判断 chunk 是否主要为「论文表格/表头」而非方法叙述（用于 paper_method 降权）。"""
    meta = getattr(doc, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    role = str(meta.get("chunk_role") or "").lower()
    typ = str(meta.get("type") or "").lower()
    if "table" in role or typ in ("table", "table_summary"):
        return True
    if bool(meta.get("has_table")) and "method" not in role:
        return True
    body = str(getattr(doc, "page_content", "") or "")[:4000]
    if not body.strip():
        return False
    head = body[:800]
    if _TABLE_LINE_PAT.search(head):
        return True
    lines = [ln for ln in body.splitlines()[:20] if ln.strip()]
    if len(lines) >= 4:
        tab_lines = sum(1 for ln in lines[:15] if ln.count("\t") >= 2)
        pipe_lines = sum(1 for ln in lines[:15] if ln.count("|") >= 3)
        if tab_lines >= 3 or pipe_lines >= 3:
            return True
    return False


def metadata_intent_boost(
    doc: object,
    *,
    wants_table: bool,
    wants_figure: bool,
    preferred_roles: list[str],
    intent: str = "",
) -> float:
    """按 metadata 对齐查询意图的轻量加权。"""
    meta = getattr(doc, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    role = str(meta.get("chunk_role") or meta.get("type") or "").lower()
    boost = 1.0
    if wants_table and ("table" in role or meta.get("has_table")):
        boost *= 1.35
    if wants_figure and ("figure" in role or "pdf_figure" in role or meta.get("has_figure")):
        boost *= 1.35
    if preferred_roles:
        for pr in preferred_roles:
            if pr.lower() in role:
                boost *= 1.15
                break
    # arXiv 摘要条略抬升（高层概述）
    if meta.get("type") == "arxiv_abstract" or meta.get("source") == "arxiv_metadata":
        boost *= 1.08
    # 方法类问题：压低表格块（RRF 增量乘 penalty → 等价于降权）
    if (
        RAG_PAPER_METHOD_DOWNRANK_TABLE_CHUNKS
        and (intent or "").strip() == "paper_method"
        and not wants_table
        and looks_like_table_evidence(doc)
    ):
        boost *= RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY
    return boost


def penalize_table_chunks_for_method_intent(
    pairs: List[Tuple[object, float]],
    *,
    intent: str,
    wants_table: bool,
    scores_higher_is_better: bool,
) -> List[Tuple[object, float]]:
    """精排分数：越高越好时用乘法惩罚；粗排分数：越小越好时加上偏移。"""
    if (
        not RAG_PAPER_METHOD_DOWNRANK_TABLE_CHUNKS
        or (intent or "").strip() != "paper_method"
        or wants_table
        or not pairs
    ):
        return pairs
    p = float(RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY)
    out: List[Tuple[object, float]] = []
    for doc, sc in pairs:
        s = float(sc)
        if looks_like_table_evidence(doc):
            if scores_higher_is_better:
                s *= p
            else:
                s += 0.002
        out.append((doc, s))
    if scores_higher_is_better:
        out.sort(key=lambda x: x[1], reverse=True)
    else:
        out.sort(key=lambda x: x[1])
    return out
