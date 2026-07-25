"""仅本地论文库的「列相关论文」检索与排序（不联网）。"""

from __future__ import annotations

import re
from typing import Any

from tools.retrieval.local_paper_service import search_local_papers


def extract_local_library_topic(question: str) -> str:
    """从「列出本地…和 XX 相关论文」类问句抽取主题词。"""
    q = (question or "").strip()
    if not q:
        return ""
    # 去掉列表任务常见套话
    strip_pats = [
        r"帮我",
        r"请",
        r"列出",
        r"罗列",
        r"看看",
        r"查询",
        r"检索",
        r"搜索",
        r"搜一下",
        r"只查本地",
        r"仅从本地",
        r"仅在本地",
        r"不要联网",
        r"本地库(?:里|中)?",
        r"我本地库",
        r"已下载的?",
        r"和",
        r"与",
        r"相关的?",
        r"方向的?",
        r"主题",
        r"有关",
        r"关于",
        r"论文",
        r"文献",
        r"paper",
        r"papers",
        r"有哪些",
        r"都有什么",
        r"全部",
        r"所有",
    ]
    out = q
    for p in strip_pats:
        out = re.sub(p, " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip(" ，,。；;")
    return out if len(out) >= 1 else q


def _score_row(topic: str, row: dict[str, Any]) -> float:
    tlow = (topic or "").strip().lower()
    title = str(row.get("title") or "")
    summ = str(row.get("summary") or "")
    if not tlow:
        return 0.0
    tl, sl = title.lower(), summ.lower()
    score = 0.0
    if tlow in tl:
        score += 3.0
    if tlow in sl:
        score += 1.2
    for w in tlow.split():
        if len(w) < 2:
            continue
        if w in tl:
            score += 0.75
        if w in sl:
            score += 0.35
    # 缩写 / 短语命中（如 3DGS）
    if len(tlow) <= 12 and tlow.isalnum() and tlow in tl.replace(" ", ""):
        score += 0.5
    return score


def search_local_library_ranked(topic: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """多关键词召回后按标题/摘要相关性混排。"""
    topic = (topic or "").strip()
    keys: list[str] = []
    if topic:
        keys.append(topic)
        # 空格拆词补召回
        parts = [p for p in re.split(r"[\s,，]+", topic) if len(p) >= 2]
        for p in parts[:5]:
            if p not in keys:
                keys.append(p)
    raw = search_local_papers(keys or [""], limit=max(int(limit) * 3, 24))
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for r in raw:
        aid = str(r.get("arxiv_id") or "").strip().lower()
        if not aid or aid in seen:
            continue
        seen.add(aid)
        scored.append((_score_row(topic, r), dict(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[: max(1, int(limit))]]


def format_local_library_answer_rows(
    rows: list[dict[str, Any]], *, preamble: str | None = None
) -> str:
    lines: list[str] = []
    if preamble:
        lines.append(preamble)
        lines.append("")
    if not rows:
        lines.append("本地论文库中暂无与当前描述匹配的论文。")
        lines.append("如需联网补充，可以说：请联网搜索 …")
        return "\n".join(lines)
    lines.append("【本地库】")
    for i, r in enumerate(rows, start=1):
        title = (r.get("title") or "").strip()
        authors = ", ".join((r.get("authors") or [])[:3])
        pubs = (r.get("published") or "").strip()
        aid = str(r.get("arxiv_id") or "").strip()
        pdf = (r.get("pdf_path") or "").strip()
        lines.append(f"{i}. {title}")
        if authors:
            lines.append(f"   作者: {authors}")
        if pubs:
            lines.append(f"   时间: {pubs}")
        if aid:
            lines.append(f"   arXiv: {aid}")
        if pdf:
            lines.append(f"   本地PDF: {pdf}")
        lines.append("")
    lines.append(
        "可回复「读第 N 篇」或「说一说第 N 篇论文的方法」继续阅读；"
        "若需只在此范围内检索，可说「只查本地」。"
    )
    return "\n".join(lines).rstrip()
