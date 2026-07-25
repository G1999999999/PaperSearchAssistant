from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from tools.storage.papers_db import list_papers as db_list_papers


@dataclass
class LocalPaperMatch:
    matched: bool = False
    score: float = 0.0
    paper: dict[str, Any] | None = None
    reason: str = "none"


def _norm(s: str) -> str:
    x = (s or "").lower().strip()
    x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def _clean_question_for_title(question: str) -> str:
    q = str(question or "")
    q = q.replace("《", " ").replace("》", " ").replace("\"", " ").replace("'", " ")
    # 常见问句噪声，避免拉低标题匹配分数
    noise = [
        "这篇论文讲了什么",
        "这篇论文主要讲了什么",
        "这篇论文说了什么",
        "讲了什么",
        "主要贡献是什么",
        "总结一下",
        "介绍一下",
        "解释一下",
        "这篇论文",
        "从本地库里面检索",
        "从本地库检索",
        "从本地库里检索",
        "从本地库",
        "本地库里面",
        "本地库",
        "说一说",
        "方法部分",
        "实验部分",
        "结论部分",
        "论文",
        "paper",
    ]
    for n in noise:
        q = re.sub(re.escape(n), " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def match_local_paper(question: str, limit: int = 80) -> LocalPaperMatch:
    q = _norm(_clean_question_for_title(question))
    if not q or len(q) < 3:
        return LocalPaperMatch()
    rows = db_list_papers(limit=max(10, min(200, int(limit))), offset=0)
    if not rows:
        return LocalPaperMatch()

    best_row: dict[str, Any] | None = None
    best_score = 0.0
    best_reason = "none"

    # Try arXiv ID direct
    m = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", question or "", re.IGNORECASE)
    aid = re.sub(r"v\d+$", "", m.group(0), flags=re.IGNORECASE) if m else ""
    if aid:
        for r in rows:
            if str(r.get("arxiv_id") or "").strip().lower() == aid.lower():
                return LocalPaperMatch(matched=True, score=1.0, paper=r, reason="arxiv_id_exact")

    for r in rows:
        title = _norm(str(r.get("title") or ""))
        if not title:
            continue
        if q in title or title in q:
            s = min(len(q), len(title)) / max(len(q), len(title))
            if s > best_score:
                best_score = s
                best_row = r
                best_reason = "title_contains"
            continue
        # 标题相似 + token 重叠混合分数
        s_fuzzy = SequenceMatcher(None, q, title).ratio()
        q_tokens = set(t for t in q.split(" ") if t)
        t_tokens = set(t for t in title.split(" ") if t)
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens) or 1
        s_overlap = inter / union
        s = max(s_fuzzy, s_overlap * 0.92)
        if s > best_score:
            best_score = s
            best_row = r
            best_reason = "title_fuzzy"
    if best_row and best_score >= 0.56:
        return LocalPaperMatch(matched=True, score=float(best_score), paper=best_row, reason=best_reason)
    return LocalPaperMatch()

