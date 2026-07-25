from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from tools.storage.paper_library import list_papers


def _norm_text(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s\-_:./]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class PaperEntityMatchResult:
    confidence: float = 0.0
    match_mode: str = "none"
    paper_ids: list[int] = field(default_factory=list)
    paper_titles: list[str] = field(default_factory=list)
    arxiv_ids: list[str] = field(default_factory=list)


def match_local_paper_entities(question: str, limit: int = 3) -> PaperEntityMatchResult:
    """本地论文标题/ID 近似匹配（轻量规则版）。"""
    q = _norm_text(question)
    try:
        from tools.retrieval.local_paper_matcher import _clean_question_for_title, _norm as lm_norm

        qc = lm_norm(_clean_question_for_title(question))
        if len(qc) >= 4:
            q = qc
    except Exception:
        pass
    if not q:
        return PaperEntityMatchResult()

    papers = list_papers() or []
    if not papers:
        return PaperEntityMatchResult()

    scored: list[tuple[float, dict]] = []
    for p in papers:
        title = _norm_text(str(p.get("title") or ""))
        if not title:
            continue
        aid = str(p.get("arxiv_id") or "").strip()

        ratio = SequenceMatcher(None, q, title).ratio()
        if title in q:
            ratio = max(ratio, 0.88)
        if aid and aid.lower() in q:
            ratio = max(ratio, 0.98)
        q_words = set(q.split())
        t_words = set(title.split())
        if q_words and t_words:
            jac = len(q_words & t_words) / max(1, len(q_words | t_words))
            ratio = max(ratio, jac * 0.95)
        scored.append((float(ratio), p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(s, p) for s, p in scored[: max(1, limit)] if s >= 0.45]
    if not top:
        return PaperEntityMatchResult()

    best = top[0][0]
    mode = "local_title_match"
    if best >= 0.9:
        mode = "local_title_exact_or_id"
    elif best < 0.65:
        mode = "local_title_fuzzy"

    out = PaperEntityMatchResult(confidence=float(best), match_mode=mode)
    for _, p in top:
        t = str(p.get("title") or "").strip()
        a = str(p.get("arxiv_id") or "").strip()
        if t and t not in out.paper_titles:
            out.paper_titles.append(t)
        if a and a not in out.arxiv_ids:
            out.arxiv_ids.append(a)
    return out

