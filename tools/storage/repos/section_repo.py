from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import Paper, PaperSection


_SEC_TITLE_KEY_PAT = re.compile(
    r"(?i)\b(abstract|introduction|related work|background|preliminaries|"
    r"method(?:s|ology)?|approach|architecture|model|"
    r"experiment(?:s)?|experimental|evaluation|result(?:s)?|ablation|"
    r"discussion|conclusion(?:s)?|future work|limitations|"
    r"摘要|引言|方法|实验|结果|结论)\b"
)
_SEC_PREFIX_PAT = re.compile(r"^\s*((\d+(\.\d+)*)|([IVXLCDM]+))[\)\.\s]+", re.I)


def looks_like_section_heading(title: str) -> bool:
    """
    判断字符串是否像「章节标题」而不是正文句子。
    用于避免把正文句子（含 method/experiment 等词）误判成 section role。
    """
    t = (title or "").strip()
    if not t:
        return False
    if len(t) > 140:
        return False
    has_prefix = bool(_SEC_PREFIX_PAT.match(t))
    # 句子末尾常见标点：正文概率高（含英文句号）
    if re.search(r"[。！？!?;；\.]\s*$", t):
        return False
    # 无章节编号前缀时，若中间出现句点，通常是正文句子而非标题（如 "xxx. For yyy"）
    if (not has_prefix) and ("." in t):
        return False
    # 逗号过多通常是正文句子
    if t.count(",") + t.count("，") >= 2:
        return False
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) > 18:
        return False
    key_m = _SEC_TITLE_KEY_PAT.search(t)
    # 无关键词就不是目标章节
    if not key_m:
        return False
    # 关键词离行首太远，且没有编号前缀，通常是正文句子
    if (not has_prefix) and int(key_m.start()) > 26:
        return False
    # 小写起始且无编号，且较长，通常是正文句子
    if (not has_prefix) and t and t[0].islower() and len(t) > 48:
        return False
    return True


def get_paper_by_arxiv_id(arxiv_id: str) -> dict[str, Any] | None:
    aid = (arxiv_id or "").strip()
    if not aid:
        return None
    factory = get_session_factory()
    if factory is None:
        return None
    session = factory()
    try:
        p = session.scalar(select(Paper).where(Paper.arxiv_id == aid))
        if p is None:
            return None
        return {"id": int(p.id), "arxiv_id": p.arxiv_id, "title": p.title}
    finally:
        session.close()


def list_sections_for_paper(paper_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        stmt = (
            select(PaperSection)
            .where(PaperSection.paper_id == int(paper_id))
            .order_by(PaperSection.order_index.asc())
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": int(r.id),
            "paper_id": int(r.paper_id),
            "title": r.title,
            "title_norm": r.title_norm,
            "section_number": r.section_number,
            "section_level": int(r.section_level),
            "order_index": int(r.order_index),
            "page_start": r.page_start,
            "page_end": r.page_end,
        }
        for r in rows
    ]


def infer_section_role(title: str) -> str:
    """
    基于 section 标题/归一化标题推断章节角色，用于 `section_ids_by_role`。

    说明：原实现只覆盖了 method/result/introduction 的少量关键词，未命中
    "Methodology/Experiments/Experimental Setup/Related Work/Background" 等常见英文写法，
    会导致 paper_retriever 在 `paper_method` 场景下拿不到 method section（method_hits=0）。
    """
    t = (title or "").strip().lower()
    if not t:
        return "other"
    if not looks_like_section_heading(t):
        return "other"

    # method：method / methodology / approach 等；也把实验设置类内容映射到 method，
    # 因为用户问“方法部分”通常会把实验设计/实现细节也算作方法。
    if re.search(
        r"(\bmethod(?:ology)?\b|approach|architecture|model|pipeline|framework|algorithm|objective|loss|optimization|training|implementation|implementation details|materials and methods|experimental setup|experimental design|inference|procedure|方法)",
        t,
        re.I,
    ):
        return "method"

    # result：result(s) / experiment(s) / evaluation / ablation / metrics / performance
    if re.search(
        r"(\bresult(?:s)?\b|experiment(?:s)?|evaluation|ablation|metrics|metric|performance|comparison|analysis|实验|结果)",
        t,
        re.I,
    ):
        return "result"

    # conclusion：conclusion / discussion / limitations 等
    if re.search(
        r"(conclusion|discussion|limitations|future work|总结|结论|局限)",
        t,
        re.I,
    ):
        return "conclusion"

    if re.search(r"(abstract|摘要)", t, re.I):
        return "abstract"

    if re.search(
        r"(introduction|background|overview|preliminaries|related work|literature review|prior work|引言)",
        t,
        re.I,
    ):
        return "introduction"

    return "other"


def section_ids_by_role(paper_id: int, role: str) -> list[int]:
    role_norm = (role or "").strip().lower()
    out: list[int] = []
    for sec in list_sections_for_paper(paper_id):
        # 优先使用 title_norm（通常包含更稳定的英文大小写/空白归一化），其次 fallback 到 title。
        title_norm = str(sec.get("title_norm") or "").strip()
        title = str(sec.get("title") or "").strip()
        probe = title_norm if title_norm else title
        if infer_section_role(probe) == role_norm:
            out.append(int(sec["id"]))
    return out

