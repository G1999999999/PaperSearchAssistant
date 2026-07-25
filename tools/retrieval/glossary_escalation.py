"""论文问答场景下：术语/定义类问题且检索上下文未命中时，触发受控联网补充（门控逻辑）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_OVERVIEW_PAT = re.compile(
    r"(这篇论文讲了什么|论文(?:主要)?内容|整体讲了|全文概述|概括这篇|总结这篇论文|"
    r"论文的梗概|主要讲什么|讲了啥)",
    re.I,
)
_ABBR_PAT = re.compile(r"[（(]\s*([A-Za-z][A-Za-z0-9\-]{0,14})\s*[）)]")
_TERM_DEF_PAT = re.compile(
    r"(?:是什么|何谓|什么叫|指的是什么|如何理解|的定义|含义(?:是|为)?|"
    r"全称是|英文全称|缩写[^\n]{0,8}含义|stand\s+for|stands\s+for)",
    re.I,
)


def user_forbids_web_in_question(question: str) -> bool:
    qn = (question or "").lower()
    return any(
        w in qn
        for w in (
            "不要联网",
            "不联网",
            "仅依据论文",
            "只根据论文",
            "别看网上",
            "禁止联网",
        )
    )


def looks_like_term_definition_question(question: str) -> bool:
    qn = (question or "").strip()
    if not qn or _OVERVIEW_PAT.search(qn):
        return False
    if _TERM_DEF_PAT.search(qn):
        return True
    if _ABBR_PAT.search(qn) and re.search(r"(是什么|指什么|含义|缩写)", qn):
        return True
    return False


def _merged_context_casefold(grouped: list) -> str:
    parts: list[str] = []
    for item in grouped or []:
        doc = item[0] if isinstance(item, (list, tuple)) and item else item
        txt = getattr(doc, "page_content", str(doc or "")) or ""
        parts.append(txt)
    return "\n".join(parts).casefold()


def _term_absent_in_blob(term: str, blob: str) -> bool:
    tl = (term or "").casefold().strip()
    if not tl:
        return False
    if len(tl) <= 3 and tl.isascii() and tl.isalpha():
        return re.search(rf"(?<![a-z0-9]){re.escape(tl)}(?![a-z0-9])", blob) is None
    return tl not in blob


def extract_glossary_terms(question: str) -> list[str]:
    q = (question or "").strip()
    out: list[str] = []
    for m in _ABBR_PAT.finditer(q):
        t = (m.group(1) or "").strip()
        if 1 <= len(t) <= 20:
            out.append(t)
    m = re.search(r"(.{2,40})是什么", q)
    if m:
        frag = m.group(1).strip()
        frag = re.sub(r"^[（）()\s]+|[（）()\s]+$", "", frag)
        if len(frag) >= 2 and not re.match(
            r"^(这篇|论文|它|作者|其中|这里|这个|这样|那句话)",
            frag,
        ):
            out.append(frag)
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        k = t.casefold()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


@dataclass
class GlossaryWebDecision:
    need_supplement: bool
    reason: str


def paper_glossary_should_supplement_web(question: str, grouped: list) -> GlossaryWebDecision:
    if user_forbids_web_in_question(question):
        return GlossaryWebDecision(False, "user_forbids_web")
    if not looks_like_term_definition_question(question):
        return GlossaryWebDecision(False, "not_term_definition_shape")
    terms = extract_glossary_terms(question)
    if not terms:
        return GlossaryWebDecision(False, "no_extractable_term")
    blob = _merged_context_casefold(grouped)
    missing = [t for t in terms if _term_absent_in_blob(t, blob)]
    if not missing:
        return GlossaryWebDecision(False, "term_in_context")
    return GlossaryWebDecision(
        True,
        "term_missing_in_context:" + ",".join(missing[:6]),
    )
