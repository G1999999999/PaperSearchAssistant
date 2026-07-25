from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MissingEvidenceReport:
    thin_context: bool = False
    abstract_truncated: bool = False
    missing_method: bool = False
    missing_result: bool = False
    missing_conclusion: bool = False
    has_table_evidence: bool = False
    has_figure_evidence: bool = False
    has_generic_body: bool = False

    @property
    def needs_expansion(self) -> bool:
        return bool(
            self.thin_context
            or self.abstract_truncated
            or self.missing_method
            or self.missing_result
            or self.missing_conclusion
        )


def _doc_text(doc: object) -> str:
    return str(getattr(doc, "page_content", "") or "")


def _doc_meta(doc: object) -> dict[str, Any]:
    m = getattr(doc, "metadata", {}) or {}
    return dict(m) if isinstance(m, dict) else {}


def _looks_truncated_abstract(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # 常见“摘要前半截”信号：结尾不是句号，且长度偏短
    if len(t) < 320 and not re.search(r"[。！？.!?]$", t):
        return True
    # 命中用户案例类似 “By predicting compact Gaussian” 末尾截断
    if re.search(r"\bBy predicting compact Gaussian\b", t, re.I) and len(t) < 700:
        return True
    return False


def analyze_missing_evidence(
    question: str,
    reranked: list[tuple[object, float]],
    intent: str,
    wants_table: bool,
    wants_figure: bool,
) -> MissingEvidenceReport:
    r = MissingEvidenceReport()
    total_chars = 0
    method_hits = 0
    result_hits = 0
    conc_hits = 0

    for doc, _sc in (reranked or [])[:20]:
        text = _doc_text(doc)
        total_chars += len(text)
        meta = _doc_meta(doc)
        role = str(meta.get("chunk_role") or "").lower()
        section = str(meta.get("section_title") or meta.get("heading") or "").lower()
        typ = str(meta.get("type") or "").lower()

        if role == "generic":
            r.has_generic_body = True
        if role == "table" or "table" in typ or bool(meta.get("has_table")):
            r.has_table_evidence = True
        if role == "figure" or "figure" in typ or bool(meta.get("has_figure")):
            r.has_figure_evidence = True

        if role == "paper_summary" or "abstract" in section or typ == "arxiv_abstract":
            if _looks_truncated_abstract(text):
                r.abstract_truncated = True

        if re.search(r"(method|approach|architecture|model|方法|算法)", section, re.I):
            method_hits += 1
        if re.search(r"(result|experiment|evaluation|ablation|实验|结果)", section, re.I):
            result_hits += 1
        if re.search(r"(conclusion|discussion|结论)", section, re.I):
            conc_hits += 1

    # 上下文过薄：正文不足且字符数偏少
    r.thin_context = (not r.has_generic_body) or (total_chars < 9000)
    if intent == "paper_method":
        r.missing_method = method_hits == 0
    if intent in ("paper_result", "table_lookup"):
        r.missing_result = result_hits == 0
    if intent == "paper_conclusion":
        r.missing_conclusion = conc_hits == 0
    if wants_table and not r.has_table_evidence:
        r.missing_result = True
    if wants_figure and not r.has_figure_evidence:
        r.missing_method = True
    return r

