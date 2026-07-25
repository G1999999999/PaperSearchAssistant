from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PaperQueryUnderstanding:
    """规则型查询理解；后续可接轻量 LLM router。"""

    intent: str = "open_ended_analysis"
    section_keywords: list[str] = field(default_factory=list)
    wants_table: bool = False
    wants_figure: bool = False
    extra_retrieval_queries: list[str] = field(default_factory=list)

    # 与 Chroma metadata / 后续 PG chunk_role 对齐的提示角色
    preferred_roles: list[str] = field(default_factory=list)


_TABLE_PAT = re.compile(
    r"(table|表\s*\d+|表\s*格|指标|ablation|accuracy|f1|bleu|em\s|实验结果|结果如何)",
    re.I,
)
_FIG_PAT = re.compile(
    r"(figure|fig\.?|图\s*\d+|架构图|流程图|示意图)",
    re.I,
)
_METHOD_PAT = re.compile(r"(方法|model|architecture|approach|算法|提出)", re.I)
_RESULT_PAT = re.compile(r"(结果|性能|对比|提升|实验|evaluation)", re.I)
_EXP_SETUP_PAT = re.compile(r"(实验.*怎么做|怎么做.*实验|实验设置|实验方案|evaluation setup|experimental setup)", re.I)
_SUMMARY_PAT = re.compile(r"(摘要|abstract|总结|贡献|主要工作|这篇论文)", re.I)
_CONC_PAT = re.compile(r"(结论|conclusion|局限)", re.I)
_COMPARE_PAT = re.compile(r"(区别|比较|vs\.?|相较|different)", re.I)


def analyze_paper_query(question: str) -> PaperQueryUnderstanding:
    q = (question or "").strip()
    low = q.lower()
    out = PaperQueryUnderstanding()

    if _COMPARE_PAT.search(q):
        out.intent = "paper_comparison"
        out.preferred_roles.extend(["method", "result", "experiment", "conclusion"])
    if _TABLE_PAT.search(q):
        out.wants_table = True
        out.intent = "table_lookup"
        out.preferred_roles.extend(["table", "result", "experiment"])
    if _FIG_PAT.search(q):
        out.wants_figure = True
        if out.intent == "open_ended_analysis":
            out.intent = "figure_lookup"
        out.preferred_roles.extend(["figure", "method"])

    if out.intent == "open_ended_analysis":
        # 优先级：方法/实验/结论 > 总览（summary）
        # 避免“这篇论文的方法是什么”因含“这篇论文”被误判为 summary。
        has_method = bool(_METHOD_PAT.search(q))
        has_result = bool(_RESULT_PAT.search(q))
        asks_exp_setup = bool(_EXP_SETUP_PAT.search(q))
        # 像“实验是怎么做的/实验设置”这类问法，优先归到 paper_result，避免被“怎么做”误路由到 method。
        if asks_exp_setup or (has_result and not has_method):
            out.intent = "paper_result"
            out.section_keywords.extend(["result", "experiment", "evaluation"])
            out.preferred_roles.extend(["result", "experiment"])
        elif has_method:
            out.intent = "paper_method"
            out.section_keywords.extend(["method", "approach", "model", "architecture"])
            out.preferred_roles.extend(["method", "introduction"])
        elif _CONC_PAT.search(q):
            out.intent = "paper_conclusion"
            out.preferred_roles.extend(["conclusion"])
        elif _SUMMARY_PAT.search(q):
            out.intent = "paper_summary"
            out.preferred_roles.extend(["paper_summary", "abstract", "introduction"])

    # 多路检索用英文短语（不调用 LLM，成本低）
    if out.intent == "paper_method":
        out.extra_retrieval_queries.extend(
            [
                "method architecture proposed approach",
                "model design contributions",
                "Section 3 methodology implementation pipeline",
            ]
        )
    elif out.intent in ("paper_result", "table_lookup"):
        out.extra_retrieval_queries.extend(
            ["experimental results metrics evaluation", "performance comparison benchmark"]
        )
    elif out.intent == "paper_summary":
        out.extra_retrieval_queries.extend(
            ["paper title abstract contributions summary", "main contribution overview"]
        )
    elif out.intent == "figure_lookup":
        out.extra_retrieval_queries.extend(["figure caption illustration visualization"])

    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for r in out.preferred_roles:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    out.preferred_roles = uniq

    eq: list[str] = []
    seenq: set[str] = set()
    for x in out.extra_retrieval_queries:
        t = x.strip()
        if t and t not in seenq:
            seenq.add(t)
            eq.append(t)
    out.extra_retrieval_queries = eq
    return out


def subquestions_for_decomposition(question: str) -> list[str]:
    """极简复合问句拆分（按中文/英文分号与问号）。"""
    q = (question or "").strip()
    if not q:
        return []
    parts = re.split(r"[；;]+|(?<=[\?？])\s*", q)
    out = [p.strip() for p in parts if p and len(p.strip()) > 3]
    return out[:5] if len(out) > 1 else []
