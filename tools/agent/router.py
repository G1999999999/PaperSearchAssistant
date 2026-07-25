"""
路由器（兼容旧接口 + 新结构化输出）：
- 旧接口：`route_query` / `is_paper_intent`
- 新接口：`route_query_structured`，输出高级路由字段供检索层使用
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Route(str, Enum):
    RAG = "rag"
    ARXIV = "arxiv"
    WEATHER = "weather"


@dataclass
class StructuredRouteResult:
    intent: str = "local_rag"
    sub_intent: str = "open_ended_analysis"
    source_preference: str = "local_first"
    paper_match_mode: str = "none"
    paper_ids: list[int] = field(default_factory=list)
    paper_titles: list[str] = field(default_factory=list)
    section_hint: str = ""
    needs_table: bool = False
    needs_figure: bool = False
    needs_web: bool = False
    confidence: float = 0.5
    route: Route = Route.RAG

    def to_dict(self) -> dict:
        d = asdict(self)
        d["route"] = str(self.route)
        return d


_ARXIV_NEW_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r"\b[a-z\-]+\/\d{7}(v\d+)?\b", re.IGNORECASE)
# 仅匹配「整份 basename」为新格式 arXiv PDF，避免误识别普通 PDF 文件名中的片段
_ARXIV_STYLE_PDF_BASENAME = re.compile(
    r"^\d{4}\.\d{4,5}(v\d+)?\.pdf$",
    re.IGNORECASE,
)


def arxiv_id_from_arxiv_style_pdf_basename(name: str) -> str | None:
    """若文件名为 arXiv 新格式 PDF（如 ``1706.03762.pdf``、``2112.10752v2.pdf``），返回去掉版本后缀的 ID。

    使用 ``Path(name).name`` 防止路径片段干扰。
    """
    base = Path(name or "").name
    if not base or not _ARXIV_STYLE_PDF_BASENAME.match(base):
        return None
    stem = Path(base).stem
    from tools.agent.agent_tools import _normalize_arxiv_id

    aid = _normalize_arxiv_id(stem)
    return aid or None


def extract_arxiv_id(text: str) -> str | None:
    """从文本中提取 arXiv ID（支持新/旧格式）。"""
    if not text:
        return None
    m = _ARXIV_NEW_RE.search(text)
    if m:
        return m.group(0)
    m = _ARXIV_OLD_RE.search(text)
    if m:
        return m.group(0)
    return None


def paper_namespace_arxiv_id(namespace: str) -> str | None:
    """从 ``paper:<arxiv_id>:...`` 解析 arXiv id；非论文分区返回 None。"""
    ns = (namespace or "").strip()
    if not ns.startswith("paper:"):
        return None
    try:
        part = ns.split(":", 2)[1]
        aid = part.split(":", 1)[0].strip()
        return aid or None
    except (IndexError, ValueError):
        return None


def find_all_arxiv_ids(text: str) -> list[str]:
    """从文本中提取全部 arXiv ID（去重保序，去掉版本后缀 vN）。"""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for rx in (_ARXIV_NEW_RE, _ARXIV_OLD_RE):
        for m in rx.finditer(text):
            raw = m.group(0)
            aid = re.sub(r"v\d+$", "", raw, flags=re.IGNORECASE).strip()
            if aid and aid not in seen:
                seen.add(aid)
                out.append(aid)
    return out


def is_paper_intent(text: str) -> bool:
    """启发式判断：用户是否在询问论文/文献信息/arXiv。"""
    return route_query_structured(text).intent in (
        "paper_search",
        "paper_qa",
        "local_paper_search",
    )


def _is_weather_intent(q: str) -> bool:
    return any(k in q for k in ["天气", "weather", "温度", "cold", "hot", "humidity", "湿度", "wind"])


def _paper_sub_intent(q: str) -> str:
    if re.search(r"(table|表\s*\d+|指标|f1|miou|accuracy|bleu|em)", q, re.I):
        return "table_lookup"
    if re.search(r"(figure|fig\.?|图\s*\d+|结构图|流程图)", q, re.I):
        return "figure_lookup"
    if re.search(r"(方法|method|approach|architecture|模型)", q, re.I):
        return "method"
    if re.search(r"(实验|结果|result|evaluation|ablation|对比)", q, re.I):
        return "experiment"
    if re.search(r"(结论|conclusion|局限)", q, re.I):
        return "conclusion"
    if re.search(r"(比较|区别|vs|对比)", q, re.I):
        return "comparison"
    if re.search(r"(摘要|总结|贡献|讲了什么|是什么)", q, re.I):
        return "summary"
    return "open_ended_analysis"


def _section_hint_from_sub_intent(sub_intent: str) -> str:
    mp = {
        "method": "method",
        "experiment": "experiment",
        "conclusion": "conclusion",
        "summary": "abstract",
        "table_lookup": "result",
        "figure_lookup": "method",
        "comparison": "result",
    }
    return mp.get(sub_intent, "")


def _local_library_paper_search_intent(q: str) -> bool:
    """仅列本地库相关论文，默认不联网（local_paper_search）。"""
    qn = (q or "").strip()
    if not qn:
        return False
    from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

    triggers = (
        "列出本地",
        "本地库里",
        "本地库中",
        "本地库和",
        "本地有哪些",
        "看看本地库",
        "我本地库",
        "只查本地",
        "仅在本地",
        "本地的论文有哪些",
        "本地论文有哪些",
        "搜本地库",
        "查本地库",
        "本地库有哪些",
        "本地库里有哪些",
    )
    hit_trigger = any(t in qn for t in triggers)
    if not hit_trigger and re.search(r"只查本地.{0,16}(列出|有哪些|罗列)", qn):
        hit_trigger = True
    # 「本地库…关于…论文」不依赖「看看」等前缀
    if not hit_trigger and re.search(r"本地库.{0,24}关于", qn) and re.search(
        r"(论文|paper|文献)", qn, re.I
    ):
        hit_trigger = True
    if not hit_trigger:
        return False
    # 「从本地库…说一说某篇」属于 paper_qa，除非同时带「列出/有哪些」等列表用语
    if looks_like_paper_content_qa(qn) and not re.search(
        r"(列出|有哪些|罗列|搜出|检索出).{0,30}(论文|paper|文献)", qn, re.I
    ):
        return False
    if "本地有哪些" in qn and re.search(r"(论文|paper|文献)", qn, re.I):
        return True
    if re.search(
        r"(列出|有哪些|罗列|看看|查询|搜|查).{0,50}关于.{0,60}(论文|paper|文献)",
        qn,
        re.I,
    ) and re.search(r"(本地|已下载|入库|我的库)", qn):
        return True
    return bool(
        re.search(r"(列出|有哪些|罗列|看看).{0,40}(论文|paper|文献)", qn, re.I)
    ) or bool(re.search(r"(相关|方向|主题).{0,16}(论文|paper)", qn, re.I))


def _paper_search_intent(q: str) -> bool:
    """找/推荐多篇论文（可联网补充）。不含「从本地库说一说某篇」类 paper_qa。"""
    qn = (q or "").strip()
    if not qn:
        return False
    if _local_library_paper_search_intent(qn):
        return False
    from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

    if looks_like_paper_content_qa(qn):
        return False
    return bool(
        re.search(
            r"(帮我找|帮我查找|帮我检索|帮我搜索|查找一下|检索一下|搜索一下|找一下|找几篇|"
            r"推荐几篇|推荐.*论文|搜一下.*论文|搜.*论文|搜索.*论文|检索.*论文文献|最近.*论文|最新.*论文|"
            r"查找.*(论文|paper|文献|arxiv|work)|搜索.*(论文|paper|文献|work)|"
            r"检索.*(论文|paper|文献)(?![^。，]{0,12}(方法|实验|结论|内容|讲讲|介绍))|"
            r".*方向.*(有哪些|有什么).*(论文|work)|search.*paper|find.*paper|"
            r"please recommend.*paper|请联网找)",
            qn,
            re.I,
        )
    )


def paper_discovery_forbids_web_only(q: str) -> bool:
    """用户明确要求只查本地、不要联网（用于 paper_search 仍走本地列表但禁 arXiv）。"""
    qn = (q or "").lower()
    return any(
        x in qn
        for x in (
            "只查本地",
            "仅本地",
            "不要联网",
            "不联网",
            "仅从本地",
            "只要本地",
            "别联网",
            "仅列出本地",
        )
    )


def paper_discovery_prefers_web(q: str) -> bool:
    """「最新/最近」类发现需求：应允许联网（arXiv 按时间排序等）。"""
    qn = (q or "").lower()
    return any(
        x in qn
        for x in (
            "最新",
            "最近",
            "近期",
            "latest",
            "recent",
            "新发表",
            "新近",
        )
    )


def _paper_read_intent(q: str) -> bool:
    return bool(
        re.search(
            r"(读第\s*\d+\s*篇|读这篇|打开这篇|看这篇|阅读这篇|帮我看看.*论文|open.*paper|read.*paper)",
            q,
            re.I,
        )
    )


def open_ended_question_prefers_web_supplement(q: str) -> bool:
    """非文献检索的开放问句：倾向联网摘要（供 tools 模式子问题预取等使用）。"""
    qn = (q or "").strip()
    if not qn:
        return False
    if paper_discovery_forbids_web_only(qn):
        return False
    if any(
        x in qn
        for x in (
            "不要联网",
            "不联网",
            "仅依据库",
            "只要本地",
            "仅限本地",
            "根据上文",
            "基于提供的上下文",
            "基于检索片段",
        )
    ):
        return False
    if user_requests_forced_web_search(qn):
        return True
    if re.search(
        r"(区别|差异|不同点|对比一下|对比|比较|哪个更好|孰优孰劣|versus|\bvs\.?\b|相较于|优缺点)",
        qn,
        re.I,
    ):
        return True
    if re.search(r"(和|与|跟).{0,48}(区别|差异|不同|对比|比较好|哪家强)", qn, re.I):
        return True
    return False


def route_query_structured(question: str) -> StructuredRouteResult:
    q = (question or "").strip()
    ql = q.lower()
    sub = _paper_sub_intent(q)
    if _is_weather_intent(ql):
        return StructuredRouteResult(
            intent="tool_task",
            sub_intent="weather",
            source_preference="tool_first",
            route=Route.WEATHER,
            confidence=0.98,
        )

    if _local_library_paper_search_intent(q):
        return StructuredRouteResult(
            intent="local_paper_search",
            sub_intent="list_local_catalog",
            source_preference="local_only",
            needs_web=False,
            confidence=0.96,
            route=Route.RAG,
        )

    from tools.retrieval.paper_entity_matcher import match_local_paper_entities
    from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

    entity = match_local_paper_entities(q)
    has_arxiv = extract_arxiv_id(q) is not None
    explicit_paper = _paper_search_intent(q) or bool(
        re.search(r"(论文|paper|arxiv|作者|摘要|title|pdf|download|文献)", ql, re.I)
    )

    if explicit_paper or has_arxiv or entity.confidence >= 0.65 or sub in ("table_lookup", "figure_lookup"):
        if _paper_search_intent(q):
            intent = "paper_search"
        elif _paper_read_intent(q):
            intent = "paper_read"
        else:
            intent = "paper_qa"
        if sub in ("table_lookup", "figure_lookup"):
            intent = "paper_qa"
        if intent == "paper_search":
            needs_web = not paper_discovery_forbids_web_only(q)
            if paper_discovery_prefers_web(q):
                needs_web = True
        else:
            needs_web = False
        if looks_like_paper_content_qa(q):
            needs_web = False
        return StructuredRouteResult(
            intent=intent,
            sub_intent=sub,
            source_preference="local_first",
            paper_match_mode=entity.match_mode,
            paper_ids=list(entity.paper_ids),
            paper_titles=list(entity.paper_titles),
            section_hint=_section_hint_from_sub_intent(sub),
            needs_table=(sub == "table_lookup"),
            needs_figure=(sub == "figure_lookup"),
            needs_web=needs_web,
            confidence=max(0.55, entity.confidence if entity.confidence > 0 else 0.72),
            route=Route.ARXIV if intent == "paper_search" else Route.RAG,
        )

    return StructuredRouteResult(
        intent="local_rag",
        sub_intent="open_ended_analysis",
        source_preference="local_first",
        route=Route.RAG,
        confidence=0.65,
        needs_web=open_ended_question_prefers_web_supplement(q),
    )


def route_query(question: str) -> Route:
    return route_query_structured(question).route


def allow_web_search_when_local_misses(route: Route) -> bool:
    """意图路由：仅当主路径是「本地知识 RAG」时，才允许在 Chroma 无命中后联网兜底。

    天气 / arXiv 已在 agent 中早退，不会走到本地检索；此处用于显式约束与文档说明。
    """
    return route is Route.RAG


# 用户明确要求联网时，即使本地已有高相关片段也执行联网检索并合并（短语越长越优先匹配）
_FORCE_WEB_PHRASES_ZH = (
    "请联网搜索",
    "请联网检索",
    "请联网查",
    "请联网",
    "请帮我联网搜索",
    "请上网搜索",
    "帮我联网搜索",
    "联网搜索一下",
    "联网检索一下",
    "必须联网",
    "需要联网搜索",
    "上网搜索一下",
    "去网上搜",
    "去网上查",
    "网上搜索一下",
    "用网络搜索",
    "从网上查",
)
_FORCE_WEB_PHRASES_EN = (
    "search the web",
    "search online",
    "web search",
    "look it up online",
    "google it",
)

# 从问题里去掉「请联网…」等，避免污染检索 query（长短语优先）
_STRIP_FORCE_WEB_PHRASES: tuple[str, ...] = (
    "请联网搜索一下",
    "请联网搜索",
    "请联网检索一下",
    "请联网检索",
    "请联网查一下",
    "请联网查",
    "请联网，",
    "请联网",
    "请帮我联网搜索一下",
    "请帮我联网搜索",
    "请联网搜索一下",
    "请联网检索一下",
    "请联网搜索",
    "请联网检索",
    "请上网搜索一下",
    "请上网搜索",
    "帮我联网搜索一下",
    "帮我联网搜索",
    "联网搜索一下",
    "联网检索一下",
    "需要联网搜索",
    "上网搜索一下",
    "去网上搜一下",
    "去网上搜",
    "去网上查一下",
    "去网上查",
    "网上搜索一下",
    "用网络搜索",
    "从网上查一下",
    "从网上查",
    "search the web for",
    "search the web",
    "please search online",
    "web search for",
)


def user_requests_forced_web_search(question: str) -> bool:
    """用户是否明确要求联网检索（与本地是否命中无关）。"""
    raw = question or ""
    if not raw.strip():
        return False
    ql = raw.lower()
    if any(p in raw for p in _FORCE_WEB_PHRASES_ZH):
        return True
    return any(p in ql for p in _FORCE_WEB_PHRASES_EN)


def strip_forced_web_search_phrases(question: str) -> str:
    """去掉「请联网搜索」等指令性短语，得到更适合搜索引擎的 query。"""
    s = (question or "").strip()
    if not s:
        return s
    for p in sorted(_STRIP_FORCE_WEB_PHRASES, key=len, reverse=True):
        if re.search(r"[\u4e00-\u9fff]", p):
            s = s.replace(p, " ")
        else:
            s = re.sub(r"(?i)" + re.escape(p), " ", s)
    s = " ".join(s.split())
    return s if s else (question or "").strip()

