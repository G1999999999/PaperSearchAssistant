from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config import (
    RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD,
)
from tools.agent.middleware import trace_event
from tools.agent.router import (
    Route,
    paper_discovery_forbids_web_only,
    route_query_structured,
)


@dataclass
class QueryRouteDecision:
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


def build_query_route(question: str) -> QueryRouteDecision:
    """统一高级路由入口：规则短路 + 实体匹配 + 结构化输出。"""
    base = route_query_structured(question)
    out = QueryRouteDecision(
        intent=base.intent,
        sub_intent=base.sub_intent,
        source_preference=base.source_preference,
        paper_match_mode=base.paper_match_mode,
        paper_ids=list(base.paper_ids),
        paper_titles=list(base.paper_titles),
        section_hint=base.section_hint,
        needs_table=bool(base.needs_table),
        needs_figure=bool(base.needs_figure),
        needs_web=bool(base.needs_web),
        confidence=float(base.confidence),
        route=base.route,
    )
    # 本地命中置信高：优先本地源；但 paper_search 不因高置信单独关掉联网（发现新论文、最新等）
    if out.confidence >= float(RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD):
        out.source_preference = "local_first"
        if out.intent == "local_paper_search":
            out.needs_web = False
        elif out.intent == "paper_search":
            if paper_discovery_forbids_web_only(question):
                out.needs_web = False
            else:
                trace_event(
                    "query_route_paper_search_high_conf_keeps_needs_web",
                    {
                        "needs_web": bool(out.needs_web),
                        "confidence": float(out.confidence),
                    },
                )
        else:
            out.needs_web = False
    return out

