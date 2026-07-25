"""
供 Agent 调用的 LangChain Tools（@tool）。

设计分层（避免与 MCP 混为一谈）：
- **进程内工具**：直接在本进程执行，如 `tool_weather`、`tool_search_arxiv`（无前缀或业务名）。
- **MCP 桥接工具**：统一 `tool_mcp_*` 前缀，内部仅转发到 `mcp_runtime` 拉起的 **stdio 子进程**
  （filesystem / browser / search / weather 等）；真实逻辑在独立 MCP server 里。

天气示例：`tool_weather` = LangChain 同进程；`tool_mcp_weather_query` = 调用 FastMCP 子进程
`mcp_servers/weather_fastmcp.py`（需在 `MCP_ENABLED` 中包含 `weather`）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from config import DEFAULT_NAMESPACE, DEFAULT_RETRIEVAL_STRATEGY, DEFAULT_TOP_K
from tools.agent.arxiv_search import download_pdf, get_paper_by_id, search_arxiv, Paper
from tools.agent.middleware import trace_event
from tools.rag.document import load_pdf
from tools.rag.knowledge import vector_store
from tools.rag.language import expand_retrieval_queries
from tools.storage.long_memory import retrieve_conversation_memories
from tools.agent.mcp_runtime import McpRuntimeError, mcp_runtime
from tools.storage.paper_library import (
    LocalPaper,
    get_paper,
    list_papers,
    reconcile_index_with_disk,
    upsert_paper,
)
from tools.storage.papers_db import list_papers as db_list_papers
from tools.agent.weather import get_weather, weather_info_tool_text

# 延迟导入，避免循环依赖且兼容未安装 langchain_core 的情况
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool  # type: ignore[no-redef]


def _resolve_safe_path(path: str) -> Path:
    """使用保守的允许列表来解析用户路径。

    允许的根目录：
    - 项目工作区根目录（若设置了 `WORKSPACE_ROOT` 环境变量）
    - `PaperSearchAssistant/data`
    - `PaperSearchAssistant/PaperSearchAssistant`（代码目录）
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError("path 参数是必填项")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()

    allowed_roots: list[Path] = []
    ws = os.getenv("WORKSPACE_ROOT")
    if ws:
        allowed_roots.append(Path(ws).resolve())
    allowed_roots.append((Path.cwd() / "data").resolve())
    allowed_roots.append(Path.cwd().resolve())

    if not any(str(p).startswith(str(root) + os.sep) or p == root for root in allowed_roots):
        raise ValueError(f"不允许的路径：{p}")
    return p


def _normalize_arxiv_id(value: str) -> str:
    """规范化 arXiv 输入：URL / `arXiv:` 前缀 / 版本 -> 规范化的论文 ID。"""
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = raw.replace("arXiv:", "").replace("ARXIV:", "").strip()
    # 从 URL 或自由文本中提取 arXiv ID
    m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", raw, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
    else:
        m_old = re.search(r"([a-z\-]+/\d{7}(?:v\d+)?)", raw, flags=re.IGNORECASE)
        if m_old:
            raw = m_old.group(1)
    # 去掉版本后缀，例如 2312.00732v2 -> 2312.00732
    raw = re.sub(r"v\d+$", "", raw, flags=re.IGNORECASE)
    return raw


@tool
def tool_weather(query: str) -> str:
    """查询指定城市的当前天气（**本进程内**调用 Open-Meteo，不经 MCP）。

    当用户问天气、气温、温度、冷热等相关问题时使用本工具。
    输入可以是城市名或包含城市的问题，如「北京天气」「上海温度」。
    若已启用 MCP `weather` 并希望演示跨进程工具边界，可改用 `tool_mcp_weather_query`。
    """
    # 城市抽取：输入可能是“查看一下今天北京的天气”这类句式，
    # 直接 split 会得到“查看一下今天北京”这种长字符串，导致落入 weather.py 的兜底。
    # 这里先在原句里找已知城市名；找不到再做一次弱解析兜底。
    q = (query or "").strip()
    city = ""
    for c in ["北京", "上海", "广州", "深圳", "杭州"]:
        if c in q:
            city = c
            break
    if not city:
        city = q
        for sep in ["天气", "温度", "气温", "的", " "]:
            if sep in city:
                city = city.split(sep)[0].strip() or city
        # 再次尝试：如果弱解析后仍包含城市名，则取城市名
        for c in ["北京", "上海", "广州", "深圳", "杭州"]:
            if c in city:
                city = c
                break
    if not city:
        city = "北京"
    info = get_weather(city)
    return weather_info_tool_text(info)


@tool
def tool_search_arxiv(query: str, max_results: int = 5) -> str:
    """在 arXiv 上检索学术论文。

    当用户明确要查论文、paper、arxiv 或学术文献时使用。
    输入为检索关键词或主题，返回论文标题、作者、摘要链接等。
    """
    refined_query = query
    q_raw = (query or "").strip()
    ql = q_raw.lower()

    # “最新/近几/近期”这类需求：需要按时间排序而不是 relevance
    want_latest = any(k in ql for k in ["最新", "近几", "近期", "recent", "latest"])
    # 数量偏好识别（支持“几篇/两篇/3篇”等）
    want_two = any(k in ql for k in ["两篇", "2篇", "2 papers", "two papers", "两条", "2条"])
    want_few = any(k in ql for k in ["几篇", "几条", "some papers", "a few papers"])
    num_m = re.search(r"(\d+)\s*(篇|条|papers?)", ql)
    if num_m:
        try:
            count_hint = max(1, min(20, int(num_m.group(1))))
        except Exception:
            count_hint = max_results
    elif want_two:
        count_hint = 2
    elif want_few:
        count_hint = 5
    else:
        count_hint = max_results
    effective_max_results = max(1, min(20, int(count_hint)))

    # 如果“用户显式要求最新”，同时还提取到时间范围，则两者都生效：
    # - latest: sort_by lastUpdatedDate
    # - range: submittedDate:[start TO end]
    extracted: dict = {}
    try:
        from datetime import datetime, timezone
        from models_qwen import qwen

        current_year = datetime.now(timezone.utc).year
        extract_prompt = (
            "你是一个 arXiv 检索参数提取器。\n"
            "输入是用户的自然语言检索需求，你需要从中抽取 constraints，输出严格 JSON（只能输出 JSON，不要输出解释、不要代码块）。\n\n"
            "输出 JSON 字段：\n"
            "{\n"
            '  "keywords": string[] ,\n'
            '  "author": string|null ,\n'
            '  "title_phrase": string|null ,\n'
            '  "year_start": number|null ,\n'
            '  "year_end": number|null ,\n'
            '  "want_latest": boolean\n'
            "}\n\n"
            "规则：\n"
            "1) 若用户提到“3DGS”，必须在 keywords 里展开为“3D Gaussian Splatting”。\n"
            "2) keywords 只包含与主题相关的检索词（尽量用英文技术词），不要包含多余的口语；"
            "若用户提到编辑/修改类需求，可加入 editing（或 manipulation）等与 arXiv 摘要常用英语一致的词。\n"
            "3) author：从原句抽取作者姓名/姓氏（例如 Kerbl），若抽不到则为 null。\n"
            "4) title_phrase：从原句抽取可能的标题短语（如果用户明确说了标题的一部分），否则为 null。\n"
            "5) year_start/year_end：\n"
            "   - 若用户问“最新/近几/近期”，可保持为 null（排序另行处理）；若用户也明确给出年份/时间范围，则填入。\n"
            "   - 支持解析：2023、2023年、2021-2023、2023年之后、2024年之前、近三年/近两年。\n"
            f"   - 当前年份(current_year)={current_year}，用于把“近三年”换算为 year_start/year_end。\n"
            "6) want_latest：若用户明确提到最新/近几/近期/latest/recent，则为 true，否则 false。\n"
        )

        resp = qwen.invoke(
            [
                {"role": "system", "content": extract_prompt},
                {"role": "user", "content": q_raw},
            ]
        )
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = raw.strip()

        import json

        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            extracted = json.loads(m.group(0))
    except Exception:
        extracted = {}

    # 若结构化抽取失败：回退到旧的“检索词改写”策略（提升召回稳定性）
    if not extracted:
        try:
            from models_qwen import qwen

            rewrite_prompt = (
                "你将用户的意图改写为简洁的 arXiv 检索查询。\n"
                "规则：\n"
                "- 保留关键技术术语。\n"
                "- 若用户提到“3DGS”，必须展开为“3D Gaussian Splatting”。\n"
                "- 优先使用短关键词短语，不要解释。\n"
                "- 只返回一行。\n"
                "- 如果用户使用中文提问，请在可能的情况下输出英文技术检索词。\n"
            )
            rewritten = qwen.invoke(
                [
                    {"role": "system", "content": rewrite_prompt},
                    {"role": "user", "content": q_raw},
                ]
            )
            candidate = (
                rewritten.content
                if hasattr(rewritten, "content")
                else str(rewritten)
            ).strip()
            if candidate:
                refined_query = candidate.splitlines()[0].strip()
        except Exception:
            refined_query = q_raw

    def _escape_arxiv_phrase(s: str) -> str:
        # arXiv query 里主要用引号包裹的字段：去掉双引号避免破坏语法
        return (s or "").replace('"', " ").strip()

    keywords = extracted.get("keywords") if isinstance(extracted, dict) else None
    keywords = keywords if isinstance(keywords, list) else None
    author = extracted.get("author") if isinstance(extracted, dict) else None
    title_phrase = extracted.get("title_phrase") if isinstance(extracted, dict) else None
    year_start = extracted.get("year_start") if isinstance(extracted, dict) else None
    year_end = extracted.get("year_end") if isinstance(extracted, dict) else None
    extracted_want_latest = (
        bool(extracted.get("want_latest")) if isinstance(extracted, dict) else False
    )

    # 关键词兜底：没有抽到就用 refined_query 当作 keywords 查询串
    keywords_clause = None
    if keywords:
        # 清理空项并拼成一个关键词子句（arXiv 会把空格解析为检索词组合）
        kws = [str(x).strip() for x in keywords if str(x).strip()]
        if kws:
            keywords_clause = " ".join(kws)
    if not keywords_clause:
        keywords_clause = refined_query

    # 日期范围：用 submittedDate 字段过滤（格式：YYYYMMDDhhmm）
    date_clause = None
    inferred_recent_range = False
    try:
        from datetime import datetime, timezone

        current_year = datetime.now(timezone.utc).year
        ys = int(year_start) if year_start is not None else None
        ye = int(year_end) if year_end is not None else None
        if ys is not None:
            if ye is None:
                ye = current_year
            if ye < ys:
                ys, ye = ye, ys
            # arXiv：上界用当前 UTC（YYYYMMDDhhmm），不下探到未来年份的虚构时点；与年末 cap 取较早者
            now = datetime.now(timezone.utc)
            start = f"{ys:04d}01010000"
            year_end_cap = f"{ye:04d}12312359"
            end_now = now.strftime("%Y%m%d%H%M")
            end = end_now if end_now < year_end_cap else year_end_cap
            date_clause = f"submittedDate:[{start} TO {end}]"
    except Exception:
        date_clause = None

    def _arxiv_group_kw(s: str) -> str:
        """多词关键词与日期 AND 时加括号，降低 arXiv 解析歧义。"""
        t = str(s or "").strip()
        if not t:
            return t
        if t.startswith("(") and t.endswith(")"):
            return t
        return f"({t})"

    clauses: list[str] = []
    if keywords_clause:
        clauses.append(_arxiv_group_kw(str(keywords_clause)))
    if author:
        a = _escape_arxiv_phrase(str(author))
        if a:
            clauses.append(f'au:"{a}"')
    if title_phrase:
        t = _escape_arxiv_phrase(str(title_phrase))
        if t:
            clauses.append(f'ti:"{t}"')
    if date_clause:
        clauses.append(date_clause)

    full_query = " AND ".join(clauses) if clauses else refined_query

    # latest 排序优先：用户显式要最新/近几 -> lastUpdatedDate
    sort_by = (
        "lastUpdatedDate"
        if want_latest or extracted_want_latest
        else "relevance"
    )

    # 分层回退：「最新」优先无 submittedDate、仅靠 lastUpdatedDate 排序（日期 AND 常导致 0 命中）；
    # 再试严格约束 -> 去作者/标题 -> 仅关键词 -> 编辑类同义 OR -> 核心词。
    fallback_queries: list[str] = []

    def _edit_relaxed_queries(kw_clause: str) -> list[str]:
        if not re.search(r"编辑|editing|改动|修改|操控", q_raw, re.I):
            return []
        k0 = str(kw_clause or "").strip()
        if not k0:
            return []
        # 去掉尾部的 editing，用 OR 覆盖 edit / manipulation 等，缓解词形不一致
        base = re.sub(r"\s+editing\s*$", "", k0, flags=re.I).strip()
        base = re.sub(r"\bediting\b", "", base, flags=re.I).strip() or k0
        syn = "(editing OR edit OR manipulation OR interactive OR sculpt OR deformation)"
        return [
            f"{_arxiv_group_kw(base)} AND {syn}",
            f"{_arxiv_group_kw(base)} AND (editing OR manipulation)",
        ]

    want_sort_latest = bool(want_latest or extracted_want_latest)
    if keywords_clause:
        # 1) 最新：先看关键词（无日期过滤），交给 lastUpdatedDate 排序
        if want_sort_latest:
            fallback_queries.append(str(keywords_clause))
        for eq in _edit_relaxed_queries(str(keywords_clause)):
            fallback_queries.append(eq)
    fallback_queries.append(full_query)
    if title_phrase or author:
        loose_fields: list[str] = []
        if keywords_clause:
            loose_fields.append(_arxiv_group_kw(str(keywords_clause)))
        if date_clause:
            loose_fields.append(date_clause)
        if loose_fields:
            fallback_queries.append(" AND ".join(loose_fields))
    if date_clause and keywords_clause:
        fallback_queries.append(str(keywords_clause))
    if keywords and isinstance(keywords, list):
        for kw in keywords:
            s = str(kw or "").strip()
            if s:
                fallback_queries.append(s)
    if "3dgs" in ql or "gaussian splatting" in ql or "高斯" in q_raw:
        fallback_queries.extend(
            [
                "3D Gaussian Splatting",
                "Gaussian Splatting",
                "3DGS",
                '(3D Gaussian Splatting) AND (editing OR manipulation)',
            ]
        )
    if refined_query and refined_query not in fallback_queries:
        fallback_queries.append(refined_query)
    if q_raw and q_raw not in fallback_queries:
        fallback_queries.append(q_raw)

    seen_q: set[str] = set()
    dedup_queries: list[str] = []
    for qq in fallback_queries:
        qx = " ".join(str(qq or "").split()).strip()
        if not qx:
            continue
        lk = qx.lower()
        if lk in seen_q:
            continue
        seen_q.add(lk)
        dedup_queries.append(qx)

    papers: list[Paper] = []
    used_query = ""
    fetch_n = max(int(effective_max_results), min(20, int(effective_max_results) * 3))

    trace_event(
        "tool_search_arxiv_start",
        {
            "want_latest": bool(want_latest or extracted_want_latest),
            "date_clause_present": bool(date_clause),
            "sort_by": str(sort_by),
            "raw_len": len(q_raw),
        },
    )

    def _search_arxiv_direct(
        api_query: str,
        *,
        max_results: int,
        sort_by: str,
    ) -> tuple[list[Paper], str | None]:
        """直接调用 arxiv API，并返回异常原因（避免上层吞掉异常）。"""
        import arxiv

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
        try:
            search = arxiv.Search(
                query=api_query,
                max_results=max_results,
                sort_by=sort_criterion,
            )
            out: list[Paper] = []
            for result in search.results():
                out.append(
                    Paper(
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        summary=result.summary,
                        url=result.entry_id,
                        published=result.published,
                    )
                )
            return out, None
        except Exception as e:
            return [], str(e)[:600]

    last_error: str | None = None
    for qq in dedup_queries[:12]:
        trace_event(
            "tool_search_arxiv_query_attempt",
            {"qq": str(qq)[:160], "sort_by": str(sort_by), "fetch_n": fetch_n},
        )

        got, err = _search_arxiv_direct(qq, max_results=fetch_n, sort_by=sort_by)
        if not got and sort_by == "lastUpdatedDate":
            got2, err2 = _search_arxiv_direct(
                qq,
                max_results=fetch_n,
                sort_by="relevance",
            )
            if got2:
                got = got2
                err = None
            else:
                err = err2 or err

        if err:
            last_error = err
            trace_event(
                "tool_search_arxiv_query_error",
                {"qq": str(qq)[:160], "error": err[:180]},
            )

        if got:
            papers = got[: effective_max_results]
            used_query = qq
            break

    trace_event(
        "tool_search_arxiv_end",
        {"used_query": str(used_query)[:160], "papers": len(papers)},
    )

    if not papers:
        err_hint = f"\n- arXiv 异常原因: {last_error}" if last_error else ""
        return (
            "arXiv 暂未返回结果（已尝试多级放宽查询）。\n"
            f"- 原始请求: {q_raw}\n"
            f"- 最终尝试查询: {', '.join(dedup_queries[:5])}"
            f"{err_hint}"
        )

    def _brief_summary(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return "暂无摘要信息。"
        # 取首句作为“简要概括”；过长则截断
        m = re.split(r"(?<=[\.\!\?。！？])\s+", s, maxsplit=1)
        first = m[0].strip() if m else s
        if len(first) > 180:
            first = first[:180].rstrip() + "..."
        return first

    lines = []
    for i, p in enumerate(papers, start=1):
        authors = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors += " 等"
        published = getattr(p, "published", None)
        published_str = str(published) if published else ""
        # arXiv ID 用于后续“标题->ID 映射/下载”，尽量从 entry_id 中解析
        # 注意：不同工具可能返回 http(s)://arxiv.org/abs/<id> 或直接 <id>。
        arxiv_id = _normalize_arxiv_id(getattr(p, "url", "") or "")
        lines.append(
            f"{i}. {p.title}\n"
            f"   作者: {authors}\n"
            f"   时间: {published_str}\n"
            f"   简要概括: {_brief_summary(p.summary or '')}\n"
            f"   arXiv: {arxiv_id or '(unknown)'}\n"
            f"   链接: {p.url}\n"
            f"   摘要: {(p.summary or '')[:4000]}"
        )
    header = "arXiv 检索结果（按论文分点）："
    recognized = [
        f"- 主题关键词: {keywords_clause}" if keywords_clause else "- 主题关键词: （未识别）",
        f"- 数量偏好: {effective_max_results} 篇",
        f"- 时间偏好: {'最新/最近' if (want_latest or extracted_want_latest) else '未指定'}",
    ]
    if date_clause:
        recognized.append(f"- 日期范围: {date_clause}")
        if inferred_recent_range:
            recognized.append("- 日期范围来源: 已按“最新”自动映射为近三年")
    if author:
        recognized.append(f'- 作者约束: au:"{_escape_arxiv_phrase(str(author))}"')
    if title_phrase:
        recognized.append(f'- 标题短语约束: ti:"{_escape_arxiv_phrase(str(title_phrase))}"')
    if full_query and full_query != q_raw:
        header += f"\n(检索约束: {q_raw} -> {full_query})"
    if used_query and used_query != full_query:
        header += f"\n(结果回退命中查询: {used_query})"
    header += "\n\n需求拆分识别：\n" + "\n".join(recognized)
    return header + "\n\n" + "\n\n".join(lines)


@tool
def tool_search_knowledge(
    query: str,
    namespace: str = DEFAULT_NAMESPACE,
    k: int = DEFAULT_TOP_K,
) -> str:
    """从本地向量知识库中检索与问题相关的文档片段。

    当用户问的是基于已入库文档的内容（笔记、报告、论文摘要等）时使用。
    输入为用户的自然语言问题；可选 namespace 指定知识库分区，k 为返回条数。
    返回检索到的上下文文本，供模型基于此生成回答。
    """
    from models_qwen import qwen

    queries = expand_retrieval_queries(
        query,
        strategy="hybrid_rerank",
        llm=qwen,
    )
    if not queries:
        fb = (query or "").strip()
        queries = [fb] if fb else []
    from tools.rag.retrieval_merge import retrieve_with_public_merge

    docs_scores = retrieve_with_public_merge(
        vector_store,
        queries=queries,
        namespace=namespace,
        k=k,
        score_threshold=0.5,
        strategy="hybrid_rerank",
    )
    if not docs_scores:
        return "知识库中未检索到与问题相关的内容。"
    parts = []
    for doc, score in docs_scores:
        content = getattr(doc, "page_content", str(doc))
        source = getattr(doc, "metadata", {}).get("source", "unknown")
        parts.append(f"[来源: {source}, 相关度: {score:.3f}]\n{content}")
    return "\n\n---\n\n".join(parts)


@tool
def tool_search_chat_history(query: str, session_id: str = "all", k: int = 6) -> str:
    """从已向量化的历史对话中检索相关问答记录。

    当用户问“我之前问过什么/上周聊过什么/历史里提到过哪篇论文”时使用。
    """
    sid = (session_id or "all").strip() or "all"
    sid_arg = None if sid == "all" else sid
    items = retrieve_conversation_memories(
        query=query,
        session_id=sid_arg,
        k=max(1, min(int(k), 20)),
        score_threshold=0.8,
        strategy=DEFAULT_RETRIEVAL_STRATEGY,
    )
    if not items:
        return "历史对话向量库中未检索到相关记录。"
    parts = []
    for i, it in enumerate(items, start=1):
        parts.append(
            f"{i}. [session={it.get('session_id')}, turn={it.get('turn_index')}, score={it.get('score'):.3f}]\n"
            f"{it.get('text')}"
        )
    return "历史对话检索结果：\n\n" + "\n\n---\n\n".join(parts)


@tool
def tool_read_file(path: str, max_chars: int = 8000) -> str:
    """读取一个本地文本文件内容（受路径白名单限制）。

    适用场景：用户在文本框里明确要求“读取某个文件内容”。
    注意：本工具可能被 HumanApprovalMiddleware 配置为需要人工审批。
    """
    p = _resolve_safe_path(path)
    if not p.exists():
        return f"文件不存在：{p}"
    if p.is_dir():
        return f"目标是目录，不是文件：{p}"
    txt = p.read_text(encoding="utf-8", errors="replace")
    lim = max(200, min(int(max_chars or 8000), 50000))
    if len(txt) > lim:
        return txt[:lim] + f"\n\n... (truncated, total_chars={len(txt)})"
    return txt


@tool
def tool_delete_file(path: str) -> str:
    """删除一个本地文件（受路径白名单限制）。

    强烈建议配合 HumanApprovalMiddleware 对该工具启用人工审批。
    """
    p = _resolve_safe_path(path)
    if not p.exists():
        return f"文件不存在：{p}"
    if p.is_dir():
        return f"拒绝删除目录：{p}"
    p.unlink()
    return f"已删除：{p}"


@tool
def tool_mcp_list_tools(server: str = "filesystem") -> str:
    """列出指定 MCP server 暴露的 tools。server 典型值：filesystem / browser / search / weather。"""
    try:
        tools = mcp_runtime.list_tools((server or "").strip() or "filesystem")
    except McpRuntimeError as e:
        return f"MCP 运行时错误：{e}"
    except Exception as e:
        return f"MCP 错误：{e}"
    if not tools:
        return "没有可用的 MCP tools。"
    lines = []
    for t in tools:
        lines.append(f"- {t.get('name')}: {t.get('description') or ''}".rstrip())
    return "\n".join(lines)


def _mcp_result_to_text(res) -> str:
    content = getattr(res, "content", None)
    if content is None and isinstance(res, dict):
        content = res.get("content")
    if not content:
        return str(res)
    parts: list[str] = []
    for item in content:
        txt = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if txt:
            parts.append(str(txt))
        else:
            parts.append(str(item))
    return "\n\n".join(parts) if parts else str(res)


def _mcp_call_tool_from_json(server: str, tool_name: str, arguments_json: str = "{}") -> str:
    """进程内直接调 MCP（不经 LangChain `BaseTool.__call__`）。供多个 `@tool` 复用。"""
    srv = (server or "").strip()
    name = (tool_name or "").strip()
    if not srv or not name:
        return "server 与 tool_name 为必填项。"
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            return "arguments_json 必须是一个 JSON 对象。"
    except Exception as e:
        return f"arguments_json 非法：{e}"

    try:
        res = mcp_runtime.call_tool(srv, name, args)
    except McpRuntimeError as e:
        return f"MCP 运行时错误：{e}"
    except Exception as e:
        return f"MCP 错误：{e}"
    return _mcp_result_to_text(res)


@tool
def tool_mcp_call(server: str, tool_name: str, arguments_json: str = "{}") -> str:
    """通用 MCP tool 调用（server+tool_name+JSON参数）。

    适用场景：当你不确定工具封装是否存在，或想调用 browser MCP 的具体动作时。
    """
    return _mcp_call_tool_from_json(server, tool_name, arguments_json)


def _try_mcp_tool_candidates(server: str, candidates: list[str], args: dict) -> str:
    errors: list[str] = []
    for name in candidates:
        try:
            res = mcp_runtime.call_tool(server, name, args)
            text = _mcp_result_to_text(res)
            return f"[tool={name}]\n{text}"
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
    return (
        f"MCP 错误：server={server} 的候选工具中没有一个可用。\n"
        f"已尝试：{', '.join(candidates)}\n"
        f"错误列表：\n- " + "\n- ".join(errors[:8])
    )


@tool
def tool_mcp_fs_list_directory(path: str = "data") -> str:
    """通过 MCP filesystem server 列目录（受 MCP_FILESYSTEM_ALLOWED_DIRS 限制）。"""
    return _mcp_call_tool_from_json("filesystem", "list_directory", f'{{"path": "{path}"}}')


@tool
def tool_mcp_fs_read_text_file(path: str, head: int | None = None, tail: int | None = None) -> str:
    """通过 MCP filesystem server 读文本文件（UTF-8）。"""
    import json

    args: dict = {"path": path}
    if head is not None:
        args["head"] = int(head)
    if tail is not None:
        args["tail"] = int(tail)
    return _mcp_call_tool_from_json("filesystem", "read_text_file", json.dumps(args, ensure_ascii=False))


@tool
def tool_mcp_fs_write_file(path: str, content: str) -> str:
    """通过 MCP filesystem server 写文件（会覆盖）。"""
    import json

    return _mcp_call_tool_from_json(
        "filesystem",
        "write_file",
        json.dumps({"path": path, "content": content}, ensure_ascii=False),
    )


def _mcp_browser_open_url(url: str) -> str:
    """供 browser 相关 `@tool` 内部复用；勿嵌套调用 `tool_mcp_browser_open`（BaseTool）。"""
    if not (url or "").strip():
        return "缺少 url 参数。"
    args = {"url": url.strip()}
    candidates = ["browser_navigate", "navigate", "goto", "open_page", "open_url"]
    return _try_mcp_tool_candidates("browser", candidates, args)


@tool
def tool_mcp_browser_open(url: str) -> str:
    """浏览器 MCP 专用：打开/跳转到指定 URL。"""
    return _mcp_browser_open_url(url or "")


@tool
def tool_mcp_browser_get_title(url: str = "") -> str:
    """浏览器 MCP 专用：读取当前页面标题；可选先打开 URL。"""
    target = (url or "").strip()
    if target:
        _ = _mcp_browser_open_url(target)
    candidates = ["browser_evaluate", "evaluate", "get_title", "page_title", "title"]

    # 优先使用评估脚本，因为它在 browser MCP 服务器中更常见。
    eval_args = {"script": "document.title"}
    out = _try_mcp_tool_candidates("browser", candidates, eval_args)
    if "候选工具中没有一个可用" in out:
        # 有些实现提供的 get_title 无需参数。
        out = _try_mcp_tool_candidates("browser", ["get_title", "page_title", "title"], {})
    return out


@tool
def tool_mcp_browser_get_text(url: str = "", max_chars: int = 4000) -> str:
    """浏览器 MCP 专用：读取页面主要文本；可选先打开 URL。"""
    target = (url or "").strip()
    if target:
        _ = _mcp_browser_open_url(target)
    candidates = ["browser_evaluate", "evaluate", "extract_text", "get_text", "page_text", "snapshot"]
    eval_args = {
        "script": (
            "(() => { const t = (document.body && document.body.innerText) || ''; "
            f"return t.slice(0, {max(200, min(int(max_chars), 20000))}); }})()"
        )
    }
    out = _try_mcp_tool_candidates("browser", candidates, eval_args)
    if "候选工具中没有一个可用" in out:
        out = _try_mcp_tool_candidates(
            "browser",
            ["extract_text", "get_text", "page_text", "snapshot"],
            {"max_chars": max(200, min(int(max_chars), 20000))},
        )
    return out


@tool
def tool_mcp_brave_web_search(query: str, count: int = 8, offset: int = 0) -> str:
    """通过 MCP（Brave Search）做**通用网页检索**，用于新闻、百科、实时信息等（非论文库）。

    需启用 MCP `search` server 并配置 `BRAVE_API_KEY`（或 `MCP_BRAVE_API_KEY`）。
    与本地 `tool_search_knowledge` / 论文工具互补：本工具面向开放互联网摘要。
    """
    q = (query or "").strip()
    if not q:
        return "query 不能为空。"
    n = max(1, min(int(count or 8), 20))
    off = max(0, min(int(offset or 0), 9))
    args = {"query": q, "count": n, "offset": off}
    return _try_mcp_tool_candidates(
        "search",
        ["brave_web_search", "brave_search", "web_search"],
        args,
    )


@tool
def tool_mcp_brave_news_search(query: str, count: int = 8) -> str:
    """通过 MCP（Brave Search）做**新闻向检索**（若当前 MCP server 暴露对应 tool）。

    与 `tool_mcp_brave_web_search` 相同依赖；若 server 无独立 news tool，将尝试回退到网页检索。
    """
    q = (query or "").strip()
    if not q:
        return "query 不能为空。"
    n = max(1, min(int(count or 8), 20))
    args = {"query": q, "count": n}
    return _try_mcp_tool_candidates(
        "search",
        ["brave_news_search", "news_search", "brave_web_search"],
        args,
    )


@tool
def tool_mcp_weather_query(city: str) -> str:
    """通过 MCP **独立子进程**查询城市当前天气（FastMCP `weather_fastmcp` → Open-Meteo）。

    与 `tool_weather` 区别：本工具走 stdio MCP，用于演示「插件式」工具与进程边界。
    需在环境变量 `MCP_ENABLED` 中包含 `weather`。城市名建议用英文小写拼音（如 beijing）。
    """
    c = (city or "").strip() or "beijing"
    return _mcp_call_tool_from_json("weather", "query_weather", json.dumps({"city": c}, ensure_ascii=False))


@tool
def tool_mcp_weather_season_tips(season: str) -> str:
    """通过 MCP `weather` server 获取季节穿衣/出行提示（演示用静态内容，非实时气象）。"""
    s = (season or "").strip() or "spring"
    return _mcp_call_tool_from_json(
        "weather",
        "season_weather_tips",
        json.dumps({"season": s}, ensure_ascii=False),
    )


@tool
def tool_mcp_browser_screenshot(
    url: str = "",
    file_path: str = "data/screenshots/mcp_page.png",
    full_page: bool = True,
) -> str:
    """浏览器 MCP 专用：页面截图；可选先打开 URL。"""
    target = (url or "").strip()
    if target:
        _ = _mcp_browser_open_url(target)
    args = {"path": file_path, "full_page": bool(full_page)}
    candidates = ["browser_screenshot", "take_screenshot", "screenshot", "capture_screenshot"]
    return _try_mcp_tool_candidates("browser", candidates, args)

@tool
def tool_list_local_papers(query: str = "") -> str:
    """列出本地已下载/登记的论文（data/papers/index.json）。

    当用户问“我本地有哪些论文/有哪些已保存论文/列一下论文库”时使用。
    query 可为空；若不为空，会对 title/arxiv_id 做简单包含过滤。
    """
    reconcile_index_with_disk()
    q = (query or "").strip().lower()
    items = list_papers()
    if q:
        items = [
            p
            for p in items
            if q in (str(p.get("title") or "").lower())
            or q in (str(p.get("arxiv_id") or "").lower())
        ]
    if not items:
        return "本地论文库为空（data/papers 下无 PDF 或 index.json 为空）。"
    lines = []
    for i, p in enumerate(items[:30], start=1):
        title = p.get("title") or "(no title)"
        arxiv_id = p.get("arxiv_id") or "unknown"
        pdf_path = p.get("pdf_path") or ""
        lines.append(f"{i}. {title}\n   arxiv_id: {arxiv_id}\n   pdf: {pdf_path}")
    if len(items) > 30:
        lines.append(f"... 还有 {len(items) - 30} 篇未显示")
    return "本地论文库：\n\n" + "\n\n".join(lines)


@tool
def tool_download_arxiv_pdf(arxiv_id: str) -> str:
    """下载指定 arXiv ID 的 PDF 到本地 data/papers，并写入论文库索引。"""
    arxiv_id = _normalize_arxiv_id(arxiv_id)
    if not arxiv_id:
        return "请提供 arXiv ID，例如 2401.12345。"
    from pathlib import Path

    reconcile_index_with_disk()
    existing = get_paper(arxiv_id) or {}
    pdf_path = Path(existing.get("pdf_path") or f"data/papers/{arxiv_id}.pdf")
    downloaded = False
    if not pdf_path.exists():
        pdf_path = download_pdf(arxiv_id, dest_dir="data/papers")
        downloaded = True
    paper = get_paper_by_id(arxiv_id)
    rec = upsert_paper(
        LocalPaper(
            arxiv_id=arxiv_id,
            pdf_path=str(pdf_path.as_posix()),
            title=paper.title if paper else None,
            authors=paper.authors if paper else [],
            published=(paper.published.isoformat() if paper else None),
            url=(paper.url if paper else f"https://arxiv.org/abs/{arxiv_id}"),
        )
    )
    if downloaded:
        return f"已下载 PDF：{rec.pdf_path}\n可在接口 /papers/view/{pdf_path.name} 查看。"
    return f"本地已存在 PDF（跳过下载）：{rec.pdf_path}\n可在接口 /papers/view/{pdf_path.name} 查看。"


@tool
def tool_get_local_paper(arxiv_id: str) -> str:
    """查看某篇本地论文的元数据与查看链接（如果已下载）。"""
    reconcile_index_with_disk()
    rec = get_paper(arxiv_id)
    if not rec:
        return "本地论文库中未找到该 arXiv ID。你可以先用 tool_search_arxiv 检索，再下载。"
    pdf_path = rec.get("pdf_path") or ""
    view_url = f"/papers/view/{__import__('pathlib').Path(pdf_path).name}" if pdf_path else ""
    title = rec.get("title") or "(no title)"
    return (
        f"标题: {title}\n"
        f"arxiv_id: {rec.get('arxiv_id')}\n"
        f"url: {rec.get('url')}\n"
        f"pdf: {pdf_path}\n"
        f"view: {view_url}\n"
        f"namespaces: {rec.get('namespaces') or []}"
    )


@tool
def tool_get_local_paper_db_info(arxiv_id: str) -> str:
    """从本地论文数据库（SQLite）读取一篇论文的结构化信息（标题/作者/摘要/命名空间等）。"""
    reconcile_index_with_disk()
    aid = _normalize_arxiv_id(arxiv_id)
    if not aid:
        return "请提供有效 arXiv ID，例如 2312.00732。"
    rec = get_paper(aid)
    if not rec:
        return f"本地数据库中未找到论文：{aid}"
    authors = ", ".join(rec.get("authors") or [])
    namespaces = rec.get("namespaces") or []
    summary = (rec.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "..."
    return (
        "本地论文数据库信息：\n"
        f"- arxiv_id: {rec.get('arxiv_id')}\n"
        f"- title: {rec.get('title') or ''}\n"
        f"- authors: {authors}\n"
        f"- published: {rec.get('published') or ''}\n"
        f"- url: {rec.get('url') or ''}\n"
        f"- pdf_path: {rec.get('pdf_path') or ''}\n"
        f"- namespaces: {namespaces}\n"
        f"- summary: {summary or '(empty)'}"
    )


@tool
def tool_search_local_papers(
    query: str = "",
    author: str = "",
    title: str = "",
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 10,
) -> str:
    """在本地论文库（SQLite）中检索论文元数据（标题/作者/摘要/年份）。

    适用场景：
    - 用户要找“我本地有哪些关于 X 的论文”
    - 用户要按作者/标题/时间范围筛选本地论文

    参数：
    - query: 关键词（走 FTS/LIKE）
    - author/title: 可选过滤
    - year_from/year_to: 可选年份范围
    - limit: 返回条数（默认 10）
    """
    kw = (query or "").strip() or None
    au = (author or "").strip() or None
    ti = (title or "").strip() or None
    lim = max(1, min(int(limit or 10), 30))
    results = db_list_papers(
        keyword=kw,
        author=au,
        title=ti,
        year_from=year_from,
        year_to=year_to,
        limit=lim,
        offset=0,
    )
    if not results:
        return "本地论文库未检索到匹配的论文。"
    lines = []
    for i, p in enumerate(results, start=1):
        authors = ", ".join((p.get("authors") or [])[:5])
        if len(p.get("authors") or []) > 5:
            authors += " 等"
        lines.append(
            f"{i}. {p.get('title') or '(no title)'}\n"
            f"   arxiv_id: {p.get('arxiv_id')}\n"
            f"   authors : {authors}\n"
            f"   published: {p.get('published')}\n"
            f"   url     : {p.get('url')}\n"
            f"   pdf     : {p.get('pdf_path')}\n"
            f"   summary : {(p.get('summary') or '')[:220]}..."
        )
    return "本地论文检索结果：\n\n" + "\n\n".join(lines)


@tool
def tool_ingest_arxiv_paper(
    arxiv_id: str,
    embed_full_text: bool = True,
    namespace: str = "",
) -> str:
    """将指定 arXiv 论文“下载到本地 + 写入论文库 + 可选全文入库（向量库）”。

    适用场景：
    - 用户要求“把这篇论文保存到本地库/下载 PDF/入库后再问”
    - 后续希望基于论文全文做 RAG（需要 embed_full_text=True）

    行为：
    1) 下载 PDF 到 data/papers/<arxiv_id>.pdf（若已存在则复用）
    2) 写入/更新本地论文库元数据（SQLite）
    3) 可选：解析 PDF 并嵌入 paper:<id>:full（或你传入的 namespace），并默认再写入公共向量库（RAG_PUBLIC_NAMESPACE）
    """
    from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline

    override = namespace.strip() if namespace and namespace.strip() else None
    return ingest_arxiv_paper_full_pipeline(
        arxiv_id,
        embed_full_text=embed_full_text,
        paper_namespace_override=override,
    )


def get_agent_tools():
    """返回供 bind_tools / ReAct 使用的工具列表。"""
    tools = [
        tool_weather,
        tool_search_arxiv,
        tool_search_chat_history,
        tool_read_file,
        tool_delete_file,
        tool_search_local_papers,
        tool_list_local_papers,
        tool_get_local_paper,
        tool_get_local_paper_db_info,
        tool_download_arxiv_pdf,
        tool_ingest_arxiv_paper,
        tool_search_knowledge,
    ]
    # MCP 开关：
    # - 未设置时：默认启用 MCP 工具（兼容现有行为）
    # - 显式设置为空字符串（MCP_ENABLED=）：禁用所有 tool_mcp_*
    mcp_enabled = os.getenv("MCP_ENABLED", "filesystem,browser")
    if (mcp_enabled or "").strip():
        tools.extend(
            [
                tool_mcp_list_tools,
                tool_mcp_call,
                tool_mcp_fs_list_directory,
                tool_mcp_fs_read_text_file,
                tool_mcp_fs_write_file,
                tool_mcp_browser_open,
                tool_mcp_browser_get_title,
                tool_mcp_browser_get_text,
                tool_mcp_browser_screenshot,
            ]
        )
        # Brave Search MCP（仅当 mcp_runtime 已注册 search server，通常需 BRAVE_API_KEY）
        if "search" in mcp_runtime.enabled_servers():
            tools.extend(
                [
                    tool_mcp_brave_web_search,
                    tool_mcp_brave_news_search,
                ]
            )
        if "weather" in mcp_runtime.enabled_servers():
            tools.extend(
                [
                    tool_mcp_weather_query,
                    tool_mcp_weather_season_tips,
                ]
            )
    return tools
