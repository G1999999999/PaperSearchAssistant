"""
联网检索摘要：在本地 Chroma 无命中时作为 RAG 上下文的兜底。

优先级（默认 RAG_WEB_BACKEND=auto，可在环境变量中调整）：
1. （auto）MCP Streamable HTTP：MCP_ENABLED 含 mcpmarket（或 MCP_STREAMABLE_SERVER_NAME）且配置
   MCP_STREAMABLE_URL 时，经 `mcp_runtime` 调用远程 search 类 tool（默认 search_engine）；
   可用 RAG_WEB_STREAMABLE_FIRST=0 跳过此步。
2. （auto）MCP Brave stdio：MCP_ENABLED 含 `search` 时调用 brave_web_search 等。
3. DuckDuckGo：多后端依次尝试（lite / html / api）。
4. 可选 SearXNG：RAG_WEB_SEARXNG_URL。
5. 可选 Brave Search：RAG_WEB_BRAVE_API_KEY。

`RAG_WEB_BACKEND=ddg_first` 可关闭上述两类 MCP，仅保留旧顺序（先 DDG 等）。

依赖：pip install duckduckgo-search requests
可选：pip install lxml；Streamable MCP 另需 mcp httpx httpx-sse
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlencode

import requests


def _normalize_hit(item: dict[str, Any]) -> dict[str, Any] | None:
    title = (item.get("title") or "").strip()
    url = (
        (item.get("href") or item.get("url") or item.get("link") or "")
        .strip()
    )
    body = (
        item.get("body")
        or item.get("snippet")
        or item.get("content")
        or item.get("description")
        or ""
    )
    body = str(body).strip()
    if not body and not title:
        return None
    return {"title": title, "url": url or "web", "snippet": body or title}


def _search_duckduckgo_multi(
    q: str, *, max_results: int, backends: list[str]
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [], (
            "未安装联网检索依赖。请执行：pip install duckduckgo-search\n"
            "启用 DDG lite/html 后端建议：pip install lxml"
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for backend in backends:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(
                    q, max_results=max_results, backend=backend
                ):
                    if isinstance(r, dict):
                        norm = _normalize_hit(r)
                        if norm:
                            rows.append(norm)
                    if len(rows) >= max_results:
                        break
            if rows:
                return rows[:max_results], None
        except Exception as e:
            msg = str(e).strip() or repr(e)
            errors.append(f"{backend}: {msg[:200]}")

    if errors:
        return [], (
            "DuckDuckGo 各后端均失败（常见为 202 Ratelimit）。"
            "可：① 降低提问频率；② 配置 RAG_WEB_SEARXNG_URL；"
            "③ 配置 RAG_WEB_BRAVE_API_KEY。"
            f" 详情：{' | '.join(errors[:3])}"
        )
    return [], (
        "DuckDuckGo 未返回条目。可尝试配置 SearXNG 或 Brave Search API（见环境变量说明）。"
    )


def _search_searxng(
    base_url: str, q: str, *, max_results: int
) -> tuple[list[dict[str, Any]], str | None]:
    url = f"{base_url}/search?{urlencode({'q': q, 'format': 'json'})}"
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "PaperSearchAssistant/1.0"})
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        return [], f"SearXNG 请求失败：{e!s}"[:400]

    raw = data.get("results") or []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            norm = _normalize_hit(item)
            if norm:
                rows.append(norm)
        if len(rows) >= max_results:
            break
    if rows:
        return rows[:max_results], None
    return [], "SearXNG 返回结果为空（实例可能限流或关键词无匹配）。"


def _search_brave(
    api_key: str, q: str, *, max_results: int
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": q, "count": min(max_results, 20)},
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            timeout=25,
        )
        if r.status_code == 401:
            return [], "Brave Search API Key 无效或未授权。"
        if r.status_code == 429:
            return [], "Brave Search API 触发限流，请稍后重试或升级额度。"
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        return [], f"Brave Search 请求失败：{e!s}"[:400]

    web = data.get("web") or {}
    raw = web.get("results") or []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            norm = _normalize_hit(item)
            if norm:
                rows.append(norm)
        if len(rows) >= max_results:
            break
    if rows:
        return rows[:max_results], None
    return [], "Brave Search 未返回可用网页结果。"


def _mcp_call_tool_result_text(res: Any) -> str:
    """从 MCP CallToolResult 抽取纯文本。"""
    content = getattr(res, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for item in content:
        txt = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if txt:
            parts.append(str(txt))
    return "\n\n".join(parts)


def _parse_brave_mcp_web_text(text: str) -> list[dict[str, Any]]:
    """解析 @modelcontextprotocol/server-brave-search 返回的「Title/Description/URL」文本块。"""
    text = (text or "").strip()
    if not text:
        return []
    if text.lstrip().lower().startswith("error:"):
        return []
    rows: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n+", text):
        block = block.strip()
        if not block:
            continue
        title = ""
        desc = ""
        url = ""
        for line in block.splitlines():
            ls = line.strip()
            if ls.startswith("Title:"):
                title = ls[len("Title:") :].strip()
            elif ls.startswith("Description:"):
                desc = ls[len("Description:") :].strip()
            elif ls.startswith("URL:"):
                url = ls[len("URL:") :].strip()
        norm = _normalize_hit({"title": title, "body": desc, "href": url})
        if norm:
            rows.append(norm)
    return rows


def _search_mcp_brave_web(
    q: str, *, max_results: int
) -> tuple[list[dict[str, Any]], str | None]:
    """若已启用 MCP `search` server，经 stdio 调用 brave_web_search。

    返回 ([], None) 表示未配置 search server（非错误，交由其它后端）。
    返回 ([], "…") 表示已尝试 MCP 但失败。
    """
    try:
        from tools.agent.mcp_runtime import mcp_runtime
    except Exception as exc:  # pragma: no cover
        return [], f"MCP 模块不可用：{exc!s}"[:240]

    if "search" not in mcp_runtime.enabled_servers():
        return [], None

    n = max(1, min(int(max_results), 20))
    args: dict[str, Any] = {"query": q, "count": n, "offset": 0}
    last_err: str | None = None
    for tname in ("brave_web_search", "brave_search", "web_search"):
        try:
            res = mcp_runtime.call_tool("search", tname, args)
        except Exception as e:
            last_err = f"{tname}: {e!s}"[:280]
            continue
        if getattr(res, "isError", False):
            body = _mcp_call_tool_result_text(res)
            last_err = (body or f"{tname} isError").strip()[:280]
            continue
        body = _mcp_call_tool_result_text(res)
        rows = _parse_brave_mcp_web_text(body)
        if rows:
            return rows[:max_results], None
        last_err = f"{tname} 返回空结果"
    return [], last_err


def _rows_from_json_serp(data: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    """从 MCP/BrightData 等返回的 JSON 中抽取若干 {title,url,href,snippet,...}。"""
    if depth > 8:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                norm = _normalize_hit(item)
                if norm:
                    rows.append(norm)
                else:
                    rows.extend(_rows_from_json_serp(item, depth=depth + 1))
            elif isinstance(item, list):
                rows.extend(_rows_from_json_serp(item, depth=depth + 1))
        return rows
    if isinstance(data, dict):
        norm = _normalize_hit(data)
        if norm and (norm.get("snippet") or data.get("url") or data.get("href") or data.get("link")):
            return [norm]
        for key in (
            "results",
            "items",
            "organic_results",
            "organic",
            "web",
            "data",
            "hits",
            "documents",
            "entries",
            "result",
        ):
            if key in data:
                sub = data[key]
                got = _rows_from_json_serp(sub, depth=depth + 1)
                if got:
                    return got
        for v in data.values():
            if isinstance(v, (list, dict)):
                got = _rows_from_json_serp(v, depth=depth + 1)
                if got:
                    return got
    return []


def _parse_streamable_mcp_search_text(
    text: str, *, max_results: int
) -> list[dict[str, Any]]:
    """解析 Streamable MCP 搜索工具返回（JSON、Brave 风格文本块或 Markdown 链接）。"""
    text = (text or "").strip()
    if not text or text.lstrip().lower().startswith("error:"):
        return []
    try:
        data = json.loads(text)
        rows = _rows_from_json_serp(data)
        if rows:
            return rows[:max_results]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    rows = _parse_brave_mcp_web_text(text)
    if rows:
        return rows[:max_results]
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", text):
        norm = _normalize_hit({"title": m.group(1), "href": m.group(2), "body": ""})
        if norm:
            out.append(norm)
        if len(out) >= max_results:
            break
    return out[:max_results]


def _search_mcp_streamable_web(
    q: str, *, max_results: int
) -> tuple[list[dict[str, Any]], str | None]:
    """若已注册 Streamable HTTP MCP（默认别名 mcpmarket），调用搜索类 tool。"""
    try:
        from tools.agent.mcp_runtime import mcp_runtime
    except Exception as exc:  # pragma: no cover
        return [], f"MCP 模块不可用：{exc!s}"[:240]

    from config import MCP_STREAMABLE_SERVER_NAME, MCP_STREAMABLE_TOOL_NAME

    server = MCP_STREAMABLE_SERVER_NAME
    if server not in mcp_runtime.enabled_servers():
        return [], None

    primary = MCP_STREAMABLE_TOOL_NAME
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    for t in ("search_engine", "brave_web_search", "web_search", "brave_search"):
        if t not in candidates:
            candidates.append(t)

    n = max(1, min(int(max_results), 20))
    arg_sets: list[dict[str, Any]] = [
        {"query": q, "count": n},
        {"query": q},
        {"q": q, "count": n},
        {"q": q},
    ]
    last_err: str | None = None
    for tname in candidates:
        for args in arg_sets:
            try:
                res = mcp_runtime.call_tool(server, tname, args)
            except Exception as e:
                last_err = f"{tname}: {e!s}"[:280]
                continue
            if getattr(res, "isError", False):
                body = _mcp_call_tool_result_text(res)
                last_err = (body or f"{tname} isError").strip()[:280]
                continue
            body = _mcp_call_tool_result_text(res)
            rows = _parse_streamable_mcp_search_text(body, max_results=max_results)
            if rows:
                return rows[:max_results], None
            last_err = f"{tname} 返回空或可解析结果"
    return [], last_err


def search_web_snippets(
    query: str, *, max_results: int = 5
) -> tuple[list[dict[str, Any]], str | None]:
    """返回 (结果列表, 错误说明)。

    - 结果非空时，第二项为 None。
    - 结果为空时，第二项为给用户/日志看的原因。
    """

    q = (query or "").strip()
    if not q:
        return [], "查询为空。"

    from config import (
        RAG_WEB_BACKEND,
        RAG_WEB_BRAVE_API_KEY,
        RAG_WEB_DDG_BACKENDS,
        RAG_WEB_SEARXNG_URL,
        RAG_WEB_STREAMABLE_FIRST,
    )

    err: str | None = None
    err_parts: list[str] = []

    if RAG_WEB_BACKEND == "ddg_only":
        rows, derr = _search_duckduckgo_multi(
            q, max_results=max_results, backends=RAG_WEB_DDG_BACKENDS
        )
        if rows:
            return rows, None
        return [], derr or "DuckDuckGo 无结果。"

    if RAG_WEB_BACKEND == "auto":
        if RAG_WEB_STREAMABLE_FIRST:
            rows_s, snote = _search_mcp_streamable_web(q, max_results=max_results)
            if rows_s:
                return rows_s, None
            if snote:
                err_parts.append(f"MCP Streamable：{snote}")
        rows_m, mnote = _search_mcp_brave_web(q, max_results=max_results)
        if rows_m:
            return rows_m, None
        if mnote:
            err_parts.append(f"MCP Brave：{mnote}")

    # 1) DuckDuckGo 多后端
    rows, err = _search_duckduckgo_multi(
        q, max_results=max_results, backends=RAG_WEB_DDG_BACKENDS
    )
    if rows:
        return rows, None
    if err:
        err_parts.append(err)

    # 2) SearXNG（用户自建或可信实例）
    if RAG_WEB_SEARXNG_URL:
        rows, serr = _search_searxng(RAG_WEB_SEARXNG_URL, q, max_results=max_results)
        if rows:
            return rows, None
        if serr:
            err_parts.append(f"SearXNG：{serr}")

    # 3) Brave Search API（进程内直连，与 MCP 使用同一 Key 时可作最后兜底）
    if RAG_WEB_BRAVE_API_KEY:
        rows, berr = _search_brave(
            RAG_WEB_BRAVE_API_KEY, q, max_results=max_results
        )
        if rows:
            return rows, None
        if berr:
            err_parts.append(f"Brave API：{berr}")

    merged = "\n".join(err_parts).strip()
    if merged:
        return [], merged
    return [], (
        "未配置可用联网渠道。可：① MCP Streamable（MCP_STREAMABLE_URL + MCP_ENABLED 含 mcpmarket）；"
        "② MCP search（BRAVE_API_KEY + MCP_ENABLED 含 search）；"
        "③ DuckDuckGo；④ RAG_WEB_SEARXNG_URL；⑤ RAG_WEB_BRAVE_API_KEY。详见 .env.*.example。"
    )


def search_web_with_subquestions(
    question: str,
    llm: Any,
    *,
    max_merged_results: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """与本地检索对齐：在开启子问题拆分时，对每个子问题分别联网搜索，按 URL 去重后合并。

    每条结果带 ``subquestion`` 字段，便于 Prompt 里标注来源子问题并综合回答。
    """

    import time

    from config import (
        RAG_MAX_SUBQUESTIONS,
        RAG_SUBQUESTION_SPLIT,
        RAG_WEB_MERGED_MAX_RESULTS,
        RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE,
        RAG_WEB_SUBQUERY_DELAY_SEC,
    )
    from tools.agent.temporal_context import expand_query_for_web_search
    from tools.rag.language import split_compound_question

    cap = max_merged_results if max_merged_results is not None else RAG_WEB_MERGED_MAX_RESULTS
    cap = max(3, min(25, cap))
    q = (question or "").strip()
    if not q:
        return [], "查询为空。"

    use_split = RAG_SUBQUESTION_SPLIT
    if RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE is not None:
        use_split = RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE

    if use_split:
        subs = split_compound_question(q, llm)
    else:
        subs = [q]
    subs = subs[:RAG_MAX_SUBQUESTIONS]
    n = max(1, len(subs))
    # 每个子问题请求的条数：保证合并后大致能填满 cap
    per = max(2, min(5, (cap + n - 1) // n))

    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    errs: list[str] = []

    for idx, sq in enumerate(subs):
        sq = (sq or "").strip()
        if not sq:
            continue
        if idx > 0 and RAG_WEB_SUBQUERY_DELAY_SEC > 0:
            time.sleep(RAG_WEB_SUBQUERY_DELAY_SEC)
        wq = expand_query_for_web_search(sq)
        rows, err = search_web_snippets(wq, max_results=per)
        if err and not rows:
            errs.append(err[:220])
        for row in rows:
            url = (row.get("url") or "").strip()
            key = url.lower().split("?")[0].rstrip("/") if url else ""
            if not key:
                key = f"t:{(row.get('title') or '')[:120]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            item = dict(row)
            item["subquestion"] = sq
            merged.append(item)
            if len(merged) >= cap:
                break
        if len(merged) >= cap:
            break

    if merged:
        return merged[:cap], None
    if errs:
        return [], "所有子问题联网检索均未返回可用摘要。" + " | ".join(errs[:4])[:900]
    return [], "联网检索无结果。"


def web_items_to_document_pairs(
    web_items: list[dict[str, Any]],
    *,
    score_base: float = 0.2,
    meta_type: str = "web_search",
) -> list[tuple[Any, float]]:
    """将联网摘要列表转为 (Document, score) 列表，score 递增以便排在本地片段之后。"""

    from langchain_core.documents import Document

    pairs: list[tuple[Any, float]] = []
    for i, item in enumerate(web_items):
        snippet = (item.get("snippet") or "").strip()
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "web").strip()
        subq = (item.get("subquestion") or "").strip()
        body = snippet or title
        if not body:
            continue
        if title and title not in body:
            body = f"{title}\n{body}"
        if subq:
            body = f"【联网补充 · 检索查询：{subq}】\n{body}"
        pairs.append(
            (
                Document(
                    page_content=body[:8000],
                    metadata={
                        "source": url,
                        "type": meta_type,
                        "title": title,
                        "subquestion": subq or None,
                    },
                ),
                float(score_base) + i * 0.02,
            )
        )
    return pairs


def search_web_from_query_list(
    queries: list[str],
    *,
    max_merged_results: int = 8,
    max_per_query: int = 4,
) -> tuple[list[dict[str, Any]], str | None]:
    """按多条查询分别搜索，按 URL 去重合并（用于 LLM 评判后给出的 web_queries）。"""

    import time
    from concurrent.futures import ThreadPoolExecutor

    from config import RAG_WEB_SUBQUERY_DELAY_SEC, RAG_WEB_SUBQUERY_MAX_CONCURRENT
    from tools.agent.temporal_context import expand_query_for_web_search

    cap = max(3, min(20, max_merged_results))
    qs = [((q or "").strip()) for q in queries if (q or "").strip()][:5]
    if not qs:
        return [], "未提供联网搜索查询。"

    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    errs: list[str] = []

    def _run_one(sq: str) -> tuple[str, list[dict[str, Any]], str | None]:
        wq = expand_query_for_web_search(sq)
        rows, err = search_web_snippets(wq, max_results=max_per_query)
        return sq, list(rows or []), err

    if RAG_WEB_SUBQUERY_MAX_CONCURRENT > 1:
        max_w = min(RAG_WEB_SUBQUERY_MAX_CONCURRENT, len(qs))
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            batches = list(ex.map(_run_one, qs))
        for sq, rows, err in batches:
            if err and not rows:
                errs.append(err[:180])
            for row in rows:
                url = (row.get("url") or "").strip()
                key = url.lower().split("?")[0].rstrip("/") if url else ""
                if not key:
                    key = f"t:{(row.get('title') or '')[:100]}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                item = dict(row)
                item["subquestion"] = sq
                merged.append(item)
                if len(merged) >= cap:
                    break
            if len(merged) >= cap:
                break
    else:
        for idx, sq in enumerate(qs):
            if idx > 0 and RAG_WEB_SUBQUERY_DELAY_SEC > 0:
                time.sleep(RAG_WEB_SUBQUERY_DELAY_SEC)
            _sq, rows, err = _run_one(sq)
            if err and not rows:
                errs.append(err[:180])
            for row in rows:
                url = (row.get("url") or "").strip()
                key = url.lower().split("?")[0].rstrip("/") if url else ""
                if not key:
                    key = f"t:{(row.get('title') or '')[:100]}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                item = dict(row)
                item["subquestion"] = sq
                merged.append(item)
                if len(merged) >= cap:
                    break
            if len(merged) >= cap:
                break

    if merged:
        return merged[:cap], None
    if errs:
        return [], "联网补充检索失败：" + " | ".join(errs[:3])[:800]
    return [], "联网补充检索无结果。"
