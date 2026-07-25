"""
RAGAgent: 检索 + 上下文拼接 + 调用 LLM 生成回答。

支持：1）规则路由 + RAG；2）LangChain Tools（tool_weather / tool_search_arxiv / tool_search_knowledge）；
3）Middleware 在 before_agent / before_model / after_model / after_agent 插入逻辑。
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from pathlib import Path

from config import (
    ARXIV_INGEST_DISAMBIGUATION_FETCH_MAX,
    ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC,
    RAG_MAX_IMAGES_PER_MESSAGE,
    RAG_PUBLIC_NAMESPACE,
)
from langchain_core.documents import Document
from models_qwen import qwen
from prompts import (
    build_rag_prompt,
    MULTI_QUESTION_REASONING_TEMPLATE,
    TOOLS_FINAL_ANSWER_TEMPLATE,
)
from tools.rag.multimodal_content import (
    build_openai_multimodal_user_content,
    dict_messages_to_lc,
)
from tools.agent.agent_tools import (
    _normalize_arxiv_id,
    get_agent_tools,
    tool_search_arxiv,
    tool_search_knowledge,
    tool_weather,
)
from tools.agent.arxiv_search import (
    get_arxiv_id,
    search_arxiv,
    search_arxiv_for_ingest_disambiguation,
)
from tools.agent.conversation import (
    collect_session_embed_ids_for_namespace,
    conversation_manager,
)
from tools.agent.approvals import create_approval
from tools.rag.knowledge import NamespaceVectorStore, vector_store
from tools.rag.language import expand_retrieval_queries
from tools.rag.math_utils import merge_ranked_lists
from tools.rag.retrieval_merge import retrieve_with_public_merge
from tools.agent.middleware import (
    AgentMiddleware,
    LoggingMiddleware,
    run_after_agent,
    run_after_model,
    run_before_agent,
    run_before_model,
    trace_event,
)
from tools.agent.router import Route, allow_web_search_when_local_misses
from tools.agent.router import extract_arxiv_id, paper_namespace_arxiv_id
from tools.agent.router import strip_forced_web_search_phrases, user_requests_forced_web_search
from tools.retrieval.query_router import build_query_route
from tools.agent.temporal_context import (
    expand_query_for_web_search,
    format_temporal_system_note,
    looks_like_live_schedule_query,
    needs_temporal_anchor,
)
from tools.agent.weather import get_weather
from tools.agent.planner import make_plan
from tools.storage.long_memory import add_memory, format_memories, retrieve_memories


class RAGAgent:
    """一个极简的 RAG Agent 封装。

    用于展示完整链路：
    用户问题 -> 查询策略 -> 检索 -> Prompt 组装 -> ChatCompletion
    可选：使用 LangChain Tools 做天气/论文/知识库调用；使用 Middleware 做日志/钩子。
    """

    def __init__(
        self,
        store: NamespaceVectorStore | None = None,
        model_name: str | None = None,
        middleware: list[AgentMiddleware] | None = None,
    ) -> None:
        self.store = store or vector_store
        self.llm = qwen
        self.middleware: list[AgentMiddleware] = middleware if middleware is not None else []

    def _build_delete_approval(self, path_value: str, session_id: str | None) -> dict:
        """删除文件前创建人工审批单（Human-in-the-loop）。"""
        item = create_approval(
            session_id=session_id,
            tool_name="tool_delete_file",
            tool_args={"path": path_value},
            allowed_decisions=["approve", "edit", "reject"],
        )
        return {
            "answer": (
                "需要人工确认后才能执行该删除操作。\n"
                f"- approval_id: {item.approval_id}\n"
                "- tool: tool_delete_file\n"
                f"- args: {{'path': '{path_value}'}}\n"
                "- allowed_decisions: ['approve', 'edit', 'reject']\n"
            ),
            "citations": [],
            "approval_required": {
                "approval_id": item.approval_id,
                "tool": "tool_delete_file",
                "args": {"path": path_value},
                "allowed_decisions": ["approve", "edit", "reject"],
            },
        }

    # 全角括号内漏写 $ 时，勿把排版命令当数学（避免误包 \cite 等）
    _NON_MATH_TEX_CMDS = frozenset(
        "cite citet citep citeauthor citeyear citealp citealt "
        "ref label bibitem emph footnote url href includegraphics "
        "subfigure figure table autoref eqref pageref".split()
    )
    _MATH_TEX_CMDS = frozenset(
        "mathcal mathbf mathrm sum prod frac cdot times log ln exp "
        "sin cos tan alpha beta gamma delta epsilon theta lambda mu "
        "sigma phi omega partial nabla infty leq geq left right "
        "begin end".split()
    )

    @staticmethod
    def _latex_cmd_at_is_probably_math(ln: str, j: int) -> bool:
        """ln[j] 为反斜杠时，判断后续是否为常见数学命令（非 cite/ref 等）。"""
        if j >= len(ln) or ln[j] != "\\":
            return False
        m = re.match(r"\\([a-zA-Z@]+)", ln[j:])
        if not m:
            return False
        cmd = m.group(1).lower()
        if cmd in RAGAgent._NON_MATH_TEX_CMDS:
            return False
        return True

    @staticmethod
    def _normalize_cjk_parenthesis_latex_line(ln: str) -> str:
        """修复中文全角括号与行内公式混排时常见的 `$` 漏写。
        - `（$\\mathcal{L}\\mathrm{n}）` → 在 `）` 前补 `$`
        - `（\\mathcal{L}\\mathrm{pnormal}...$）` → 在 `（` 后补起始 `$`
        """
        n = len(ln)
        i = 0
        block_open = False
        inline_open = False
        parts: list[str] = []
        CJK_OPEN = "\uFF08"
        CJK_CLOSE = "\uFF09"

        while i < n:
            ch = ln[i]
            if ch == "\\":
                parts.append(ln[i : i + 2] if i + 1 < n else ch)
                i += 2
                continue
            if ch == "$" and i + 1 < n and ln[i + 1] == "$":
                parts.append("$$")
                block_open = not block_open
                i += 2
                continue
            if ch == "$":
                if not block_open:
                    inline_open = not inline_open
                parts.append("$")
                i += 1
                continue
            if not block_open and inline_open and ch in (CJK_CLOSE, ")"):
                if not (parts and parts[-1].endswith("$")):
                    parts.append("$")
                inline_open = False
                parts.append(ch)
                i += 1
                continue
            if not block_open and not inline_open and ch == CJK_OPEN:
                j = i + 1
                while j < n and ln[j] in " \t":
                    j += 1
                if j < n and ln[j] == "$":
                    parts.append(ch)
                    i += 1
                    continue
                if j < n and RAGAgent._latex_cmd_at_is_probably_math(ln, j):
                    parts.append(CJK_OPEN + "$")
                    inline_open = True
                    i += 1
                    continue
                parts.append(ch)
                i += 1
                continue
            parts.append(ch)
            i += 1
        return "".join(parts)

    @staticmethod
    def _normalize_cjk_parenthesis_latex(text: str) -> str:
        """对非代码围栏行应用 `_normalize_cjk_parenthesis_latex_line`。"""
        s = str(text or "")
        if not s:
            return s
        lines = s.splitlines(keepends=True)
        out: list[str] = []
        in_fence = False
        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                out.append(ln)
                continue
            if in_fence:
                out.append(ln)
                continue
            out.append(RAGAgent._normalize_cjk_parenthesis_latex_line(ln))
        return "".join(out)

    @staticmethod
    def _wrap_bare_latex_math_spans_line(ln: str) -> str:
        """
        把未被 `$...$` 包裹的裸数学 LaTeX 片段自动包裹。
        仅处理常见数学命令，避免误包 `\\cite` 等文本命令。
        """
        n = len(ln)
        i = 0
        block_open = False
        inline_open = False
        out: list[str] = []
        stop_chars = "，。；：,.!?！？\n"
        while i < n:
            ch = ln[i]
            if ch == "\\":
                if i + 1 < n and ln[i + 1] == "\\":
                    out.append("\\\\")
                    i += 2
                    continue
                m = re.match(r"\\([a-zA-Z@]+)", ln[i:])
                if m and not block_open and not inline_open:
                    cmd = m.group(1).lower()
                    if cmd in RAGAgent._MATH_TEX_CMDS and cmd not in RAGAgent._NON_MATH_TEX_CMDS:
                        j = i + len(m.group(0))
                        brace = 0
                        while j < n:
                            c = ln[j]
                            if c == "\\" and j + 1 < n:
                                j += 2
                                continue
                            if c == "{":
                                brace += 1
                                j += 1
                                continue
                            if c == "}":
                                brace = max(0, brace - 1)
                                j += 1
                                continue
                            if brace == 0 and c in stop_chars:
                                break
                            if brace == 0 and c in "）)]" and j + 1 < n and ln[j + 1] in stop_chars + "）)]":
                                break
                            j += 1
                        span = ln[i:j].strip()
                        if span:
                            # 片段本身若混入孤立 `$`，包裹前先清理，避免产生嵌套 `$...$...$`
                            span = span.replace("$", "")
                            out.append("$" + span + "$")
                            i = j
                            continue
                out.append(ch)
                i += 1
                continue
            if ch == "$" and i + 1 < n and ln[i + 1] == "$":
                block_open = not block_open
                out.append("$$")
                i += 2
                continue
            if ch == "$":
                if not block_open:
                    inline_open = not inline_open
                out.append("$")
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _wrap_bare_latex_math_spans(text: str) -> str:
        """对非代码围栏文本逐行包裹裸数学 LaTeX 片段。"""
        s = str(text or "")
        if not s:
            return s
        lines = s.splitlines(keepends=True)
        out: list[str] = []
        in_fence = False
        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                out.append(ln)
                continue
            if in_fence:
                out.append(ln)
                continue
            out.append(RAGAgent._wrap_bare_latex_math_spans_line(ln))
        return "".join(out)

    @staticmethod
    def _repair_unbalanced_math_delimiters(text: str) -> str:
        """
        修复未闭合的 LaTeX 分隔符，避免 Markdown 渲染失败。
        规则：
        - 先修复全角括号与 `$...$` 混排时漏写的分隔符
        - 忽略代码围栏 ```...``` 内的内容
        - 统计未转义的 `$` / `$$` 开闭状态
        - 若末尾仍未闭合，则自动补齐闭合符
        """
        s = RAGAgent._wrap_bare_latex_math_spans(str(text or ""))
        s = RAGAgent._normalize_cjk_parenthesis_latex(s)
        # 清理明显误植的孤立 `$`（常见于复制公式时混入）
        s = re.sub(r"(?<=\d)\$(?=）)", "", s)
        s = re.sub(r"(?<=[A-Za-z0-9_}])\$(?=\))", "", s)
        if not s:
            return s
        lines = s.splitlines(keepends=True)
        in_fence = False
        inline_open = False
        block_open = False

        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            i = 0
            n = len(ln)
            while i < n:
                ch = ln[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch != "$":
                    i += 1
                    continue
                # $$ block
                if i + 1 < n and ln[i + 1] == "$":
                    block_open = not block_open
                    i += 2
                    continue
                # $ inline
                if not block_open:
                    inline_open = not inline_open
                i += 1

        fixed = s
        if block_open:
            fixed = fixed.rstrip() + "\n$$"
        if inline_open:
            fixed = fixed.rstrip() + "$"
        return fixed

    def _with_truncation_reason(self, resp_obj: Any, answer_text: str) -> str:
        """若模型因长度截断，给用户明确补充原因与续问建议。"""
        text = str(answer_text or "")
        finish_reason = ""
        try:
            meta = getattr(resp_obj, "response_metadata", {}) or {}
            if isinstance(meta, dict):
                finish_reason = str(
                    meta.get("finish_reason")
                    or meta.get("stop_reason")
                    or ""
                ).lower()
        except Exception:
            finish_reason = ""
        truncated = finish_reason in ("length", "max_tokens", "token_limit")
        # 兜底启发：末尾明显断句且文本较长
        if not truncated and len(text) > 900 and not re.search(r"[。！？.!?]\s*$", text):
            truncated = True
        if not truncated:
            return RAGAgent._repair_unbalanced_math_delimiters(text)
        note = (
            "\n\n---\n"
            "补充说明：以上回答可能被长度限制截断（触发了模型输出上限或长上下文压缩），"
            "因此后半部分未完整展开。你可以继续追问“请继续从上文中断处往下讲”，"
            "或让我按“方法/实验/结论”分段输出以避免再次截断。"
        )
        if note.strip() in text:
            return RAGAgent._repair_unbalanced_math_delimiters(text)
        return RAGAgent._repair_unbalanced_math_delimiters(text + note)

    def _retrieve_local(
        self,
        question: str,
        namespace: str,
        strategy: str,
        k: int,
        score_threshold: float,
        session_ingest_ids: list[str] | None = None,
        paper_intent_hint: bool = False,
    ) -> List[Tuple[object, float]]:
        from config import RAG_LAYERED_PAPER_RETRIEVAL
        from tools.retrieval.paper_retriever import layered_paper_retrieve

        ns = (namespace or "").strip()
        use_layered = bool(RAG_LAYERED_PAPER_RETRIEVAL) and (
            bool(paper_intent_hint) or ns.startswith("paper:")
        )
        if use_layered:
            return layered_paper_retrieve(
                self.store,
                question=question,
                namespace=namespace,
                strategy=strategy,
                k=k,
                score_threshold=score_threshold,
                session_ingest_ids=session_ingest_ids,
                llm=self.llm,
                use_layered=True,
            )
        queries = expand_retrieval_queries(
            question,
            strategy=strategy,
            llm=self.llm,
        )
        if not queries:
            fb = (question or "").strip()
            queries = [fb] if fb else []
        return retrieve_with_public_merge(
            self.store,
            queries=queries,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
        )

    @staticmethod
    def _resolve_user_image_paths(user_image_paths: list[str] | None) -> list[Path]:
        """把可能的相对路径解析为可读的绝对文件路径。"""
        if not user_image_paths:
            return []
        pr = Path.cwd()
        out: list[Path] = []
        for s in user_image_paths:
            p = Path(s)
            if not p.is_absolute():
                p = pr / p
            try:
                p = p.resolve()
            except OSError:
                continue
            if p.is_file():
                out.append(p)
        return out

    def _judge_use_user_images(
        self,
        question: str,
        grouped: List[Tuple[object, float]],
        user_image_paths: list[str] | None,
    ) -> bool:
        """用 LLM 判定：用户图片是否与“本地检索到的数据库上下文”直接相关。

        返回 True：允许在最终回答中使用“数据库上下文 + 用户图片”
        返回 False：数据库上下文与图片不相关，最终改为“只基于图片回答”（不使用数据库上下文）
        """
        from config import (
            RAG_IMAGE_RELEVANCE_JUDGE_ENABLED,
            RAG_IMAGE_RELEVANCE_JUDGE_CONTEXT_MAX_CHARS,
            RAG_IMAGE_RELEVANCE_JUDGE_MAX_IMAGES,
        )

        if not RAG_IMAGE_RELEVANCE_JUDGE_ENABLED:
            return False

        resolved_imgs = self._resolve_user_image_paths(user_image_paths)
        if not resolved_imgs:
            return False

        # 构造给裁判用的简短上下文（只做裁决，不用于最终回答）
        ctx_parts: list[str] = []
        used = 0
        max_chars = int(RAG_IMAGE_RELEVANCE_JUDGE_CONTEXT_MAX_CHARS or 0)
        for doc, score in grouped[:12]:
            raw = getattr(doc, "page_content", str(doc)) or ""
            content = raw.strip()
            if not content:
                continue
            meta = getattr(doc, "metadata", {}) or {}
            source = meta.get("source", "unknown")
            block = f"[来源: {source}, 相关度: {float(score):.3f}]\n{content}"
            if max_chars > 0 and used + len(block) > max_chars:
                block = block[: max(0, max_chars - used)] + "\n\n...（已截断用于裁决）"
            ctx_parts.append(block)
            used += len(block)
            if max_chars > 0 and used >= max_chars:
                break
        context_text = "\n\n---\n\n".join(ctx_parts) if ctx_parts else "未检索到上下文。"

        judge_system = (
            "你是一个“图像-文本证据相关性裁判”。"
            "用户会提供一张或多张图片，以及一个问题；并提供数据库检索到的上下文片段。"
            "你的任务是判断：用户图片中的关键信息（例如图表、公式、截图内容）是否与上述数据库上下文直接相关，"
            "能否为最终回答提供补充证据（例如同一张图、同一实验结果、同一页面内容、同一概念的可视化）。"
            "规则："
            "1）只基于“问题 + 数据库上下文 + 用户图片”做判断，不要编造证据。"
            "2）当图片内容看起来与数据库上下文无关，或无法建立直接对应关系时，use_user_images=false。"
            "3）输出严格 JSON（不要代码块、不要解释）："
            '{"use_user_images": true/false, "reason": "简短原因"}'
        )

        judge_user_text = (
            f"用户问题：{question}\n\n"
            "数据库检索上下文（用于判断图片是否相关）：\n"
            f"{context_text}\n\n"
            "请判断：用户图片是否与上述上下文直接相关？"
        )

        judge_paths = resolved_imgs[: int(RAG_IMAGE_RELEVANCE_JUDGE_MAX_IMAGES or 4)]
        judge_content = build_openai_multimodal_user_content(
            judge_user_text,
            judge_paths,
            max_images=min(len(judge_paths), 16),
        )
        judge_messages: list[dict[str, Any]] = [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": judge_content},
        ]

        resp = self.llm.invoke(dict_messages_to_lc(judge_messages))
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = raw.strip()

        # 容错：优先抓取第一个 JSON 对象片段
        import json
        import re

        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                use_flag = bool(data.get("use_user_images"))
                return use_flag
            except Exception:
                pass

        # fallback：用关键字启发式
        low = raw.lower()
        if "false" in low:
            return False
        if "true" in low:
            return True
        return False

    @staticmethod
    def _parse_knowledge_tool_citations(content: str) -> list[dict]:
        """从 tool_search_knowledge 返回的文本中解析出 [来源: X, 相关度: Y] 与内容，生成 citations。"""
        import re
        citations = []
        pattern = re.compile(r"\[来源:\s*([^,]+),\s*相关度:\s*([\d.]+)\]\s*\n(.*)", re.DOTALL)
        for part in content.split("\n\n---\n\n"):
            part = part.strip()
            if not part:
                continue
            m = pattern.match(part)
            if m:
                source, score_str, text = m.group(1).strip(), m.group(2), m.group(3)
                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.0
                citations.append({
                    "source": source,
                    "score": score,
                    "preview": (text or "")[:200],
                })
        return citations

    def _group_by_parent(
        self, retrieved: List[Tuple[object, float]]
    ) -> List[Tuple[object, float]]:
        """将同一个 parent_id 的 chunk 聚合为父文档级别，并按 chunk_index 拼接上下文。"""
        grouped: dict[str, list[Tuple[object, float]]] = defaultdict(list)
        singletons: list[Tuple[object, float]] = []

        for doc, score in retrieved:
            meta = getattr(doc, "metadata", {}) or {}
            parent_id = meta.get("parent_id")
            if parent_id:
                grouped[str(parent_id)].append((doc, float(score)))
            else:
                singletons.append((doc, float(score)))

        merged: list[Tuple[object, float]] = []
        for parent_id, items in grouped.items():
            items_sorted = sorted(
                items,
                key=lambda x: (
                    (getattr(x[0], "metadata", {}) or {}).get("chunk_index")
                    if isinstance((getattr(x[0], "metadata", {}) or {}).get("chunk_index"), int)
                    else 10**9
                ),
            )
            best_score = min(s for _, s in items_sorted)
            base_meta = dict(getattr(items_sorted[0][0], "metadata", {}) or {})
            parts = []
            for d, _s in items_sorted:
                t = (getattr(d, "page_content", "") or "").strip()
                if t:
                    parts.append(t)
            merged_doc = Document(
                page_content="\n\n".join(parts),
                metadata=base_meta,
            )
            merged.append((merged_doc, best_score))

        merged.extend(singletons)
        return sorted(merged, key=lambda x: x[1])

    def _retrieve_multi_source(
        self,
        question: str,
        namespace: str,
        k: int,
        score_threshold: float,
        session_ingest_ids: list[str] | None = None,
    ) -> List[Tuple[object, float]]:
        """示例性的多源检索：本地向量库 + arXiv 摘要。"""

        local_results = self._retrieve_local(
            question=question,
            namespace=namespace,
            strategy=DEFAULT_RETRIEVAL_STRATEGY,
            k=k,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids,
        )

        arxiv_papers = search_arxiv(question, max_results=k)
        arxiv_results: list[Tuple[object, float]] = [
            (paper, 0.1 * (idx + 1)) for idx, paper in enumerate(arxiv_papers)
        ]

        merged = merge_ranked_lists([local_results, arxiv_results])
        return merged[:k]

    @staticmethod
    def _is_history_query(question: str) -> bool:
        q = (question or "").lower()
        keys = [
            "上周",
            "之前",
            "历史",
            "聊过",
            "问过",
            "我说过",
            "你说过",
            "对话里",
            "history",
            "previous chat",
        ]
        return any(k in q for k in keys)

    @staticmethod
    def _is_assistant_meta_question(question: str) -> bool:
        """身份/模型/开发者等元问题，不应走「仅从 RAG 片段取证」。"""
        raw = (question or "").strip()
        if not raw or len(raw) > 120:
            return False
        q = raw.lower()
        keys_zh = [
            "你是谁",
            "你是什么",
            "你是什么模型",
            "你用的什么模型",
            "你叫什",
            "什么模型",
            "哪个模型",
            "哪家公司",
            "谁开发的",
            "你是gpt",
            "你是chatgpt",
            "你是claude",
            "你是千问",
            "你是qwen",
            "你是通义",
        ]
        keys_en = [
            "who are you",
            "what are you",
            "what model",
            "which model",
            "your name",
        ]
        return any(k in q for k in keys_zh) or any(k in q for k in keys_en)

    def _answer_assistant_meta_question(
        self,
        question: str,
        session_id: str | None,
        user_image_paths: list[str] | None = None,
    ) -> dict:
        """身份/模型等元问题：不走向量检索，直接调用 LLM；可配置回落为固定文案。"""
        from config import ASSISTANT_IDENTITY_REPLY, ASSISTANT_META_USE_LLM

        if not ASSISTANT_META_USE_LLM:
            answer = ASSISTANT_IDENTITY_REPLY
        else:
            sys_msg = (
                "用户在询问关于你自己的身份、名称、开发方、底层大模型，或本应用（论文检索助手 "
                "PaperSearchAssistant）能做什么。\n"
                "请用与用户问题一致的语言（中文提问则用中文）简洁、诚实回答。\n"
                "可参考的事实：本应用侧重本地论文库与向量知识库（RAG）问答；对话模型通过通义千问兼容 "
                "OpenAI 协议的 API 接入，具体模型名以部署配置（如 models_qwen.py）为准。\n"
                "禁止说「无法从提供的上下文得知」「检索片段未包含」——此类问题不依赖知识库检索。"
            )
            pr = Path.cwd()
            path_objs: list[Path] = []
            for s in user_image_paths or []:
                p = Path(s)
                if not p.is_absolute():
                    p = pr / p
                if p.is_file():
                    path_objs.append(p.resolve())
            user_content = build_openai_multimodal_user_content(
                question,
                path_objs,
                max_images=RAG_MAX_IMAGES_PER_MESSAGE,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content},
            ]
            run_before_model(self.middleware, messages)
            config: dict[str, Any] = {}
            if self.middleware:
                try:
                    from tools.agent.middleware import AgentCallbackHandler

                    config["callbacks"] = [AgentCallbackHandler(self.middleware)]
                except Exception:
                    pass
            resp = self.llm.invoke(dict_messages_to_lc(messages), config=config)
            run_after_model(self.middleware, resp)
            answer = resp.content if hasattr(resp, "content") else str(resp)
            answer = self._with_truncation_reason(resp, str(answer))

        out: dict = {"answer": answer, "citations": []}
        if session_id:
            conversation_manager.add_turn(
                session_id, "user", question, image_paths=user_image_paths
            )
            conversation_manager.add_turn(session_id, "assistant", out["answer"])
        return out

    @staticmethod
    def _is_paper_coref_query(question: str) -> bool:
        q = (question or "").lower()
        keys = ["这篇论文", "该论文", "这篇文章", "this paper", "the paper", "it"]
        return any(k in q for k in keys)

    @staticmethod
    def _is_download_or_ingest_intent(question: str) -> bool:
        q = (question or "").lower()
        keys = [
            "下载",
            "入库",
            "保存",
            "download",
            "ingest",
            "embed",
            "存本地",
            "保存到本地",
            "全文入库",
            "拉取",
            "pull",
            "fetch",
            "加到库",
            "加入库",
        ]
        return any(k in q for k in keys)

    @staticmethod
    def _looks_like_paper_content_qa_not_ingest(question: str) -> bool:
        """判断用户是在「读论文 / 问方法结论」而非「下载入库 / 仅要 arXiv 候选列表」。

        否则包含「这篇论文」的追问会误触 `_maybe_handle_paper_ingest_dialogue` 里的
        title_like_download，整句被拿去 arXiv 搜标题，本地库 RAG 永远不会执行。
        """
        from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

        return looks_like_paper_content_qa(question or "")

    @staticmethod
    def _is_delete_intent(question: str) -> bool:
        q = (question or "").lower()
        keys = ["删除", "删掉", "移除", "清除", "delete", "remove"]
        return any(k in q for k in keys)

    @staticmethod
    def _is_session_paper_mirror_intent(ql: str) -> bool:
        """是否要求把论文只读复制到当前会话向量 namespace（与「下载入库」区分）。"""
        q = (ql or "").lower()
        if "attach paper" in q:
            return True
        if "镜像" in q and "会话" in q:
            return True
        if "复制到会话" in q or "复制到当前会话" in q:
            return True
        if "放进会话" in q or "放进当前会话" in q:
            return True
        pairs = (
            ("打开", "论文"),
            ("载入", "论文"),
            ("加载", "论文"),
            ("打开", "这篇"),
            ("载入", "这篇"),
            ("加载", "这篇"),
        )
        return any(a in q and b in q for a, b in pairs)

    @staticmethod
    def _is_paper_search_local_only(question: str) -> bool:
        q = (question or "").lower()
        keys = ["只查本地", "不要联网", "仅从本地", "仅查本地", "本地库里有没有", "本地已下载"]
        return any(k in q for k in keys)

    @staticmethod
    def _looks_like_paper_search_request(question: str) -> bool:
        q = (question or "").strip()
        ql = q.lower()
        if not q:
            return False
        # 「从本地库检索，说一说某篇论文的方法」里的「检索」是 RAG/阅读，不是「找几篇论文」清单任务
        if RAGAgent._looks_like_paper_content_qa_not_ingest(q):
            return False
        patterns = [
            r"(找|推荐|搜|检索).*(论文|paper|文献|arxiv)",
            r"(最新|最近|近期).*(论文|paper|work|arxiv)",
            r"(关于|方向).*(有哪些|有什么).*(论文|paper|work)",
            r"(recommend|find|search).*(paper|papers|arxiv)",
            r"(latest|recent).*(paper|papers|arxiv|work)",
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    @staticmethod
    def _is_paper_search_force_web(question: str) -> bool:
        q = (question or "").lower()
        keys = ["请联网", "联网搜索", "去 arxiv", "去arxiv", "online", "latest on arxiv"]
        return any(k in q for k in keys)

    @staticmethod
    def _is_latest_or_recent_paper_search(question: str) -> bool:
        q = (question or "").lower()
        keys = ["最新", "最近", "近期", "latest", "recent", "new"]
        return any(k in q for k in keys)

    @staticmethod
    def _paper_search_count_hint(question: str) -> int:
        q = (question or "").lower()
        if "两篇" in q or "2篇" in q or "两条" in q:
            return 2
        if "三篇" in q or "3篇" in q:
            return 3
        if "五篇" in q or "5篇" in q:
            return 5
        if "几篇" in q or "几条" in q:
            return 5
        m = re.search(r"(\d+)\s*篇", q)
        if m:
            try:
                return max(1, min(20, int(m.group(1))))
            except ValueError:
                return 5
        return 5

    @staticmethod
    def _paper_search_topic_query(question: str) -> str:
        """抽取论文搜索主题，去掉请求词，保留关键方向词。"""
        q = (question or "").strip()
        if not q:
            return q
        # 去掉常见指令性短语，避免污染检索关键词
        pats = [
            r"帮我找几篇",
            r"帮我找",
            r"推荐几篇",
            r"推荐",
            r"搜一下",
            r"找一下",
            r"找几篇",
            r"论文",
            r"文献",
            r"最新的?",
            r"最近的?",
            r"近期的?",
            r"请联网",
            r"联网搜索",
            r"只查本地",
            r"不要联网",
            r"关于",
        ]
        out = q
        for p in pats:
            out = re.sub(p, " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip()
        return out or q

    @staticmethod
    def _paper_search_expand_bilingual_keywords(topic: str) -> list[str]:
        """把中文主题扩展为中英双语关键词，提升 arXiv/本地英文库召回。"""
        base = (topic or "").strip()
        if not base:
            return []
        keys: list[str] = [base]

        # 常见术语的确定性映射（优先保证稳定）
        low = base.lower()
        if "3dgs" in low or "高斯泼溅" in base or "高斯喷溅" in base:
            keys.extend(["3D Gaussian Splatting", "Gaussian Splatting", "3DGS"])
        if "编辑" in base:
            keys.extend(["editing", "edit", "3d scene editing"])
        if "分割" in base:
            keys.extend(["segmentation", "scene segmentation"])
        if "重建" in base:
            keys.extend(["reconstruction", "novel view synthesis"])

        # LLM 翻译/术语规范化（失败则静默退化）
        try:
            prompt = (
                "将用户给出的论文主题短语改写为 1~3 个英文检索关键词短语，"
                "只输出 JSON：{\"keywords\": [\"...\"]}。"
                "要求：保留技术实体，不要解释。"
            )
            resp = qwen.invoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": base},
                ]
            )
            raw = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\{.*\}", str(raw).strip(), flags=re.DOTALL)
            if m:
                import json

                obj = json.loads(m.group(0))
                arr = obj.get("keywords") if isinstance(obj, dict) else None
                if isinstance(arr, list):
                    for x in arr:
                        s = str(x).strip()
                        if s:
                            keys.append(s)
        except Exception:
            pass

        # 去重并保序
        seen: set[str] = set()
        out: list[str] = []
        for k in keys:
            kk = k.strip()
            if not kk:
                continue
            lk = kk.lower()
            if lk in seen:
                continue
            seen.add(lk)
            out.append(kk)
        return out

    @staticmethod
    def _search_local_papers_for_query(question: str, limit: int) -> list[dict]:
        topic = RAGAgent._paper_search_topic_query(question)
        keyword_candidates = RAGAgent._paper_search_expand_bilingual_keywords(topic)
        try:
            from tools.storage.papers_db import list_papers as db_list_papers

            merged: list[dict] = []
            seen_ids: set[str] = set()
            query_list = keyword_candidates or ([topic] if topic else [question])
            for kw in query_list[:6]:
                rows = db_list_papers(
                    keyword=kw,
                    limit=max(1, min(50, int(limit))),
                    offset=0,
                )
                for r in list(rows or []):
                    rid = str(r.get("arxiv_id") or r.get("title") or "").strip().lower()
                    if not rid or rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    merged.append(r)
                    if len(merged) >= max(1, int(limit)):
                        return merged
            return merged
        except Exception:
            return []

    @staticmethod
    def _is_local_paper_search_result_sufficient(
        local_rows: list[dict],
        question: str,
        count_hint: int,
    ) -> bool:
        """本地结果充足性判断：数量 + 新近性（latest/recent 强要求）。"""
        rows = list(local_rows or [])
        if not rows:
            return False
        # 数量不足
        min_needed = 3 if count_hint >= 3 else count_hint
        if len(rows) < min_needed:
            return False
        # latest/recent 场景：若最前结果年份过旧（早于 2023）则判不足
        if RAGAgent._is_latest_or_recent_paper_search(question):
            years: list[int] = []
            for r in rows[: min(5, len(rows))]:
                pub = str(r.get("published") or "")
                mm = re.search(r"(19|20)\d{2}", pub)
                if mm:
                    try:
                        years.append(int(mm.group(0)))
                    except ValueError:
                        pass
            if years and max(years) < 2023:
                return False
        return True

    @staticmethod
    def _format_paper_search_result_text(
        *,
        question: str,
        local_rows: list[dict],
        web_text: str | None,
        used_web: bool,
    ) -> str:
        lines: list[str] = []
        lines.append("论文检索结果（本地优先，结果不足时已自动联网补充）：")
        if local_rows:
            lines.append("")
            lines.append("【本地库】")
            for i, r in enumerate(local_rows[:8], start=1):
                title = str(r.get("title") or "（无标题）").strip()
                authors = ", ".join(list(r.get("authors") or [])[:4])
                pub = str(r.get("published") or "").strip()
                aid = str(r.get("arxiv_id") or "").strip()
                pdf = str(r.get("pdf_path") or "").strip()
                indexed = bool(r.get("indexed", True))
                lines.append(f"{i}. {title}")
                if authors:
                    lines.append(f"   作者: {authors}")
                if pub:
                    lines.append(f"   时间: {pub}")
                if aid:
                    lines.append(f"   arXiv: {aid}")
                if pdf:
                    lines.append(f"   本地PDF: {pdf}")
                lines.append(f"   来源: local | 已建索引: {'是' if indexed else '否'}")
        else:
            lines.append("")
            lines.append("【本地库】未命中相关论文。")
        if used_web and web_text:
            lines.append("")
            lines.append("【联网补充(arXiv)】")
            lines.append(web_text.strip())
        lines.append("")
        lines.append("提示：若你只想看本地结果，可说“只查本地”。")
        return "\n".join(lines)

    @staticmethod
    def _count_numbered_items(text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"(?m)^\s*\d+\.\s+", str(text)))

    @staticmethod
    def _paper_context_is_thin(grouped: list[tuple[object, float]]) -> bool:
        """判断论文上下文是否过薄：只命中摘要/图注，缺少正文 generic chunk。"""
        if not grouped:
            return True
        generic_hits = 0
        total_len = 0
        for doc, _sc in grouped[:12]:
            meta = getattr(doc, "metadata", {}) or {}
            role = str(meta.get("chunk_role") or "").lower()
            typ = str(meta.get("type") or "").lower()
            txt = str(getattr(doc, "page_content", "") or "")
            total_len += len(txt)
            if role == "generic":
                generic_hits += 1
            elif typ in ("arxiv_abstract", "pdf_figure", "figure_summary", "table_summary"):
                continue
        if generic_hits >= 2:
            return False
        # 文本总量过小且几乎无正文，通常会触发“上下文缺失说明”
        return total_len < 9000

    @staticmethod
    def _question_wants_method(question: str) -> bool:
        q = (question or "").lower()
        # “实验是怎么做的/实验设置”更偏实验设计，不应触发 method fallback。
        if re.search(r"(实验.*怎么做|怎么做.*实验|实验设置|实验方案|evaluation setup|experimental setup)", q, re.I):
            return False
        if re.search(r"(实验|结果|evaluation|experiment)", q, re.I) and not re.search(
            r"(方法|method|approach|architecture|模型结构|技术路线|第\s*3\s*节|第三节)",
            q,
            re.I,
        ):
            return False
        return bool(
            re.search(
                r"(method|approach|architecture|model|方法|方法部分|方法论|怎么做|技术路线|模型结构|第\s*3\s*节|第三节)",
                q,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _has_method_evidence(grouped: list[tuple[object, float]]) -> bool:
        for doc, _sc in grouped[:20]:
            meta = getattr(doc, "metadata", {}) or {}
            role = str(meta.get("chunk_role") or "").lower()
            heading = str(meta.get("section_title") or meta.get("heading") or "").lower()
            txt = str(getattr(doc, "page_content", "") or "")[:1200].lower()
            if re.search(r"(method|approach|architecture|model|方法|算法)", heading):
                return True
            if role == "generic" and re.search(
                r"(we propose|our method|approach|architecture|方法|提出了|模型由)",
                txt,
                re.IGNORECASE,
            ):
                return True
        return False

    @staticmethod
    def _pdf_path_for_arxiv_id(arxiv_id: str) -> str:
        normalized = re.sub(r"v\d+$", "", str(arxiv_id or "").strip(), flags=re.IGNORECASE)
        return f"data/papers/{normalized}.pdf"

    @staticmethod
    def _extract_last_arxiv_id_from_history(messages: list[dict]) -> str | None:
        patterns = [
            re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE),
            re.compile(r"\b[a-z\-]+\/\d{7}(?:v\d+)?\b", re.IGNORECASE),
        ]
        for m in reversed(messages or []):
            content = str((m or {}).get("content") or "")
            for p in patterns:
                mm = p.search(content)
                if mm:
                    return mm.group(0)
        return None

    @staticmethod
    def _extract_last_namespace_from_history(messages: list[dict]) -> str | None:
        pat = re.compile(r"\bpaper:[\w.\-\/]+:full\b", re.IGNORECASE)
        for m in reversed(messages or []):
            content = str((m or {}).get("content") or "")
            mm = pat.search(content)
            if mm:
                return mm.group(0)
        return None

    @staticmethod
    def _namespace_for_arxiv_id(arxiv_id: str | None) -> str | None:
        if not arxiv_id:
            return None
        normalized = re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE).strip()
        if not normalized:
            return None
        return f"paper:{normalized}:full"

    @staticmethod
    def _normalize_route_value(route_like: Any) -> Route:
        """兼容 Route 枚举 / 'arxiv' / 'Route.ARXIV' 等多种路由表示。"""
        if isinstance(route_like, Route):
            return route_like
        s = str(route_like or "").strip()
        if not s:
            return Route.RAG
        # 兼容 "Route.ARXIV" / "Route.RAG"
        if "." in s:
            s = s.split(".")[-1]
        s_low = s.lower()
        if s_low in ("arxiv", "route_arxiv"):
            return Route.ARXIV
        if s_low in ("weather", "route_weather"):
            return Route.WEATHER
        return Route.RAG

    @staticmethod
    def _arxiv_title_query_from_ingest_question(question: str) -> str:
        """从「下载/入库 + 标题」类问句里抽出用于 arXiv 检索的查询串。"""
        q = (question or "").strip()
        q = q.replace("《", " ").replace("》", " ")
        # 中英文/数字粘连时先切开，否则会出现 need下载 → 去不掉「下载」、残留「并」等
        q = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", q)
        q = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", q)
        phrases = (
            "下载并入库",
            "保存并入库",
            "并入库",
            "帮我把",
            "帮我",
            "请帮我把",
            "请帮我",
            "麻烦把",
            "麻烦帮我",
            "把下面这篇",
            "把这篇",
            "以下论文",
            "下面这篇",
            "这篇论文",
            "这篇",
            "论文",
            "请",
            "全文入库",
            "入库",
            "下载到本地",
            "下载",
            "保存到本地",
            "保存",
            "存本地",
            "加到库",
            "加入库",
            "拉取",
            "pull",
            "fetch",
            "ingest",
            "embed",
            "一下",
            "呢",
            "吧",
            "本地库",
            "本地",
            "arxiv",
            "到库",
            "pdf",
        )
        for ph in sorted(phrases, key=len, reverse=True):
            q = re.sub(re.escape(ph), " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip()
        # 去掉去语料后常挂在英文标题后的「并/且/和」等，避免 ti:"...并" 在 arXiv 上零命中
        q = re.sub(r"(?i)\s*(并|且|还有|以及)\s*$", "", q)
        q = q.strip().strip("，。,.;；、 ")
        return q

    @staticmethod
    def _build_ingest_title_relaxed_queries(raw_title_query: str) -> list[str]:
        """标题检索失败时的放宽查询候选（用于相似项召回）。"""
        q = (raw_title_query or "").strip()
        if not q:
            return []
        out: list[str] = [q]
        # 去标点、压缩空白
        q2 = re.sub(r"[\(\)\[\]\{\}:;,，。！？!?\-_/]+", " ", q)
        q2 = re.sub(r"\s+", " ", q2).strip()
        if q2 and q2.lower() != q.lower():
            out.append(q2)
        # 标题前缀（很多论文标题较长，前半部分通常足够召回）
        toks = [t for t in re.split(r"\s+", q2 or q) if t]
        if len(toks) >= 5:
            out.append(" ".join(toks[:5]))
        if len(toks) >= 8:
            out.append(" ".join(toks[:8]))
        # 专项兜底
        low = q.lower()
        if "3dgs" in low or "gaussian splatting" in low:
            out.extend(["3D Gaussian Splatting", "Gaussian Splatting"])
        # 去重保序
        seen: set[str] = set()
        dedup: list[str] = []
        for s in out:
            x = " ".join(str(s or "").split()).strip()
            if not x:
                continue
            lk = x.lower()
            if lk in seen:
                continue
            seen.add(lk)
            dedup.append(x)
        return dedup[:6]

    @staticmethod
    def _is_paper_read_selection_intent(question: str) -> bool:
        q = (question or "").lower()
        return any(k in q for k in ["读第", "读这篇", "打开这篇", "看这篇", "阅读这篇", "读一下"])

    @staticmethod
    def _parse_selection_index(question: str, size: int) -> int | None:
        if size <= 0:
            return None
        q = (question or "").strip()
        m = re.fullmatch(r"(\d+)", q)
        if m:
            i = int(m.group(1))
            return i if 1 <= i <= size else None
        m = re.search(r"第\s*(\d+)\s*篇", q)
        if m:
            i = int(m.group(1))
            return i if 1 <= i <= size else None
        return None

    @staticmethod
    def _parse_ingest_selection_index(question: str, pending_len: int) -> int | None:
        """解析用户对「候选列表」的序号确认；无法解析则返回 None。"""
        if pending_len <= 0:
            return None
        q = (question or "").strip()
        if not q:
            return None
        if pending_len == 1 and len(q) <= 36:
            if any(
                k in q
                for k in ("就这篇", "就这个", "用这篇", "第一篇", "第一个", "这篇就行")
            ):
                return 1
            low = q.lower()
            if low in ("y", "yes", "ok", "ok."):
                return 1
            if q in ("好", "行"):
                return 1
        m = re.fullmatch(r"(\d+)", q)
        if m:
            i = int(m.group(1))
            return i if 1 <= i <= pending_len else None
        m = re.fullmatch(r"第\s*(\d+)\s*篇\s*", q)
        if m:
            i = int(m.group(1))
            return i if 1 <= i <= pending_len else None
        m = re.fullmatch(r"选\s*(\d+)\s*", q)
        if m:
            i = int(m.group(1))
            return i if 1 <= i <= pending_len else None
        return None

    @staticmethod
    def _parse_ingest_selection_by_title(
        question: str, pending: list[dict[str, Any]]
    ) -> int | None:
        """解析用户通过标题（全称/片段）确认候选，返回 1-based 序号。"""
        if not pending:
            return None
        raw = (question or "").strip()
        if not raw:
            return None
        cleaned = re.sub(
            r"(下载|入库|保存|这篇|论文|帮我|请|选择|选中|就是|我要|确认|一下)",
            " ",
            raw,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 4:
            return None

        def _norm(s: str) -> str:
            x = (s or "").lower()
            x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", x)
            return x

        qn = _norm(cleaned)
        if len(qn) < 4:
            return None

        best_idx = None
        best_score = 0.0
        for i, item in enumerate(pending, start=1):
            tn = _norm(str((item or {}).get("title") or ""))
            if not tn:
                continue
            if qn == tn:
                return i
            if qn in tn or tn in qn:
                score = min(len(qn), len(tn)) / max(len(qn), len(tn))
                if score > best_score:
                    best_score = score
                    best_idx = i
                continue
            qset, tset = set(qn), set(tn)
            inter = len(qset & tset)
            if inter == 0:
                continue
            score = inter / max(1, len(qset | tset))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score >= 0.52:
            return int(best_idx)
        return None

    def binding_and_paper_merge_namespace(
        self,
        question: str,
        session_id: str | None,
        session_rag_namespace: str,
    ) -> tuple[str, str | None, dict[str, Any] | None, dict[str, Any] | None]:
        """绑定「当前论文」并得到 ``paper:<arxiv>:full`` namespace。

        与 ``answer()`` 内逻辑一致，供 LangGraph 等对会话做向量检索时复用，避免只查会话库混入他文。

        返回：
            (question, read_paper_merge_ns, current_paper, early_response)
        early_response 非空时表示应直接返回该 dict（如「读第 N 篇」短确认）。
        """
        ql = (question or "").lower()
        read_paper_merge_ns: str | None = None
        current_paper: dict[str, Any] | None = None

        if session_id:
            try:
                from tools.retrieval.session_paper_state import paper_state_store
                from tools.retrieval.local_paper_service import bind_local_paper_if_mentioned

                if self._is_paper_read_selection_intent(question):
                    cands = paper_state_store.get_candidates(session_id)
                    idx = self._parse_selection_index(question, len(cands))
                    if idx is not None:
                        chosen = dict(cands[idx - 1] or {})
                        paper_state_store.set_current_paper(session_id, chosen)
                        aid = str(chosen.get("arxiv_id") or "").strip()
                        title = str(chosen.get("title") or "该论文").strip()
                        if aid:
                            ns = self._namespace_for_arxiv_id(aid)
                            if ns:
                                read_paper_merge_ns = ns
                                current_paper = chosen
                                if (
                                    any(
                                        k in ql
                                        for k in ["读第", "打开这篇", "看这篇", "阅读这篇"]
                                    )
                                    and len(ql) <= 20
                                ):
                                    early = {
                                        "answer": (
                                            f"已切换到当前论文：{title}（arXiv: {aid}）。"
                                            "你可以继续问：方法、实验、表格或图示细节。"
                                        ),
                                        "citations": [],
                                    }
                                    return question, read_paper_merge_ns, current_paper, early

                if current_paper is None:
                    matched = bind_local_paper_if_mentioned(question)
                    if matched and float(matched.get("match_score") or 0.0) >= 0.56:
                        paper_state_store.set_current_paper(session_id, matched)
                        current_paper = matched
                        aid = str(matched.get("arxiv_id") or "").strip()
                        ns = self._namespace_for_arxiv_id(aid)
                        if ns:
                            read_paper_merge_ns = ns
            except Exception:
                pass

        if (not read_paper_merge_ns) and (current_paper is None):
            try:
                from tools.retrieval.local_paper_service import bind_local_paper_if_mentioned

                matched_any = bind_local_paper_if_mentioned(question)
                if matched_any and float(matched_any.get("match_score") or 0.0) >= 0.56:
                    current_paper = matched_any
                    aid = str(matched_any.get("arxiv_id") or "").strip()
                    ns = self._namespace_for_arxiv_id(aid)
                    if ns:
                        read_paper_merge_ns = ns
            except Exception:
                pass

        if session_id and (not read_paper_merge_ns):
            try:
                from tools.retrieval.session_paper_state import paper_state_store

                cur = paper_state_store.get_current_paper(session_id)
                content_qa = self._looks_like_paper_content_qa_not_ingest(question)
                if cur:
                    aid_cur = str(cur.get("arxiv_id") or "").strip()
                    if aid_cur and (
                        self._is_paper_coref_query(question)
                        or any(
                            k in ql
                            for k in [
                                "方法",
                                "实验",
                                "结论",
                                "table",
                                "figure",
                                "图",
                                "表",
                            ]
                        )
                        or content_qa
                    ):
                        ns_cur = self._namespace_for_arxiv_id(aid_cur)
                        if ns_cur:
                            read_paper_merge_ns = ns_cur
                elif content_qa:
                    # 未写入 current_paper，但上文出现过 arXiv：仍应读到对应 paper: 分区（零额外 LLM）
                    recent = conversation_manager.get_recent_messages(session_id)
                    hist_id = self._extract_last_arxiv_id_from_history(recent)
                    if hist_id:
                        if hist_id not in question:
                            question = f"{question}（arXiv: {hist_id}）"
                        ns_hist = self._namespace_for_arxiv_id(hist_id)
                        if ns_hist:
                            read_paper_merge_ns = ns_hist
            except Exception:
                pass

        if (
            session_id
            and self._is_paper_coref_query(question)
            and not read_paper_merge_ns
        ):
            recent = conversation_manager.get_recent_messages(session_id)
            last_id = self._extract_last_arxiv_id_from_history(recent)
            if (not last_id) and current_paper:
                last_id = str(current_paper.get("arxiv_id") or "").strip() or None
            if last_id and last_id not in question:
                question = f"{question}（arXiv: {last_id}）"
            forced_ns = self._namespace_for_arxiv_id(last_id)
            if forced_ns:
                read_paper_merge_ns = forced_ns

        primary_aid: str | None = None
        if read_paper_merge_ns and str(read_paper_merge_ns).startswith("paper:"):
            try:
                seg = str(read_paper_merge_ns).split(":", 2)[1]
                primary_aid = seg.split(":", 1)[0].strip() or None
            except Exception:
                primary_aid = None
        if not primary_aid and session_id:
            try:
                from tools.retrieval.session_paper_state import paper_state_store

                curp = paper_state_store.get_current_paper(session_id)
                if curp:
                    primary_aid = str(curp.get("arxiv_id") or "").strip() or None
            except Exception:
                pass

        aid_in_question = extract_arxiv_id(question or "")
        if (
            primary_aid
            and aid_in_question
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
        ):
            try:
                p = _normalize_arxiv_id(primary_aid)
                qid = _normalize_arxiv_id(aid_in_question)
                if p and qid and p != qid:
                    trace_event(
                        "paper_binding_lock_skip_secondary_arxiv",
                        {
                            "primary_arxiv_id": p,
                            "ignored_arxiv_in_question": qid,
                            "question": (question or "")[:240],
                        },
                    )
                    aid_in_question = None
            except Exception:
                pass

        if (
            aid_in_question
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
        ):
            ns_paper = self._namespace_for_arxiv_id(aid_in_question)
            if ns_paper:
                read_paper_merge_ns = ns_paper

        return question, read_paper_merge_ns, current_paper, None

    def local_paper_search_list_response(
        self,
        question: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """仅本地论文列表（local_paper_search），供 answer / LangGraph 短路径复用。"""
        from tools.retrieval.local_library_search_service import (
            extract_local_library_topic,
            format_local_library_answer_rows,
            search_local_library_ranked,
        )
        from tools.retrieval.session_paper_state import paper_state_store

        topic = extract_local_library_topic(question)
        rows = search_local_library_ranked(topic, limit=20)
        answer_text = format_local_library_answer_rows(
            rows,
            preamble="本地论文库检索（仅本地索引，未联网）：",
        )
        out: dict[str, Any] = {
            "answer": answer_text,
            "citations": [],
            "local_paper_search": True,
            "intent": "local_paper_search",
        }
        if session_id:
            try:
                paper_state_store.set_candidates(session_id, rows)
            except Exception:
                pass
        trace_event(
            "local_paper_search_done",
            {"topic": (topic or "")[:120], "count": len(rows)},
        )
        return out

    def _maybe_handle_paper_ingest_dialogue(
        self,
        question: str,
        session_id: str | None,
        user_image_paths: list[str] | None = None,
    ) -> dict | None:
        """论文入库对话：显式 ID、候选序号、或仅标题时 arXiv 检索列选项。已处理则返回 out dict。"""
        ql = (question or "").lower()
        sid = (session_id or "").strip()

        def _finish(answer_text: str, citations: list | None = None) -> dict:
            if sid:
                conversation_manager.add_turn(
                    sid, "user", question, image_paths=user_image_paths
                )
                conversation_manager.add_turn(sid, "assistant", answer_text)
            return {"answer": answer_text, "citations": citations or []}

        # 从用户输入抽取 arXiv ID：
        # - extract_arxiv_id 内部使用了 \b 边界；当输入形如 `2603.26665v1下载这篇`（v1 与中文紧贴）时，
        #   \b 可能无法正确切分，导致抽取失败。
        # - 因此这里在未抽到 ID 且存在下载/入库意图时，再做一个“更宽松”的直匹配兜底。
        id_in_question = extract_arxiv_id(question or "")
        loose_id_from_query = False
        if not id_in_question:
            m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", (question or ""), flags=re.IGNORECASE)
            if m:
                id_in_question = m.group(1)
                loose_id_from_query = True
        title_remainder = self._arxiv_title_query_from_ingest_question(question or "")
        arxiv_id_ingest = id_in_question
        # 话里带较长标题时不要用「历史里的 ID」，否则容易把新标题误当成旧论文入库
        if (
            not arxiv_id_ingest
            and sid
            and self._is_download_or_ingest_intent(ql)
            and len((title_remainder or "").strip()) < 12
        ):
            arxiv_id_ingest = self._extract_last_arxiv_id_from_history(
                conversation_manager.get_recent_messages(sid)
            )

        # A：上一轮展示了候选，优先解析序号（避免与历史里的 arXiv ID 冲突）
        if sid:
            pending = conversation_manager.get_pending_ingest_candidates(sid)
            if pending:
                # 用户直接贴 arXiv ID（未再说「下载」）也视为确认入库
                aid_inline = extract_arxiv_id((question or "").strip())
                if aid_inline and not self._is_download_or_ingest_intent(ql):
                    conversation_manager.clear_pending_ingest(sid)
                    try:
                        from tools.agent.paper_ingest import (
                            ingest_arxiv_paper_full_pipeline,
                        )

                        answer_text = ingest_arxiv_paper_full_pipeline(
                            aid_inline, embed_full_text=True
                        )
                        return _finish(
                            f"已按你提供的 arXiv ID `{aid_inline}` 执行入库。\n\n{answer_text}"
                        )
                    except Exception as e:
                        conversation_manager.set_pending_ingest_candidates(sid, pending)
                        return _finish(
                            f"按 ID 入库失败：{e}\n候选列表仍保留，可回复序号/标题或稍后再试。"
                        )

                sel = self._parse_ingest_selection_index(
                    (question or "").strip(), len(pending)
                )
                if sel is None:
                    sel = self._parse_ingest_selection_by_title(
                        (question or "").strip(), pending
                    )
                if sel is not None:
                    aid = (pending[sel - 1] or {}).get("arxiv_id")
                    if aid:
                        conversation_manager.clear_pending_ingest(sid)
                        try:
                            from tools.agent.paper_ingest import (
                                ingest_arxiv_paper_full_pipeline,
                            )

                            answer_text = ingest_arxiv_paper_full_pipeline(
                                str(aid), embed_full_text=True
                            )
                            return _finish(
                                f"已按你的选择（第 {sel} 篇）执行入库。\n\n{answer_text}"
                            )
                        except Exception as e:
                            conversation_manager.set_pending_ingest_candidates(
                                sid, pending
                            )
                            return _finish(
                                f"入库失败：{e}\n候选列表仍保留，可换序号/标题或改用 arXiv ID。"
                            )

        # B：话里或历史里已有 arXiv ID + 下载/入库意图
        if arxiv_id_ingest and self._is_download_or_ingest_intent(ql):
            try:
                trace_event(
                    "ingest_id_from_input",
                    {
                        "arxiv_id": str(arxiv_id_ingest),
                        "loose_regex_used": bool(loose_id_from_query),
                    },
                )
            except Exception:
                pass
            if sid:
                conversation_manager.clear_pending_ingest(sid)
            try:
                from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline

                answer_text = ingest_arxiv_paper_full_pipeline(
                    arxiv_id_ingest, embed_full_text=True
                )
                return _finish(answer_text)
            except Exception as e:
                err = (
                    f"论文入库过程出错：{e}\n"
                    f"可改用工具模式或 CLI：python cli.py embed_paper_full --arxiv-id …"
                )
                return _finish(err)

        # C：有入库意图（或“这篇+标题”）且没有 ID 时 → 按标题检索列选项
        title_like_download = (
            (not self._is_download_or_ingest_intent(ql))
            and (not self._looks_like_paper_content_qa_not_ingest(question or ""))
            and ("这篇" in ql or "this paper" in ql)
            and len((title_remainder or "").strip()) >= 12
        )
        if (self._is_download_or_ingest_intent(ql) or title_like_download) and not arxiv_id_ingest:
            tq = title_remainder or self._arxiv_title_query_from_ingest_question(
                question or ""
            )
            if not tq or len(tq) < 2:
                return _finish(
                    "未在话里识别到 arXiv ID，也未能从句子中抽出有效的论文标题/关键词。\n"
                    "请任选其一：\n"
                    "1）直接给出 ID，例如：`帮我把 1706.03762 下载并入库`\n"
                    "2）用中英文标题或关键词，例如：`下载 Attention Is All You Need 这篇`\n"
                    "3）若上一条已给出候选列表，可回复序号（`1`）或标题（全称/片段）"
                )
            # 优先尝试：从“本 session 最近的 assistant 输出”做标题 -> arXiv ID 映射，
            # 避免 ti:"..." 精确检索在网络/标题细小差异下出现 0 命中的体验不一致。
            mapped_aid: str | None = None
            if sid and tq:
                def _norm_title(s: str) -> str:
                    x = (s or "").lower()
                    x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", x)
                    return x

                tq_norm = _norm_title(tq)
                if tq_norm:
                    best_score = 0.0
                    try:
                        hist = conversation_manager.get_recent_messages(sid)
                    except Exception:
                        hist = []

                    for msg in hist:
                        if (msg or {}).get("role") != "assistant":
                            continue
                        content = str((msg or {}).get("content") or "")
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            # 候选标题行通常形如：1. <title>
                            m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
                            if not m:
                                continue
                            title_line = (m.group(1) or "").strip()
                            if not title_line:
                                continue
                            title_norm = _norm_title(title_line)
                            if not title_norm:
                                continue

                            # lookahead：在标题行后不远处找 arXiv 或 链接
                            aid_candidate = None
                            for j in range(i, min(len(lines), i + 14)):
                                if "arXiv:" in lines[j]:
                                    mm = re.search(
                                        r"arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
                                        lines[j],
                                        flags=re.IGNORECASE,
                                    )
                                    if mm:
                                        aid_candidate = mm.group(1)
                                        break
                                if "链接:" in lines[j]:
                                    mm = re.search(
                                        r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
                                        lines[j],
                                        flags=re.IGNORECASE,
                                    )
                                    if mm:
                                        aid_candidate = mm.group(1)
                                        break
                            if not aid_candidate:
                                continue

                            # 简单相似度：子串优先，其次用 token 交并比
                            if tq_norm in title_norm or title_norm in tq_norm:
                                score = 1.0
                            else:
                                tq_tokens = set(
                                    re.findall(
                                        r"[a-z0-9]+|[\u4e00-\u9fff]+", tq_norm
                                    )
                                )
                                tl_tokens = set(
                                    re.findall(
                                        r"[a-z0-9]+|[\u4e00-\u9fff]+", title_norm
                                    )
                                )
                                inter = len(tq_tokens & tl_tokens)
                                union = len(tq_tokens | tl_tokens) or 1
                                score = inter / union

                            if score > best_score:
                                best_score = score
                                mapped_aid = re.sub(
                                    r"v\d+$",
                                    "",
                                    aid_candidate,
                                    flags=re.IGNORECASE,
                                )

                    if best_score < 0.45:
                        mapped_aid = None

            if mapped_aid:
                try:
                    trace_event(
                        "ingest_id_from_history_title_map",
                        {"mapped_aid": str(mapped_aid), "tq": str(tq)[:120]},
                    )
                except Exception:
                    pass
                try:
                    from tools.agent.paper_ingest import (
                        ingest_arxiv_paper_full_pipeline,
                    )

                    answer_text = ingest_arxiv_paper_full_pipeline(
                        str(mapped_aid), embed_full_text=True
                    )
                    return _finish(
                        f"已从你刚刚展示的候选中匹配到 arXiv ID `{mapped_aid}` 并执行入库。\n\n{answer_text}"
                    )
                except Exception:
                    # 映射失败则走原有回退逻辑（按标题联网检索候选）
                    mapped_aid = None

            if sid and tq and not mapped_aid:
                try:
                    trace_event(
                        "ingest_fallback_to_title_search",
                        {"tq": str(tq)[:120]},
                    )
                except Exception:
                    pass

            papers = []
            tried_queries: list[str] = []
            try:
                for qq in self._build_ingest_title_relaxed_queries(tq):
                    tried_queries.append(qq)
                    got = search_arxiv_for_ingest_disambiguation(
                        qq,
                        display_max=ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS,
                        fetch_max=ARXIV_INGEST_DISAMBIGUATION_FETCH_MAX,
                    )
                    if got:
                        papers = got
                        break
                    # 再走一层普通 arXiv 检索兜底（更宽松）
                    got2 = search_arxiv(
                        qq,
                        max_results=ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS,
                        sort_by="relevance",
                    )
                    if got2:
                        papers = got2
                        break
            except Exception as e:
                return _finish(f"连接 arXiv 检索失败：{e}\n请稍后重试或直接提供 arXiv ID。")

            if not papers:
                return _finish(
                    f"按标题「{tq}」及其相似写法在 arXiv 上未找到论文（可能网络不通，请检查代理）。\n"
                    "可换**更完整英文标题**、**作者名**或**关键词**；"
                    "英文与中文之间建议加空格，例如：`Attention is all you need 下载`；"
                    "或直接给出 arXiv ID，如：`下载 1706.03762 并入库`。\n"
                    f"已尝试查询：{', '.join(tried_queries[:4])}"
                )

            candidates: list[dict[str, Any]] = []
            lines: list[str] = [
                "话里**没有检测到 arXiv ID**。已检索 arXiv：**优先标题完整短语**（API: `ti:\"…\"`），"
                "不足时再按**全文**补充；内部多取若干条后只展示前 "
                f"{ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS} 条。\n"
                "请回复 **序号**（如 `1`）/**论文标题**（全称或片段）或 **arXiv ID** 确认要入库的一篇：",
                "",
            ]
            if tried_queries:
                lines.insert(1, f"已按相似标题尝试检索：{', '.join(tried_queries[:3])}")
                lines.insert(2, "")
            for i, p in enumerate(papers, start=1):
                aid = get_arxiv_id(p)
                aid_norm = re.sub(r"v\d+$", "", aid, flags=re.IGNORECASE).strip()
                title = (p.title or "").strip().replace("\n", " ")
                summ = (p.summary or "").strip().replace("\n", " ")
                summ_short = (summ[:200] + "…") if len(summ) > 200 else summ
                authors = ", ".join((p.authors or [])[:3])
                if len(p.authors or []) > 3:
                    authors += " 等"
                lines.append(f"{i}. [{aid_norm}] {title}")
                lines.append(f"   作者: {authors}")
                lines.append(f"   摘要: {summ_short}")
                lines.append("")
                candidates.append({"arxiv_id": aid_norm, "title": title})

            lines.append(
                "若都不对，请发**更准的标题**重试；"
                "若已确定 ID，可直接说「下载 1234.56789 并入库」。"
            )
            answer_text = "\n".join(lines).strip()
            if sid:
                conversation_manager.set_pending_ingest_candidates(sid, candidates)
            return _finish(answer_text)

        return None

    def answer(
        self,
        question: str,
        namespace: str = DEFAULT_NAMESPACE,
        strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
        k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        use_multi_source: bool = False,
        session_id: str | None = None,
        user_image_paths: list[str] | None = None,
        preclassified_route: Any | None = None,
        preclassified_paper_intent: bool | None = None,
        skip_special_branches: bool = False,
    ) -> dict:
        inp: dict[str, Any] = {
            "question": question,
            "namespace": namespace,
            "strategy": strategy,
            "k": k,
            "session_id": session_id,
            "user_image_paths": user_image_paths,
        }
        run_before_agent(self.middleware, inp)
        if inp.get("_abort"):
            out = inp["_abort"]
            run_after_agent(self.middleware, out)
            return out

        question = inp.get("question", question)
        user_image_paths = inp.get("user_image_paths", user_image_paths)
        namespace = inp.get("namespace", namespace)
        strategy = inp.get("strategy", strategy)
        k = inp.get("k", k)

        # 统一保证：只要有 session_id，就把 user/assistant 写入会话历史。
        # 该项目里 answer() 存在多个早退 return 分支；若不写入会话文件，
        # 前端切换会话后会“看起来丢失最新轮次”。
        def _persist_turn_if_possible(out: dict) -> None:
            if not session_id:
                return
            try:
                ans = out.get("answer", "") if isinstance(out, dict) else ""
                conversation_manager.add_turn(
                    session_id,
                    "user",
                    question,
                    image_paths=user_image_paths,
                )
                conversation_manager.add_turn(
                    session_id,
                    "assistant",
                    str(ans),
                )
            except Exception:
                # 保守：避免写入失败影响主回答链路
                pass

        # 论文入库：ID / 标题检索候选 / 序号确认（任意会话 namespace）
        early_ingest = self._maybe_handle_paper_ingest_dialogue(
            question, session_id, user_image_paths=user_image_paths
        )
        if early_ingest is not None:
            run_after_agent(self.middleware, early_ingest)
            _persist_turn_if_possible(early_ingest)
            return early_ingest

        if self._is_assistant_meta_question(question):
            out = self._answer_assistant_meta_question(
                question, session_id, user_image_paths=user_image_paths
            )
            run_after_agent(self.middleware, out)
            _persist_turn_if_possible(out)
            return out

        session_rag_namespace = namespace
        session_ingest_ids: list[str] = []
        if session_id:
            session_ingest_ids = collect_session_embed_ids_for_namespace(
                conversation_manager.get_recent_messages(session_id),
                session_rag_namespace,
            )

        question, read_paper_merge_ns, current_paper, early_bind = (
            self.binding_and_paper_merge_namespace(
                question, session_id, session_rag_namespace
            )
        )
        if early_bind is not None:
            run_after_agent(self.middleware, early_bind)
            _persist_turn_if_possible(early_bind)
            return early_bind

        # 会话分区本身就是 paper:…:full 时，绑定逻辑未必回填 read_paper_merge_ns；
        # 若不补齐，后续「method 优先 / 论文合并」分支会失效，仍按 k=4 在错误分区弱检索。
        if (not read_paper_merge_ns) and str(session_rag_namespace).startswith(
            "paper:"
        ):
            read_paper_merge_ns = session_rag_namespace

        ql = (question or "").lower()
        namespace = session_rag_namespace

        # 「打开/载入论文」+ ID：只读复制到当前会话 namespace（不修改 paper:... 与 SQLite）
        aid_mirror = extract_arxiv_id(question or "")
        if (
            aid_mirror
            and self._is_session_paper_mirror_intent(ql)
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
        ):
            from tools.agent.paper_session_mirror import mirror_local_paper_to_namespace

            mirror_local_paper_to_namespace(
                self.store,
                session_rag_namespace,
                aid_mirror,
                replace=False,
            )

        route_decision = build_query_route(question)

        # 仅本地论文列表：短路径返回结构化列表，不联网、不走 RAG 摘要生成
        if (
            getattr(route_decision, "intent", "") == "local_paper_search"
            and not skip_special_branches
        ):
            out_ls = self.local_paper_search_list_response(question, session_id)
            run_after_agent(self.middleware, out_ls)
            _persist_turn_if_possible(out_ls)
            return out_ls

        # 已绑定到具体论文且明显是「问内容」：降级 paper_search → paper_qa，并禁止联网
        if read_paper_merge_ns and self._looks_like_paper_content_qa_not_ingest(question):
            route_decision.needs_web = False
            if getattr(route_decision, "intent", "") == "paper_search":
                route_decision.intent = "paper_qa"
                route_decision.route = Route.RAG
                trace_event(
                    "paper_search_demoted_to_paper_qa",
                    {
                        "read_paper_merge_ns": str(read_paper_merge_ns),
                        "question": (question or "")[:200],
                    },
                )

        # 只要句子是“论文内容问答”且携带 arXiv id，也强制走本地 paper_qa（禁止走 paper_search 联网分支）。
        if (
            self._looks_like_paper_content_qa_not_ingest(question)
            and extract_arxiv_id(question or "")
        ):
            if getattr(route_decision, "intent", "") == "paper_search":
                trace_event(
                    "paper_search_demoted_to_paper_qa_with_arxiv_id",
                    {"question": (question or "")[:220]},
                )
            route_decision.intent = "paper_qa"
            route_decision.route = Route.RAG
            route_decision.needs_web = False

        _bound_content_qa = bool(
            read_paper_merge_ns
            and self._looks_like_paper_content_qa_not_ingest(question)
        )
        if (
            getattr(route_decision, "intent", "") != "paper_search"
            and self._looks_like_paper_search_request(question)
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
            and not _bound_content_qa
        ):
            route_decision.intent = "paper_search"
            route_decision.route = Route.ARXIV
            route_decision.source_preference = "local_first"
            route_decision.needs_web = True
            trace_event(
                "paper_search_force_upgrade",
                {"question": question[:200], "reason": "pattern_force_paper_search_tools_mode"},
            )
        # 强制规则兜底：只要问句形态像“找/推荐/最新论文”，一律升级为 paper_search。
        if (
            getattr(route_decision, "intent", "") != "paper_search"
            and self._looks_like_paper_search_request(question)
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
            and not _bound_content_qa
        ):
            route_decision.intent = "paper_search"
            route_decision.route = Route.ARXIV
            route_decision.source_preference = "local_first"
            route_decision.needs_web = True
            trace_event(
                "paper_search_force_upgrade",
                {"question": question[:200], "reason": "pattern_force_paper_search"},
            )
        route = (
            preclassified_route
            if preclassified_route is not None
            else self._normalize_route_value(route_decision.route)
        )
        paper_intent = (
            bool(preclassified_paper_intent)
            if preclassified_paper_intent is not None
            else (
                route_decision.intent
                in ("paper_search", "paper_read", "paper_qa", "local_paper_search")
                or bool(read_paper_merge_ns)
                or bool(current_paper)
            )
        )
        trace_event(
            "route_decision",
            {
                "route": str(route),
                "paper_intent": bool(paper_intent),
                "intent": route_decision.intent,
                "sub_intent": route_decision.sub_intent,
                "paper_match_mode": route_decision.paper_match_mode,
                "session_namespace": session_rag_namespace,
                "read_paper_merge_ns": read_paper_merge_ns,
                "has_user_images": bool(user_image_paths),
            },
        )
        bound_arxiv_id = ""
        if current_paper:
            try:
                bound_arxiv_id = str(current_paper.get("arxiv_id") or "").strip()
            except Exception:
                bound_arxiv_id = ""
        final_retrieval_namespaces: list[str] = [str(session_rag_namespace)]
        if read_paper_merge_ns and read_paper_merge_ns not in final_retrieval_namespaces:
            final_retrieval_namespaces.append(str(read_paper_merge_ns))
        trace_event(
            "paper_local_binding_status",
            {
                "local_paper_bound": bool(read_paper_merge_ns or current_paper),
                "bound_arxiv_id": bound_arxiv_id,
                "final_retrieval_namespace": str(session_rag_namespace),
                "final_retrieval_namespaces": final_retrieval_namespaces,
                "merge_paper_namespace": str(read_paper_merge_ns or ""),
            },
        )

        # 只要本轮带了用户图片，就优先走本地 RAG（避免被“论文/arxiv”意图误路由到在线 arXiv 工具），
        # 这样才能进入“图片-数据库相关性裁决 -> 决定是否注入用户图片”的策略。
        if user_image_paths and route is Route.ARXIV:
            route = Route.RAG

        # 论文意图：不要直接走在线 arXiv 工具；优先本地检索，若无命中再在线补充。
        if route is Route.ARXIV:
            route = Route.RAG

        # 合并论文检索或显式使用 paper: 分区时，强制本地 RAG，避免走在线 arXiv 工具。
        if (
            read_paper_merge_ns or str(session_rag_namespace).startswith("paper:")
        ) and not self._is_download_or_ingest_intent(ql) and not self._is_delete_intent(ql):
            route = Route.RAG

        # 论文“找文献”类意图：本地优先，结果不足自动联网补充（除非用户明确仅本地）。
        if (
            (not skip_special_branches)
            and (getattr(route_decision, "intent", "") == "paper_search")
            and (not user_image_paths)
            and (not self._is_download_or_ingest_intent(ql))
            and (not self._is_delete_intent(ql))
            and (not self._looks_like_paper_content_qa_not_ingest(question))
            and not (
                read_paper_merge_ns
                and self._looks_like_paper_content_qa_not_ingest(question)
            )
        ):
            trace_event(
                "paper_search_entry",
                {
                    "route_result": str(route),
                    "enter_paper_search": True,
                    "question": question[:240],
                },
            )
            try:
                count_hint = self._paper_search_count_hint(question)
                local_only = self._is_paper_search_local_only(question)
                force_web = self._is_paper_search_force_web(question)
                latest_req = self._is_latest_or_recent_paper_search(question)
                topic_query = self._paper_search_topic_query(question)
                keyword_candidates = self._paper_search_expand_bilingual_keywords(topic_query)

                # Step1: 本地先查
                from tools.retrieval.local_paper_service import search_local_papers

                trace_event("paper_search_local_query", {"used_local_db": True, "topic_query": topic_query[:200]})
                local_rows = search_local_papers(
                    keyword_candidates or [topic_query],
                    limit=max(count_hint, 8),
                )
                local_ok = self._is_local_paper_search_result_sufficient(
                    local_rows,
                    question=question,
                    count_hint=count_hint,
                )

                # Step2: 判定是否联网补充
                need_web = False
                reason = "local_sufficient"
                if local_only:
                    need_web = False
                    reason = "user_forbid_web"
                elif force_web:
                    need_web = True
                    reason = "user_force_web"
                elif (not local_rows):
                    need_web = True
                    reason = "local_empty"
                elif (not local_ok):
                    need_web = True
                    reason = "local_insufficient"
                elif latest_req:
                    # “最新/最近”是强联网信号：本地即使有，也默认补充联网
                    need_web = True
                    reason = "latest_or_recent_prefers_web_supplement"

                web_text = ""
                if need_web:
                    web_text = str(
                        tool_search_arxiv.invoke(
                            {
                                # 必须保留用户原句，避免把“最新/最近”等时间偏好在预处理时丢掉
                                "query": question,
                                "max_results": max(2, int(count_hint or 4)),
                            }
                        )
                    )
                web_count = self._count_numbered_items(web_text)

                answer_text = self._format_paper_search_result_text(
                    question=question,
                    local_rows=local_rows[: max(1, count_hint)],
                    web_text=web_text,
                    used_web=bool(need_web),
                )
                if session_id:
                    try:
                        from tools.retrieval.session_paper_state import paper_state_store

                        paper_state_store.set_candidates(session_id, local_rows[:20])
                    except Exception:
                        pass
                trace_event(
                    "paper_search_fallback",
                    {
                        "topic_query": topic_query[:200],
                        "keyword_candidates": keyword_candidates[:6],
                        "count_hint": int(count_hint),
                        "local_only": bool(local_only),
                        "force_web": bool(force_web),
                        "latest_req": bool(latest_req),
                        "local_rows": len(local_rows),
                        "local_sufficient": bool(local_ok),
                        "need_web": bool(need_web),
                        "web_results_count": int(web_count),
                        "final_results_count": int(len(local_rows[: max(1, count_hint)]) + web_count),
                        "reason": reason,
                    },
                )
                out = {
                    "answer": answer_text,
                    "citations": [],
                    "web_fallback": bool(need_web),
                    "web_supplement": bool(need_web and bool(local_rows)),
                    "web_forced_by_user": bool(force_web),
                }
                run_after_agent(self.middleware, out)
                _persist_turn_if_possible(out)
                return out
            except Exception as e:
                # 禁止在 paper_search 场景落入“上下文不足拒答”模板；失败也返回任务型错误说明。
                out = {
                    "answer": (
                        "已识别为论文搜索请求，并已尝试本地优先检索与联网补充。"
                        f"本轮执行异常：{e}。请重试，或改为“只查本地/请联网搜索”来指定策略。"
                    ),
                    "citations": [],
                    "web_fallback": False,
                    "web_supplement": False,
                    "web_forced_by_user": False,
                }
                trace_event(
                    "paper_search_error",
                    {
                        "route_result": str(route),
                        "enter_paper_search": True,
                        "used_local_db": True,
                        "trigger_web": False,
                        "final_results_count": 0,
                        "error": str(e)[:400],
                    },
                )
                run_after_agent(self.middleware, out)
                _persist_turn_if_possible(out)
                return out

        # 天气：通过 LangChain Tool 调用
        if (not skip_special_branches) and route is Route.WEATHER:
            weather_text = tool_weather.invoke({"query": question})
            # 让模型把天气数据改写成“用户舒服的自然语言 + 建议”
            advise_system = (
                "你是天气建议助手。"
                "用户询问某城市当前天气，并希望得到穿衣/出行等生活建议。"
                "请根据下方工具给出的实况数据，用通顺的中文回答；**回答须充实，禁止只用一两句话带过**。"
                "必须自然语句点明：气温（摄氏度）、相对湿度（若数据中有）、体感温度（若有）、风速（若有）、天气现象（晴/多云等，若有）；"
                "缺项就不要编造。"
                "不要使用 `temp=`、`wind_speed=` 这类键值对或字段名式罗列。"
                "结构：先 **1～2 段**展开描述实况（含上述指标，可顺带一句对体感或舒适度的概括），"
                "再分点给出 **4～6 条**建议（穿衣层次与增减、是否带伞/防晒、通勤与户外活动、老人幼儿、过敏或呼吸道提示等），"
                "每条建议用 **1～2 句**并尽量点明与实况数据的对应关系。"
            )
            messages = [
                {"role": "system", "content": advise_system},
                {
                    "role": "user",
                    "content": f"气象工具输出：\n{weather_text}\n\n用户问题：{question}",
                },
            ]
            run_before_model(self.middleware, messages)
            resp = self.llm.invoke(dict_messages_to_lc(messages))
            run_after_model(self.middleware, resp)
            answer_text = resp.content if hasattr(resp, "content") else str(resp)
            answer_text = self._with_truncation_reason(resp, str(answer_text))
            out = {"answer": answer_text, "citations": []}
            run_after_agent(self.middleware, out)
            _persist_turn_if_possible(out)
            return out

        # 历史查询：优先从“历史对话向量库”检索，而不是论文库。
        if (not skip_special_branches) and self._is_history_query(question):
            from tools.storage.long_memory import retrieve_conversation_memories
            history_items = retrieve_conversation_memories(
                query=question,
                session_id=session_id,
                k=k,
                score_threshold=0.8,
                strategy=DEFAULT_RETRIEVAL_STRATEGY,
            )
            if history_items:
                history_ctx = "\n\n".join(
                    [f"- [session={h.get('session_id')} turn={h.get('turn_index')} score={h.get('score'):.3f}] {h.get('text')}" for h in history_items]
                )
                history_messages = [
                    {
                        "role": "system",
                        "content": "你是对话历史助手。请仅基于给定历史对话片段回答，不确定就明确说不知道。",
                    },
                    {
                        "role": "user",
                        "content": f"用户问题：{question}\n\n历史对话检索结果：\n{history_ctx}",
                    },
                ]
                run_before_model(self.middleware, history_messages)
                resp = self.llm.invoke(history_messages)
                run_after_model(self.middleware, resp)
                answer_text = resp.content if hasattr(resp, "content") else str(resp)
                answer_text = self._with_truncation_reason(resp, str(answer_text))
                citations = [
                    {
                        "source": h.get("source", "conversation_memory"),
                        "score": float(h.get("score") or 0.0),
                        "preview": str(h.get("text") or "")[:200],
                    }
                    for h in history_items
                ]
                out = {"answer": answer_text, "citations": citations}
                run_after_agent(self.middleware, out)
                _persist_turn_if_possible(out)
                return out

        # 非论文问题：优先使用“对话历史”（向量记忆 + 最近对话重放），不足再联网搜索并重答。
        if (not skip_special_branches) and (not paper_intent) and (not self._is_history_query(question)):
            chat_history = (
                conversation_manager.get_recent_messages(session_id)
                if session_id
                else None
            )

            # StepA：对话记忆向量检索
            history_pairs: list[tuple[object, float]] = []
            if session_id:
                try:
                    from tools.storage.long_memory import retrieve_conversation_memories

                    hist_items = retrieve_conversation_memories(
                        query=question,
                        session_id=session_id,
                        k=max(2, min(12, int(k or 6))),
                        score_threshold=0.8,
                        strategy=DEFAULT_RETRIEVAL_STRATEGY,
                    )
                    for it in hist_items or []:
                        txt = str(it.get("text") or "").strip()
                        if not txt:
                            continue
                        src = str(it.get("source") or "conversation_memory")
                        history_pairs.append(
                            (
                                Document(
                                    page_content=txt[:8000],
                                    metadata={
                                        "source": src,
                                        "type": "conversation_memory",
                                        "session_id": it.get("session_id"),
                                        "turn_index": it.get("turn_index"),
                                    },
                                ),
                                float(it.get("score") or 0.0),
                            )
                        )
                except Exception:
                    history_pairs = []

            history_grouped = self._group_by_parent(history_pairs) if history_pairs else []

            # StepC：用现有 judge_retrieval_context 评估“历史上下文是否足够”
            need_web = True
            try:
                from config import RAG_CONTEXT_SCORE_MIN, RAG_WEB_FALLBACK_ENABLED, RAG_WEB_MERGED_MAX_RESULTS
                from tools.rag.retrieval_judge import judge_retrieval_context

                if history_grouped:
                    verdict = judge_retrieval_context(
                        self.llm,
                        question,
                        history_grouped,
                        score_min=RAG_CONTEXT_SCORE_MIN,
                    )
                    score = float(verdict.get("score", 0.0))
                    should_sup = bool(verdict.get("should_supplement_web"))
                    need_web = (score < float(RAG_CONTEXT_SCORE_MIN)) or should_sup
                    trace_event(
                        "history_context_judge",
                        {
                            "history_grouped_count": len(history_grouped),
                            "score": score,
                            "should_supplement_web": should_sup,
                            "need_web": need_web,
                        },
                    )
                else:
                    need_web = True
                    trace_event(
                        "history_context_judge",
                        {"history_grouped_count": 0, "need_web": True, "reason": "no_history_context"},
                    )

                if (not need_web) or (not RAG_WEB_FALLBACK_ENABLED):
                    # 直接基于历史回答（或 web 被关闭只能用历史/无材料）
                    messages = build_rag_prompt(
                        question,
                        history_grouped,
                        chat_history=chat_history,
                        user_image_paths=user_image_paths,
                        include_user_images_in_history=bool(user_image_paths),
                    )
                    run_before_model(self.middleware, messages)
                    resp = self.llm.invoke(dict_messages_to_lc(messages))
                    run_after_model(self.middleware, resp)
                    answer_text = resp.content if hasattr(resp, "content") else str(resp)
                    answer_text = self._with_truncation_reason(resp, str(answer_text))

                    citations: list[dict] = []
                    for doc, sc in history_grouped:
                        meta = getattr(doc, "metadata", {}) or {}
                        citations.append(
                            {
                                "source": meta.get("source", "conversation_memory"),
                                "score": float(sc),
                                "preview": getattr(doc, "page_content", "")[:200],
                            }
                        )
                    out = {
                        "answer": answer_text,
                        "citations": citations,
                        "web_fallback": False,
                        "web_supplement": False,
                        "web_forced_by_user": False,
                        "retrieval_context_score": float(verdict.get("score", 0.0)) if history_grouped else 0.0,
                        "retrieval_judge_reason": str(verdict.get("reason") or "") if history_grouped else "no_history_context",
                    }
                    run_after_agent(self.middleware, out)
                    _persist_turn_if_possible(out)
                    return out

                # StepD：联网搜索并重答（非论文路径）
                from tools.agent.web_search import search_web_with_subquestions, web_items_to_document_pairs

                web_items, _web_note = search_web_with_subquestions(
                    question,
                    self.llm,
                    max_merged_results=RAG_WEB_MERGED_MAX_RESULTS,
                )
                if web_items:
                    web_pairs = web_items_to_document_pairs(
                        web_items,
                        score_base=0.1,
                        meta_type="web_search_nonpaper",
                    )
                    web_grouped = self._group_by_parent(web_pairs)
                    messages = build_rag_prompt(
                        question,
                        web_grouped,
                        chat_history=chat_history,
                        user_image_paths=user_image_paths,
                        include_user_images_in_history=bool(user_image_paths),
                    )
                    run_before_model(self.middleware, messages)
                    resp = self.llm.invoke(dict_messages_to_lc(messages))
                    run_after_model(self.middleware, resp)
                    answer_text = resp.content if hasattr(resp, "content") else str(resp)
                    answer_text = self._with_truncation_reason(resp, str(answer_text))

                    citations: list[dict] = []
                    for doc, sc in web_grouped:
                        meta = getattr(doc, "metadata", {}) or {}
                        citations.append(
                            {
                                "source": meta.get("source", "web"),
                                "score": float(sc),
                                "preview": getattr(doc, "page_content", "")[:200],
                            }
                        )
                    out = {
                        "answer": answer_text,
                        "citations": citations,
                        "web_fallback": True,
                        "web_supplement": False,
                        "web_forced_by_user": False,
                        "retrieval_context_score": float(verdict.get("score", 0.0)) if history_grouped else 0.0,
                        "retrieval_judge_reason": str(verdict.get("reason") or "") if history_grouped else "no_history_context",
                    }
                    run_after_agent(self.middleware, out)
                    _persist_turn_if_possible(out)
                    return out
            except Exception:
                pass

        method_focus_query = self._question_wants_method(question)
        # 默认走本地 RAG 或多源 RAG（主分区 = 会话 namespace）
        # 方法类问题 + 已锁定 paper:… 分区：无论会话分区是否与其相同，都只做「论文正文」强检索，
        # 避免 session=conv_* 时他文噪声；也避免 session=paper:… 时仍走 k=4 弱召回只命中摘要。
        if (
            read_paper_merge_ns
            and str(read_paper_merge_ns).startswith("paper:")
            and paper_intent
            and method_focus_query
        ):
            paper_hits = self._retrieve_local(
                question=question,
                namespace=read_paper_merge_ns,
                strategy=strategy,
                k=max(int(k or 4), 20),
                score_threshold=min(float(score_threshold or 0.5), 0.15),
                session_ingest_ids=None,
                paper_intent_hint=True,
            )
            paper_hits = self.store.expand_neighbor_chunks(
                retrieved=paper_hits,
                namespace=read_paper_merge_ns,
                window=2,
            )
            retrieved = list(paper_hits)[: max(k * 3, 24)]
            trace_event(
                "paper_method_primary_retrieve",
                {
                    "namespace": read_paper_merge_ns,
                    "k": max(int(k or 4), 20),
                    "score_threshold": min(float(score_threshold or 0.5), 0.15),
                    "hits": len(retrieved),
                    "session_merged": False,
                },
            )
        elif read_paper_merge_ns and read_paper_merge_ns != session_rag_namespace:

            def _session_retrieval_bundle() -> List[Tuple[object, float]]:
                if use_multi_source:
                    return self._retrieve_multi_source(
                        question=question,
                        namespace=session_rag_namespace,
                        k=k,
                        score_threshold=score_threshold,
                        session_ingest_ids=session_ingest_ids or None,
                    )
                loc = self._retrieve_local(
                    question=question,
                    namespace=session_rag_namespace,
                    strategy=strategy,
                    k=k,
                    score_threshold=score_threshold,
                    session_ingest_ids=session_ingest_ids or None,
                    paper_intent_hint=bool(paper_intent),
                )
                return self.store.expand_neighbor_chunks(
                    retrieved=loc,
                    namespace=session_rag_namespace,
                    window=1,
                )

            def _paper_retrieval_bundle() -> List[Tuple[object, float]]:
                ph = self._retrieve_local(
                    question=question,
                    namespace=read_paper_merge_ns,
                    strategy=strategy,
                    k=max(k, 8),
                    score_threshold=score_threshold,
                    session_ingest_ids=None,
                    paper_intent_hint=True,
                )
                return self.store.expand_neighbor_chunks(
                    retrieved=ph,
                    namespace=read_paper_merge_ns,
                    window=1,
                )

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_s = ex.submit(_session_retrieval_bundle)
                fut_p = ex.submit(_paper_retrieval_bundle)
                retrieved = fut_s.result()
                paper_hits = fut_p.result()
            retrieved = merge_ranked_lists([list(retrieved), list(paper_hits)])
            retrieved = retrieved[: max(k * 2, 12)]
        elif use_multi_source:
            retrieved = self._retrieve_multi_source(
                question=question,
                namespace=session_rag_namespace,
                k=k,
                score_threshold=score_threshold,
                session_ingest_ids=session_ingest_ids or None,
            )
        else:
            retrieved = self._retrieve_local(
                question=question,
                namespace=session_rag_namespace,
                strategy=strategy,
                k=k,
                score_threshold=score_threshold,
                session_ingest_ids=session_ingest_ids or None,
                paper_intent_hint=bool(paper_intent),
            )
            retrieved = self.store.expand_neighbor_chunks(
                retrieved=retrieved,
                namespace=session_rag_namespace,
                window=1,
            )
        trace_event(
            "retrieval_done",
            {
                "retrieved_count": len(retrieved or []),
                "namespace": session_rag_namespace,
                "strategy": strategy,
                "session_ingest_ids_count": len(session_ingest_ids or []),
            },
        )

        # 兜底：代词问题 + 仍无命中 => 用历史里出现的 paper:...:full 再合并一源（不切换会话分区）
        if (
            (not retrieved)
            and session_id
            and self._is_paper_coref_query(question)
            and not self._is_download_or_ingest_intent(question)
            and not self._is_delete_intent(question)
        ):
            recent = conversation_manager.get_recent_messages(session_id)
            fallback_ns = self._extract_last_namespace_from_history(recent)
            if (
                fallback_ns
                and str(fallback_ns).startswith("paper:")
                and fallback_ns != session_rag_namespace
            ):
                read_paper_merge_ns = read_paper_merge_ns or fallback_ns
                paper_hits = self._retrieve_local(
                    question=question,
                    namespace=fallback_ns,
                    strategy=strategy,
                    k=max(k, 8),
                    score_threshold=score_threshold,
                    session_ingest_ids=None,
                    paper_intent_hint=True,
                )
                paper_hits = self.store.expand_neighbor_chunks(
                    retrieved=paper_hits,
                    namespace=fallback_ns,
                    window=1,
                )
                retrieved = merge_ranked_lists([list(retrieved), list(paper_hits)])
                retrieved = retrieved[: max(k * 2, 12)]

        grouped = self._group_by_parent(retrieved)

        # 论文问答增强：若命中上下文偏薄（多为摘要/图注），补召回正文 generic chunk。
        if (
            paper_intent
            and (not image_only_mode if "image_only_mode" in locals() else True)
            and read_paper_merge_ns
            and str(read_paper_merge_ns).startswith("paper:")
            and self._paper_context_is_thin(grouped)
        ):
            try:
                body_queries = expand_retrieval_queries(
                    question,
                    strategy=strategy,
                    llm=self.llm,
                )
                if not body_queries:
                    qq = (question or "").strip()
                    body_queries = [qq] if qq else []
                body_hits = self.store.retrieve(
                    body_queries,
                    namespace=read_paper_merge_ns,
                    k=max(int(k or 4), 16),
                    score_threshold=min(float(score_threshold or 0.5), 0.25),
                    strategy="hybrid_rerank",
                    session_ingest_ids=None,
                    extra_chroma_filter={"chunk_role": "generic"},
                )
                body_hits = self.store.expand_neighbor_chunks(
                    retrieved=body_hits,
                    namespace=read_paper_merge_ns,
                    window=2,
                )
                if body_hits:
                    retrieved = merge_ranked_lists([list(retrieved), list(body_hits)])
                    retrieved = retrieved[: max(int(k or 4) * 3, 24)]
                    grouped = self._group_by_parent(retrieved)
                trace_event(
                    "paper_body_boost",
                    {
                        "namespace": read_paper_merge_ns,
                        "thin_before": True,
                        "body_hits": len(body_hits or []),
                        "grouped_after": len(grouped or []),
                    },
                )
            except Exception as e:
                trace_event("paper_body_boost_error", {"error": str(e)[:300]})

        # 论文问“方法”但未命中 method 证据：触发 method section 定向补召回。
        if (
            paper_intent
            and read_paper_merge_ns
            and str(read_paper_merge_ns).startswith("paper:")
            and self._question_wants_method(question)
            and (not self._has_method_evidence(grouped))
        ):
            try:
                method_queries = [
                    question,
                    "method approach architecture model design",
                    "section 3 method approach",
                    "方法 模型结构 技术路线",
                ]
                method_hits = self.store.retrieve(
                    method_queries,
                    namespace=read_paper_merge_ns,
                    k=max(int(k or 4), 26),
                    score_threshold=min(float(score_threshold or 0.5), 0.05),
                    strategy="hybrid_rerank",
                    session_ingest_ids=None,
                    extra_chroma_filter={"chunk_role": "generic"},
                )
                method_hits = self.store.expand_neighbor_chunks(
                    retrieved=method_hits,
                    namespace=read_paper_merge_ns,
                    window=2,
                )
                if method_hits:
                    retrieved = merge_ranked_lists([list(retrieved), list(method_hits)])
                    retrieved = retrieved[: max(int(k or 4) * 3, 24)]
                    grouped = self._group_by_parent(retrieved)
                trace_event(
                    "paper_method_fallback",
                    {
                        "question": question[:220],
                        "namespace": read_paper_merge_ns,
                        "method_hits": len(method_hits or []),
                        "grouped_after": len(grouped or []),
                    },
                )
            except Exception as e:
                trace_event("paper_method_fallback_error", {"error": str(e)[:300]})

        paper_ns = ""
        if read_paper_merge_ns and str(read_paper_merge_ns).startswith("paper:"):
            paper_ns = str(read_paper_merge_ns)
        elif str(session_rag_namespace).startswith("paper:"):
            paper_ns = str(session_rag_namespace)
        try:
            from tools.retrieval.method_section_context import prepend_method_section_full_context

            grouped = prepend_method_section_full_context(
                list(grouped),
                namespace=paper_ns,
                question=question,
            )
        except Exception:
            pass

        web_fallback_used = False
        web_supplement_used = False
        web_forced_by_user = False

        # 用户图片裁决：判断用户图片与“数据库检索上下文”是否直接相关。
        # - 相关：把图片注入最终回答，并允许重放历史图片
        # - 不相关：只基于图片回答（不使用数据库上下文 / 不走联网与 SQLite 兜底）
        image_relevant_to_db = True
        if user_image_paths:
            try:
                image_relevant_to_db = self._judge_use_user_images(
                    question=question,
                    grouped=list(grouped),
                    user_image_paths=user_image_paths,
                )
            except Exception:
                # 裁决失败时保守：仍使用数据库上下文与图片，避免无谓地丢证据
                image_relevant_to_db = True
        trace_event(
            "image_relevance_decision",
            {
                "has_user_images": bool(user_image_paths),
                "image_relevant_to_db": bool(image_relevant_to_db),
            },
        )

        image_only_mode = bool(user_image_paths) and not image_relevant_to_db
        include_user_images_in_history = bool(user_image_paths) and image_relevant_to_db
        filtered_user_image_paths = user_image_paths

        grouped_for_prompt = grouped if not image_only_mode else []
        grouped_for_citations = grouped if not image_only_mode else []
        retrieval_context_score: float | None = None
        retrieval_judge_reason: str | None = None

        # 论文意图：本地未命中则在线 arXiv 搜索。
        # 这里直接复用 tool_search_arxiv 的实现：它会用 Qwen 做 query rewrite，
        # 显著提升像“3DGS(=3D Gaussian Splatting)”这类缩写的召回率。
        if paper_intent and not grouped and not image_only_mode:
            trace_event(
                "paper_fallback_arxiv",
                {"question": question[:200], "k": max(2, int(k or 2))},
            )
            try:
                answer_text = str(
                    tool_search_arxiv.invoke(
                        {
                            "query": question,
                            "max_results": max(2, int(k or 2)),
                        }
                    )
                )
                out = {"answer": answer_text, "citations": []}
                run_after_agent(self.middleware, out)
                _persist_turn_if_possible(out)
                return out
            except Exception:
                pass

        # 本地已命中：可选由 LLM 评判是否充分；不足则按模型给出的查询联网补充，并与本地片段合并
        if (
            grouped
            and allow_web_search_when_local_misses(route)
            and not self._is_history_query(question)
            and not read_paper_merge_ns
            and not str(session_rag_namespace).startswith("paper:")
            and not image_only_mode
        ):
            from config import (
                RAG_AUTO_JUDGE_USE_LLM,
                RAG_CONTEXT_SCORE_MIN,
                RAG_LLM_CONTEXT_SCORE_MODE,
                RAG_WEB_FALLBACK_ENABLED,
                RAG_WEB_MERGED_MAX_RESULTS,
            )
            from tools.agent.rag_judge_route import retrieval_judge_enabled_for_question

            if user_requests_forced_web_search(question):
                run_retrieval_judge = False
            elif RAG_LLM_CONTEXT_SCORE_MODE == "off":
                run_retrieval_judge = False
            elif RAG_LLM_CONTEXT_SCORE_MODE == "on":
                run_retrieval_judge = True
            else:
                run_retrieval_judge = retrieval_judge_enabled_for_question(
                    question,
                    llm=self.llm,
                    use_llm_router=RAG_AUTO_JUDGE_USE_LLM,
                    user_forces_web=False,
                )

            if run_retrieval_judge and RAG_WEB_FALLBACK_ENABLED:
                from tools.agent.web_search import (
                    search_web_from_query_list,
                    web_items_to_document_pairs,
                )
                from tools.rag.retrieval_judge import judge_retrieval_context

                verdict = judge_retrieval_context(
                    self.llm,
                    question,
                    grouped,
                    score_min=RAG_CONTEXT_SCORE_MIN,
                )
                retrieval_context_score = float(verdict.get("score", 10))
                retrieval_judge_reason = str(verdict.get("reason") or "")
                if verdict.get("should_supplement_web"):
                    wq_list = verdict.get("web_queries") or [question]
                    web_items_sup, _w_sup_note = search_web_from_query_list(
                        wq_list,
                        max_merged_results=min(RAG_WEB_MERGED_MAX_RESULTS, 10),
                        max_per_query=4,
                    )
                    if web_items_sup:
                        local_max = max((s for _, s in grouped), default=0.0)
                        score_base = float(local_max) + 0.08
                        extra_pairs = web_items_to_document_pairs(
                            web_items_sup,
                            score_base=score_base,
                            meta_type="web_search_supplement",
                        )
                        grouped = self._group_by_parent(
                            list(grouped) + extra_pairs
                        )
                        web_supplement_used = True

        # 用户明确要求「联网搜索」等：在本地已有命中时也强制联网，合并摘要（需开启联网兜底）
        if (
            grouped
            and user_requests_forced_web_search(question)
            and allow_web_search_when_local_misses(route)
            and not self._is_history_query(question)
            and not image_only_mode
        ):
            from config import RAG_WEB_FALLBACK_ENABLED, RAG_WEB_MERGED_MAX_RESULTS

            if RAG_WEB_FALLBACK_ENABLED:
                from tools.agent.web_search import (
                    search_web_with_subquestions,
                    web_items_to_document_pairs,
                )

                q_web = strip_forced_web_search_phrases(question) or question
                web_items_u, _wu_note = search_web_with_subquestions(
                    q_web,
                    self.llm,
                    max_merged_results=RAG_WEB_MERGED_MAX_RESULTS,
                )
                if web_items_u:
                    local_max = max((s for _, s in grouped), default=0.0)
                    score_base = float(local_max) + 0.15
                    extra_u = web_items_to_document_pairs(
                        web_items_u,
                        score_base=score_base,
                        meta_type="web_search_user_requested",
                    )
                    grouped = self._group_by_parent(list(grouped) + extra_u)
                    web_forced_by_user = True

        # 论文分区（合并目标或显式 paper: 会话）且无向量命中时，回退到 SQLite 元数据（摘要）回答，
        # 避免模型在无上下文情况下“自由发挥”或偏到外部检索结果。
        paper_tgt_ns = read_paper_merge_ns or (
            session_rag_namespace if str(session_rag_namespace).startswith("paper:") else None
        )
        if paper_tgt_ns and not grouped and not image_only_mode:
            try:
                from pathlib import Path

                from tools.agent.arxiv_search import download_pdf, get_paper_by_id
                from tools.rag.document import load_pdf
                from tools.storage.paper_library import (
                    LocalPaper,
                    get_paper as get_local_paper,
                    upsert_paper,
                )

                # namespace 格式：paper:<arxiv_id>:full
                parts = str(paper_tgt_ns).split(":")
                arxiv_id = parts[1] if len(parts) >= 2 else None
                rec = get_local_paper(arxiv_id or "") if arxiv_id else None
                if arxiv_id:
                    existing_namespaces = set((rec or {}).get("namespaces") or [])
                    title = (rec or {}).get("title") or ""
                    summary = ((rec or {}).get("summary") or "").strip()
                    url = (rec or {}).get("url") or f"https://arxiv.org/abs/{arxiv_id}"
                    pdf_path = (rec or {}).get("pdf_path") or f"data/papers/{arxiv_id}.pdf"
                    pdf_exists = Path(pdf_path).exists()
                    downloaded = False
                    metadata_refreshed = False
                    embedded_full_text = False
                    embed_error = ""

                    # 本地信息缺失时的兜底：如果本地元数据不完整，就去网上补齐元数据；
                    # 如果本地 PDF 缺失，则会自动下载。
                    if (not rec) or (not summary) or (not pdf_exists):
                        paper_meta = None
                        # 当 arXiv 元数据 API 很慢/卡住时，避免阻塞整个对话。
                        try:
                            with ThreadPoolExecutor(max_workers=1) as ex:
                                fut = ex.submit(get_paper_by_id, arxiv_id)
                                paper_meta = fut.result(timeout=8)
                        except FuturesTimeoutError:
                            paper_meta = None
                        except Exception:
                            paper_meta = None
                        if paper_meta:
                            title = paper_meta.title or title
                            summary = (paper_meta.summary or summary or "").strip()
                            url = paper_meta.url or url
                            metadata_refreshed = True
                        if not pdf_exists:
                            try:
                                new_pdf = download_pdf(arxiv_id, dest_dir="data/papers")
                                pdf_path = str(new_pdf.as_posix())
                                pdf_exists = True
                                downloaded = True
                            except Exception:
                                pass
                        upsert_paper(
                            LocalPaper(
                                arxiv_id=arxiv_id,
                                pdf_path=pdf_path,
                                title=title or None,
                                authors=((paper_meta.authors if "paper_meta" in locals() and paper_meta else (rec or {}).get("authors")) or []),
                                summary=summary or None,
                                published=((paper_meta.published.isoformat() if "paper_meta" in locals() and paper_meta and paper_meta.published else (rec or {}).get("published"))),
                                url=url,
                                namespaces=list(existing_namespaces),
                            )
                        )

                    # 在决定是否需要入库向量之前，先刷新最新的记录与 namespaces。
                    rec_after_sync = get_local_paper(arxiv_id) or {}
                    existing_namespaces = set(rec_after_sync.get("namespaces") or [])

                    # 当本地 PDF 就绪且该 namespace 尚未入库时，自动把全文嵌入向量库。
                    if pdf_exists and paper_tgt_ns not in existing_namespaces:
                        try:
                            from tools.agent.paper_ingest import embed_arxiv_abstract_documents

                            public_ns = (RAG_PUBLIC_NAMESPACE or "").strip() or DEFAULT_NAMESPACE
                            rec_meta = rec_after_sync
                            embed_arxiv_abstract_documents(
                                vector_store,
                                arxiv_id=arxiv_id,
                                title=(rec_meta.get("title") or title or None),
                                authors=list(rec_meta.get("authors") or []),
                                summary=(
                                    (rec_meta.get("summary") or summary or "").strip() or None
                                ),
                                paper_namespace=paper_tgt_ns,
                                public_namespace=public_ns,
                                also_embed_public=bool(
                                    RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC
                                    and public_ns
                                    and public_ns != paper_tgt_ns
                                ),
                            )
                            text, meta = load_pdf(str(pdf_path), parent_id=arxiv_id)
                            vector_store.embed_document(
                                text=text,
                                namespace=paper_tgt_ns,
                                chunk_size=DEFAULT_CHUNK_SIZE,
                                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                                extra_metadata=meta,
                            )
                            from tools.rag.pdf_figures import embed_pdf_figures_to_namespace

                            embed_pdf_figures_to_namespace(
                                vector_store,
                                pdf_path=str(pdf_path),
                                parent_id=arxiv_id,
                                namespace=paper_tgt_ns,
                                arxiv_id=arxiv_id,
                            )
                            upsert_paper(
                                LocalPaper(
                                    arxiv_id=arxiv_id,
                                    pdf_path=pdf_path,
                                    title=title or None,
                                    authors=((paper_meta.authors if "paper_meta" in locals() and paper_meta else (rec or {}).get("authors")) or []),
                                    summary=summary or None,
                                    published=((paper_meta.published.isoformat() if "paper_meta" in locals() and paper_meta and paper_meta.published else (rec or {}).get("published"))),
                                    url=url,
                                    namespaces=sorted(set(existing_namespaces | {paper_tgt_ns})),
                                )
                            )
                            embedded_full_text = True
                        except Exception as e:
                            embed_error = str(e)

                    prefix = "我在本地向量库中暂未命中该论文正文片段。"
                    if downloaded:
                        prefix += " 已自动从网上下载该论文 PDF 并补全本地记录。"
                    elif metadata_refreshed:
                        prefix += " 已自动从网上补全该论文元数据。"
                    else:
                        prefix += " 已使用本地记录回答。"
                    if embedded_full_text:
                        prefix += f" 已自动完成全文入库（namespace: {paper_tgt_ns}）。"
                    elif embed_error:
                        prefix += f" 自动全文入库失败：{embed_error[:160]}。"

                    if summary:
                        answer_text = (
                            f"{prefix}\n\n"
                            f"标题：{title}\n"
                            f"arXiv ID：{arxiv_id}\n"
                            f"链接：{url}\n"
                            f"本地PDF：{pdf_path}\n\n"
                            f"基于当前可用摘要，这篇论文主要内容是：\n{summary}\n\n"
                            + (
                                "你现在可以直接继续问方法细节/实验结论。"
                                if embedded_full_text
                                else f"若你要问方法细节/实验结论，请继续执行全文入库（namespace: {paper_tgt_ns}）后提问。"
                            )
                        )
                    else:
                        answer_text = (
                            f"{prefix}\n"
                            f"当前仍缺少可用摘要文本。你可以先执行该论文全文入库（embed）后再进行细节问答。\n"
                            f"- arXiv ID: {arxiv_id}\n"
                            f"- 本地PDF: {pdf_path}\n"
                            f"- namespace: {paper_tgt_ns}"
                        )
                    out = {"answer": answer_text, "citations": []}
                    run_after_agent(self.middleware, out)
                    return out
            except Exception:
                pass

        # 本地 Chroma 无命中：按意图（仅 RAG 主路径）尝试联网摘要兜底
        if (
            not grouped
            and allow_web_search_when_local_misses(route)
            and not self._is_history_query(question)
            and not read_paper_merge_ns
            and not str(session_rag_namespace).startswith("paper:")
            and not image_only_mode
        ):
            from config import RAG_WEB_FALLBACK_ENABLED

            if RAG_WEB_FALLBACK_ENABLED:
                from config import RAG_WEB_FETCH_FIRST_PAGE_MODE, RAG_WEB_MERGED_MAX_RESULTS
                from tools.agent.web_page_fetch import fetch_url_plain_text
                from tools.agent.web_search import search_web_with_subquestions

                # 与本地检索一致：开启子问题拆分时按子问题分别联网搜索，去重合并后写入上下文
                web_items, web_search_note = search_web_with_subquestions(
                    question,
                    self.llm,
                    max_merged_results=RAG_WEB_MERGED_MAX_RESULTS,
                )
                if web_items:
                    web_pairs: list[tuple[object, float]] = []
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
                            body = f"【针对子问题：{subq}】\n{body}"
                        web_pairs.append(
                            (
                                Document(
                                    page_content=body[:8000],
                                    metadata={
                                        "source": url,
                                        "type": "web_search",
                                        "title": title,
                                        "subquestion": subq or None,
                                    },
                                ),
                                0.1 + i * 0.01,
                            )
                        )
                    if web_pairs:
                        # 「今天有没有比赛」类：尝试抓取首条链接正文，补充摘要里没有的当场次信息
                        _fetch = RAG_WEB_FETCH_FIRST_PAGE_MODE
                        _do_fetch = _fetch == "always" or (
                            _fetch == "auto"
                            and needs_temporal_anchor(question)
                            and looks_like_live_schedule_query(question)
                        )
                        if _do_fetch:
                            first_doc = web_pairs[0][0]
                            _u = getattr(first_doc, "metadata", {}).get("source", "")
                            _page = fetch_url_plain_text(_u)
                            if _page:
                                first_doc.page_content = (
                                    f"{getattr(first_doc, 'page_content', '')}\n\n"
                                    f"---\n「页面正文摘录（可能不完整；SPA 站点可能无赛程表格）」\n"
                                    f"{_page}"
                                )[:24000]
                        grouped = self._group_by_parent(web_pairs)
                        web_fallback_used = True
                    else:
                        web_search_note = (
                            web_search_note
                            or "联网返回了条目但无法解析为可用摘要，请稍后重试或检查代理。"
                        )
                if not grouped and web_search_note:
                    # 详细原因放在 web_search_error，由 CLI [Note] 打印，避免与 [Answer] 重复一大段
                    answer_text = (
                        f"本地知识库（namespace: {session_rag_namespace}）未检索到可用于回答的片段；"
                        f"联网摘要兜底未成功。请查看命令行中的 [Note] 了解原因与处理办法。"
                    )
                    if session_id:
                        conversation_manager.add_turn(
                            session_id,
                            "user",
                            question,
                            image_paths=user_image_paths,
                        )
                        conversation_manager.add_turn(
                            session_id, "assistant", answer_text
                        )
                    out = {
                        "answer": answer_text,
                        "citations": [],
                        "web_fallback": False,
                        "web_search_error": web_search_note,
                    }
                    run_after_agent(self.middleware, out)
                    return out

        chat_history = (
            conversation_manager.get_recent_messages(session_id)
            if session_id
            else None
        )
        user_asked_web = user_requests_forced_web_search(question)
        scope_aid = paper_namespace_arxiv_id(
            str(read_paper_merge_ns or "")
        ) or paper_namespace_arxiv_id(str(session_rag_namespace or ""))
        messages = build_rag_prompt(
            question,
            grouped_for_prompt,
            chat_history=chat_history,
            user_image_paths=filtered_user_image_paths,
            include_user_images_in_history=include_user_images_in_history,
            paper_scope_arxiv_id=scope_aid,
        )
        has_any_web_context = (
            web_fallback_used or web_supplement_used or web_forced_by_user
        )
        if (has_any_web_context or user_asked_web) and messages:
            sys0 = messages[0]
            if isinstance(sys0, dict) and sys0.get("role") == "system":
                extra_bits: list[str] = []
                if web_fallback_used:
                    extra_bits.append(
                        "（补充：本地向量库未命中；下列上下文来自网络搜索摘要，可能按子问题分别检索后合并，"
                        "请**综合所有片段**直接回答用户的原始问题，避免只罗列链接；可能有时效性，请提示用户核对来源。）"
                    )
                if web_supplement_used:
                    extra_bits.append(
                        "（补充：本地知识库已有命中，但系统判断检索片段可能不充分，已**额外补充**网络摘要；"
                        "请优先依据本地片段，不足处再结合网络摘要作答，并区分二者来源；网络内容可能有时效性。）"
                    )
                if web_forced_by_user:
                    extra_bits.append(
                        "（补充：用户在提问中**明确要求联网检索**；下列已合并网络搜索摘要（元数据 type 含 web_search_user_requested），"
                        "请必须参考这些片段作答。）"
                    )
                if user_asked_web and has_any_web_context:
                    extra_bits.append(
                        "（**重要**：用户显式要求联网；下文中的网络 URL 来源片段为合法上下文，"
                        "**禁止**再以「无法联网」「不能访问实时信息」「仅基于本地」等理由拒绝使用这些片段。）"
                    )
                elif user_asked_web and not has_any_web_context:
                    extra_bits.append(
                        "（说明：用户要求联网检索，但本次未获得任何网络摘要（可能未开启联网或检索失败）；"
                        "请仅基于当前上下文作答，并简短如实说明未能补充网络信息。）"
                    )
                if extra_bits:
                    extra = "\n\n" + "\n".join(extra_bits)
                    if needs_temporal_anchor(question):
                        extra += "\n\n" + format_temporal_system_note()
                    sys0["content"] = str(sys0.get("content") or "") + extra
        run_before_model(self.middleware, messages)
        config: dict[str, Any] = {}
        if self.middleware:
            try:
                from tools.agent.middleware import AgentCallbackHandler
                config["callbacks"] = [AgentCallbackHandler(self.middleware)]
            except Exception:
                pass
        resp = self.llm.invoke(dict_messages_to_lc(messages), config=config)
        run_after_model(self.middleware, resp)
        answer_text = resp.content if hasattr(resp, "content") else str(resp)
        answer_text = self._with_truncation_reason(resp, str(answer_text))

        if session_id:
            conversation_manager.add_turn(
                session_id, "user", question, image_paths=user_image_paths
            )
            conversation_manager.add_turn(session_id, "assistant", answer_text)

        citations: list[dict] = []
        for doc, score in grouped_for_citations:
            meta = getattr(doc, "metadata", {})
            cit = {
                "source": meta.get("source", "unknown"),
                "score": float(score),
                "preview": getattr(doc, "page_content", "")[:200],
            }
            sq_meta = meta.get("subquestion")
            if sq_meta:
                cit["subquestion"] = sq_meta
            citations.append(cit)

        out = {
            "answer": answer_text,
            "citations": citations,
            "web_fallback": web_fallback_used,
            "web_supplement": web_supplement_used,
            "web_forced_by_user": web_forced_by_user,
            "retrieval_context_score": retrieval_context_score,
            "retrieval_judge_reason": retrieval_judge_reason,
        }
        run_after_agent(self.middleware, out)
        return out

    @staticmethod
    def _format_subq_web_prefetch_system(
        web_items: list[dict], note: str | None
    ) -> str:
        """将 search_web_with_subquestions 结果格式化为 inject 到 Tool 模式的 system 说明。"""
        parts: list[str] = [
            "【子问题拆分 + 联网检索摘要】",
            "以下条目由本机 search_web_snippets 拉取（与 RAG 联网兜底相同链路："
            "RAG_WEB_BACKEND=auto 时优先 MCP Streamable/Brave stdio，再 DuckDuckGo 等）。",
            "每条前若带有「针对子问题」则为拆分后的子查询对应结果。请结合用户问题组织答案，并引用 URL。",
        ]
        if note:
            parts.append(f"[检索备注] {str(note)[:600]}")
        if not web_items:
            parts.append("（本轮未返回可解析的联网条目；可继续用工具或已有知识作答。）")
            return "\n\n".join(parts)
        blocks: list[str] = []
        for i, item in enumerate(web_items[:25], 1):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip() or "web"
            snippet = (item.get("snippet") or "").strip()
            subq = (item.get("subquestion") or "").strip()
            chunk = f"### {i}\n"
            if subq:
                chunk += f"- 针对子问题：{subq}\n"
            chunk += f"- 标题：{title}\n- URL：{url}\n- 摘要：{snippet[:1800]}"
            blocks.append(chunk)
        parts.append("\n\n".join(blocks))
        return "\n\n".join(parts)

    def answer_with_tools(
        self,
        question: str,
        namespace: str = DEFAULT_NAMESPACE,
        k: int = DEFAULT_TOP_K,
        session_id: str | None = None,
        max_tool_rounds: int = 5,
        user_image_paths: list[str] | None = None,
        prefetch_subq_web: bool | None = None,
    ) -> dict:
        """使用 LangChain Tool Calling：由 LLM 决定调用哪些工具（天气 / arXiv / 知识库），多轮直到无 tool_calls。

        prefetch_subq_web:
            None：跟随环境变量 RAG_TOOLS_PREFETCH_SUBQ_WEB；
            True/False：强制开/关「首轮前注入：子问题拆分 + 联网摘要（含 MCP 优先）」。
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        tools = get_agent_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        citations: list[dict] = []

        inp = {
            "question": question,
            "namespace": namespace,
            "session_id": session_id,
            "user_image_paths": user_image_paths,
        }
        run_before_agent(self.middleware, inp)
        if inp.get("_abort"):
            out = inp["_abort"]
            run_after_agent(self.middleware, out)
            return out

        question = inp.get("question", question)
        namespace = inp.get("namespace", namespace)
        user_image_paths = inp.get("user_image_paths", user_image_paths)

        early_ingest = self._maybe_handle_paper_ingest_dialogue(
            question, session_id, user_image_paths=user_image_paths
        )
        if early_ingest is not None:
            run_after_agent(self.middleware, early_ingest)
            return early_ingest

        if self._is_assistant_meta_question(question):
            out = self._answer_assistant_meta_question(
                question, session_id, user_image_paths=user_image_paths
            )
            run_after_agent(self.middleware, out)
            return out

        # 会话代词回指 + 强制 namespace 注入（工具模式同样生效）。
        # 问句已含可绑定标题时勿用历史 arXiv 拼接，避免串篇。
        forced_ns_from_coref = None
        early_title_bind_aid: str | None = None
        try:
            from tools.retrieval.local_paper_service import bind_local_paper_if_mentioned

            _em = bind_local_paper_if_mentioned(question or "")
            if _em and float(_em.get("match_score") or 0.0) >= 0.56:
                early_title_bind_aid = str(_em.get("arxiv_id") or "").strip() or None
        except Exception:
            early_title_bind_aid = None
        if session_id and self._is_paper_coref_query(question) and not early_title_bind_aid:
            recent = conversation_manager.get_recent_messages(session_id)
            last_id = self._extract_last_arxiv_id_from_history(recent)
            if last_id and last_id not in question:
                question = f"{question}（arXiv: {last_id}）"
            forced_ns = self._namespace_for_arxiv_id(last_id)
            if forced_ns:
                namespace = forced_ns
                forced_ns_from_coref = forced_ns

        # 无「这篇论文」等代词，但明显在问论文段落（方法/实验/结论…），且会话已绑定或历史里出现过 arXiv：
        # 注入 paper namespace，避免走 tools 去搜聊天历史。
        if (
            session_id
            and (not early_title_bind_aid)
            and forced_ns_from_coref is None
        ):
            try:
                from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

                if looks_like_paper_content_qa(question or "") and not extract_arxiv_id(
                    question or ""
                ):
                    from tools.retrieval.session_paper_state import paper_state_store

                    cur = paper_state_store.get_current_paper(session_id)
                    last_id = (
                        str(cur.get("arxiv_id") or "").strip()
                        if cur
                        else ""
                    )
                    if not last_id:
                        recent = conversation_manager.get_recent_messages(session_id)
                        last_id = self._extract_last_arxiv_id_from_history(recent) or ""
                    if last_id:
                        if last_id not in (question or ""):
                            question = f"{question}（arXiv: {last_id}）"
                        fn = self._namespace_for_arxiv_id(last_id)
                        if fn:
                            namespace = fn
                            forced_ns_from_coref = fn
            except Exception:
                pass

        arxiv_id_raw = extract_arxiv_id(question or "")
        arxiv_id = arxiv_id_raw
        ql = (question or "").lower()
        route_decision = build_query_route(question)
        current_bound_aid = None
        if session_id:
            try:
                from tools.retrieval.session_paper_state import paper_state_store

                # 工具模式也支持“读第 N 篇”从上一轮本地论文候选中绑定当前论文
                if self._is_paper_read_selection_intent(question):
                    cands = paper_state_store.get_candidates(session_id)
                    idx = self._parse_selection_index(question, len(cands))
                    if idx is not None:
                        chosen = dict(cands[idx - 1] or {})
                        paper_state_store.set_current_paper(session_id, chosen)
                        aid = str(chosen.get("arxiv_id") or "").strip()
                        if aid:
                            current_bound_aid = aid
                if current_bound_aid is None:
                    cur = paper_state_store.get_current_paper(session_id)
                    if cur:
                        aid = str(cur.get("arxiv_id") or "").strip()
                        if aid:
                            current_bound_aid = aid
                if current_bound_aid is None:
                    from tools.retrieval.local_paper_service import bind_local_paper_if_mentioned

                    matched = bind_local_paper_if_mentioned(question)
                    if matched and float(matched.get("match_score") or 0.0) >= 0.72:
                        paper_state_store.set_current_paper(session_id, matched)
                        aid = str(matched.get("arxiv_id") or "").strip()
                        if aid:
                            current_bound_aid = aid
            except Exception:
                current_bound_aid = None

        # 会话已绑定当前篇时，禁止问句里顺带出现的其它 arXiv（引用/对比）抢走检索 namespace。
        if (
            current_bound_aid
            and arxiv_id
            and not self._is_delete_intent(ql)
        ):
            try:
                if _normalize_arxiv_id(current_bound_aid) != _normalize_arxiv_id(arxiv_id):
                    trace_event(
                        "answer_with_tools_lock_skip_secondary_arxiv",
                        {
                            "bound_arxiv_id": _normalize_arxiv_id(current_bound_aid),
                            "ignored_arxiv": _normalize_arxiv_id(arxiv_id),
                            "question": (question or "")[:240],
                        },
                    )
                    arxiv_id = None
            except Exception:
                pass

        # 快速路径 #1b：显式 arXiv ID + 删除意图。
        # 直接路由到 `tool_delete_file`（由中间件拦截人工审批）。
        if arxiv_id_raw and self._is_delete_intent(ql):
            try:
                delete_path = self._pdf_path_for_arxiv_id(arxiv_id_raw)
                delete_tool = next(t for t in tools if t.name == "tool_delete_file")
                args = {"path": delete_path}
                tool_inp = {"session_id": session_id, "question": question, "namespace": namespace}
                for m in self.middleware:
                    if hasattr(m, "before_tool"):
                        m.before_tool("tool_delete_file", args, tool_inp)  # type: ignore[attr-defined]
                if tool_inp.get("_abort"):
                    out = tool_inp["_abort"]
                    run_after_agent(self.middleware, out)
                    return out
                # 若未配置审批中间件，则强制先创建审批单再执行删除，避免直接删。
                if not any(hasattr(m, "before_tool") for m in self.middleware):
                    out = self._build_delete_approval(delete_path, session_id)
                    run_after_agent(self.middleware, out)
                    return out
                answer_text = str(delete_tool.invoke(args))
                if session_id:
                    conversation_manager.add_turn(
                        session_id, "user", question, image_paths=user_image_paths
                    )
                    conversation_manager.add_turn(session_id, "assistant", answer_text)
                out = {"answer": answer_text, "citations": citations}
                run_after_agent(self.middleware, out)
                return out
            except Exception:
                pass

        # 快速路径 #2：论文正文问答应优先走本地 RAG，避免在工具循环中消耗。
        # 当能确定目标论文 namespace 且意图不是下载/全文入库时，
        # 直接基于本地向量库上下文回答。
        target_ns = (
            forced_ns_from_coref
            or (self._namespace_for_arxiv_id(current_bound_aid) if current_bound_aid else None)
            or (self._namespace_for_arxiv_id(arxiv_id) if arxiv_id else None)
            or namespace
        )
        if (
            str(target_ns).startswith("paper:")
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
        ):
            return self.answer(
                question=question,
                namespace=str(target_ns),
                strategy=DEFAULT_RETRIEVAL_STRATEGY,
                k=k,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
                use_multi_source=False,
                session_id=session_id,
                user_image_paths=user_image_paths,
            )

        # 快速路径 #3：论文“找文献”类请求在工具模式下也直接走主流程，
        # 避免 LLM 在工具循环中反复调用而耗尽 max_tool_rounds。
        if (
            getattr(route_decision, "intent", "") == "paper_search"
            and not self._is_download_or_ingest_intent(ql)
            and not self._is_delete_intent(ql)
        ):
            return self.answer(
                question=question,
                namespace=namespace,
                strategy=DEFAULT_RETRIEVAL_STRATEGY,
                k=k,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
                use_multi_source=False,
                session_id=session_id,
                user_image_paths=user_image_paths,
            )

        pr = Path.cwd()
        path_objs: list[Path] = []
        for s in user_image_paths or []:
            p = Path(s)
            if not p.is_absolute():
                p = pr / p
            if p.is_file():
                path_objs.append(p.resolve())
        first_user = build_openai_multimodal_user_content(
            question,
            path_objs,
            max_images=RAG_MAX_IMAGES_PER_MESSAGE,
        )
        from config import RAG_TOOLS_PREFETCH_SUBQ_WEB, RAG_WEB_FALLBACK_ENABLED, RAG_WEB_MERGED_MAX_RESULTS

        if prefetch_subq_web is None:
            do_prefetch = bool(RAG_TOOLS_PREFETCH_SUBQ_WEB) or (
                RAG_WEB_FALLBACK_ENABLED and bool(getattr(route_decision, "needs_web", False))
            )
        else:
            do_prefetch = prefetch_subq_web

        messages: list = []
        web_prefetch_system_text: str | None = None
        if do_prefetch and RAG_WEB_FALLBACK_ENABLED:
            from tools.agent.web_search import search_web_with_subquestions

            web_items, web_note = search_web_with_subquestions(
                question,
                self.llm,
                max_merged_results=RAG_WEB_MERGED_MAX_RESULTS,
            )
            prefetch_sys = self._format_subq_web_prefetch_system(web_items, web_note)
            if prefetch_sys:
                messages.append(SystemMessage(content=prefetch_sys))
                web_prefetch_system_text = prefetch_sys
                trace_event(
                    "tools_subq_web_prefetch",
                    {
                        "items": len(web_items or []),
                        "from_route_needs_web": bool(getattr(route_decision, "needs_web", False)),
                        "question": (question or "")[:200],
                    },
                )
        messages.append(HumanMessage(content=first_user))

        for _ in range(max_tool_rounds):
            run_before_model(self.middleware, messages)
            config: dict[str, Any] = {}
            if self.middleware:
                try:
                    from tools.agent.middleware import AgentCallbackHandler
                    config["callbacks"] = [AgentCallbackHandler(self.middleware)]
                except Exception:
                    pass
            response = llm_with_tools.invoke(messages, config=config)
            run_after_model(self.middleware, response)

            if not getattr(response, "tool_calls", None):
                # 工具调用结束后，再按“多子问题结构化输出”模板做一次合成，
                # 让最终回答把子问题、推理要点与结论组织得更清晰。
                tool_context = "\n\n".join(
                    [
                        str(getattr(m, "content", "") or "")
                        for m in messages
                        if isinstance(m, ToolMessage)
                    ]
                ).strip()
                ctx_for_final = tool_context.strip()
                if not ctx_for_final and web_prefetch_system_text:
                    # 首轮未调工具时，合成阶段仍会丢掉首轮 SystemMessage；必须把联网预取并入最终上下文。
                    ctx_for_final = web_prefetch_system_text.strip()
                if not ctx_for_final:
                    ctx_for_final = "（未提供明确的工具输出；请诚实说明并尽量根据用户问题给出建议）"
                synthesis_user = TOOLS_FINAL_ANSWER_TEMPLATE.format(
                    context=ctx_for_final,
                    question=question,
                )
                final_resp = self.llm.invoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是一位乐于助人的助手。当前处于「工具已执行完毕」阶段："
                                "请输出给用户看的完整结论；对比类、概念类问题允许分点与小标题，写详细、写充分。"
                                "若问题涉及天气或出行气象：须先说明实况再给出多条生活建议，勿过短。"
                                "不要写 MCP/SDK/调用栈等调试向技术复盘。"
                            ),
                        },
                        {"role": "user", "content": synthesis_user},
                    ]
                )
                answer_text = (
                    final_resp.content if hasattr(final_resp, "content") else str(final_resp)
                )
                if session_id:
                    conversation_manager.add_turn(
                        session_id, "user", question, image_paths=user_image_paths
                    )
                    conversation_manager.add_turn(session_id, "assistant", answer_text)
                out = {"answer": answer_text, "citations": citations}
                run_after_agent(self.middleware, out)
                return out

            tool_msgs = []
            for tc in response.tool_calls:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = (tc.get("args") or {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                tid = (tc.get("id") or "") if isinstance(tc, dict) else getattr(tc, "id", "")
                trace_event(
                    "tool_call_start",
                    {"name": name, "args": args, "tool_call_id": tid, "mode": "answer_with_tools"},
                )
                tool_obj = next((t for t in tools if t.name == name), None)
                if tool_obj:
                    if name == "tool_delete_file":
                        path_value = str((args or {}).get("path") or "")
                        if not path_value and arxiv_id:
                            path_value = self._pdf_path_for_arxiv_id(arxiv_id)
                        out = self._build_delete_approval(
                            path_value or "data/papers/unknown.pdf", session_id
                        )
                        run_after_agent(self.middleware, out)
                        return out
                    if name == "tool_search_knowledge":
                        args.setdefault("namespace", namespace)
                        args.setdefault("k", k)
                    # 人工在场（Human-in-the-loop）：允许中间件在工具执行前中断/接管流程。
                    tool_inp = {"session_id": session_id, "question": question, "namespace": namespace}
                    for m in self.middleware:
                        if hasattr(m, "before_tool"):
                            m.before_tool(name, args, tool_inp)  # type: ignore[attr-defined]
                    if tool_inp.get("_abort"):
                        out = tool_inp["_abort"]
                        run_after_agent(self.middleware, out)
                        return out
                    try:
                        content = tool_obj.invoke(args)
                    except Exception as e:
                        content = f"工具执行出错：{e}"
                else:
                    content = f"未知工具：{name}"
                content_str = str(content)
                trace_event(
                    "tool_call_end",
                    {
                        "name": name,
                        "tool_call_id": tid,
                        "output_preview": content_str[:300],
                    },
                )
                if name == "tool_search_knowledge" and content_str and "知识库中未检索到" not in content_str:
                    citations.extend(self._parse_knowledge_tool_citations(content_str))
                tool_msgs.append(ToolMessage(content=content_str, tool_call_id=tid))
            messages = list(messages) + [response] + tool_msgs

        # 到达工具轮数上限：尽量基于已收集的工具输出给出可用答案，而不是直接失败。
        tool_context = "\n\n".join(
            [
                str(getattr(m, "content", "") or "")
                for m in messages
                if isinstance(m, ToolMessage)
            ]
        ).strip()
        if tool_context:
            synthesis_user = TOOLS_FINAL_ANSWER_TEMPLATE.format(
                context=tool_context,
                question=question,
            )
            try:
                final_resp = self.llm.invoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是一位乐于助人的助手。虽然工具调用轮数达到上限，"
                                "但必须基于已有工具/检索结果给出**完整、结构化**的答案；"
                                "对比与概念类问题请分点写充分，并简要说明可能遗漏之处。"
                                "天气类须含实况与多条实用建议，勿过短。"
                            ),
                        },
                        {"role": "user", "content": synthesis_user},
                    ]
                )
                answer_text = (
                    final_resp.content if hasattr(final_resp, "content") else str(final_resp)
                )
            except Exception:
                answer_text = "工具调用达到上限，但已基于现有结果尽力整理答案失败，请重试。"
        else:
            answer_text = "达到最大工具调用轮数，请简化问题后重试。"
        out = {"answer": answer_text, "citations": citations}
        run_after_agent(self.middleware, out)
        return out

    def run_autonomous(
        self,
        *,
        goal: str,
        namespace: str = DEFAULT_NAMESPACE,
        session_id: str | None = None,
        k: int = DEFAULT_TOP_K,
        max_steps: int = 6,
        max_tool_rounds_per_step: int = 4,
        write_memory: bool = True,
        memory_k: int = 6,
    ) -> dict:
        """更自主代理的 MVP：

        规划器（结构化计划）→ 步骤循环（工具执行）→ 长期记忆（摘要+检索）。
        """

        from langchain_core.messages import HumanMessage, ToolMessage

        inp: dict[str, Any] = {
            "goal": goal,
            "namespace": namespace,
            "session_id": session_id,
            "k": k,
            "max_steps": max_steps,
        }
        run_before_agent(self.middleware, inp)
        if inp.get("_abort"):
            out = inp["_abort"]
            run_after_agent(self.middleware, out)
            return out

        goal = inp.get("goal", goal)
        namespace = inp.get("namespace", namespace)
        session_id = inp.get("session_id", session_id)
        k = int(inp.get("k", k))
        max_steps = int(inp.get("max_steps", max_steps))

        # ---- 长期记忆检索（规划前） ----
        memories = retrieve_memories(
            session_id=session_id,
            query=goal,
            k=memory_k,
            score_threshold=0.8,  # 记忆召回：适当放宽阈值
            strategy=DEFAULT_RETRIEVAL_STRATEGY,
        )
        memory_text = format_memories(memories)

        # ---- 规划 ----
        plan = make_plan(
            goal=goal,
            context=f"相关的长期记忆：\n{memory_text}",
            max_steps=max_steps,
        )

        tools = get_agent_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        citations: list[dict] = []
        trace: list[dict] = []

        executor_system = (
            "你是自主执行器。需要时可以调用工具。\n"
            "保持简洁。当你完成某一步时，请清楚写出该步结果。\n"
            "不要编造。若需要工具，请调用对应工具。\n"
        )

        for step in plan.steps[:max_steps]:
            step_prompt = (
                f"目标：{plan.goal}\n\n"
                f"相关的长期记忆：\n{memory_text}\n\n"
                f"当前步骤（{step.id}）：{step.title}\n"
                f"指令：{step.instruction}\n"
            )
            if step.done_criteria:
                step_prompt += f"完成标准：{step.done_criteria}\n"
            step_prompt += (
                f"\n当前工作 namespace：{namespace}\n"
                f"如果需要知识库上下文，请调用 `tool_search_knowledge`（namespace={namespace}）。\n"
            )

            messages: list[Any] = [
                {"role": "system", "content": executor_system},
                HumanMessage(content=step_prompt),
            ]

            step_tool_calls: list[dict] = []
            step_result_text: str | None = None

            for _ in range(max_tool_rounds_per_step):
                run_before_model(self.middleware, messages)
                config: dict[str, Any] = {}
                if self.middleware:
                    try:
                        from tools.agent.middleware import AgentCallbackHandler

                        config["callbacks"] = [AgentCallbackHandler(self.middleware)]
                    except Exception:
                        pass
                response = llm_with_tools.invoke(messages, config=config)
                run_after_model(self.middleware, response)

                # 无工具调用：该步骤已完成（或模型选择不再继续调用工具）。
                if not getattr(response, "tool_calls", None):
                    step_result_text = response.content if hasattr(response, "content") else str(response)
                    break

                tool_msgs: list[Any] = []
                for tc in response.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    args = (tc.get("args") or {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                    tid = (tc.get("id") or "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    trace_event(
                        "tool_call_start",
                        {"name": name, "args": args, "tool_call_id": tid, "mode": "run_autonomous"},
                    )
                    tool_obj = next((t for t in tools if t.name == name), None)
                    if tool_obj:
                        if name == "tool_delete_file":
                            path_value = str((args or {}).get("path") or "")
                            out = self._build_delete_approval(
                                path_value or "data/papers/unknown.pdf", session_id
                            )
                            run_after_agent(self.middleware, out)
                            return out
                        if name == "tool_search_knowledge":
                            args.setdefault("namespace", namespace)
                            args.setdefault("k", k)
                        tool_inp = {"session_id": session_id, "goal": goal, "namespace": namespace}
                        for m in self.middleware:
                            if hasattr(m, "before_tool"):
                                m.before_tool(name, args, tool_inp)  # type: ignore[attr-defined]
                        if tool_inp.get("_abort"):
                            out = tool_inp["_abort"]
                            run_after_agent(self.middleware, out)
                            return out
                        try:
                            content = tool_obj.invoke(args)
                        except Exception as e:
                            content = f"工具执行出错：{e}"
                    else:
                        content = f"未知工具：{name}"

                    content_str = str(content)
                    trace_event(
                        "tool_call_end",
                        {
                            "name": name,
                            "tool_call_id": tid,
                            "output_preview": content_str[:300],
                        },
                    )
                    step_tool_calls.append({"name": name, "args": args, "output_preview": content_str[:500]})
                    if name == "tool_search_knowledge" and content_str and "知识库中未检索到" not in content_str:
                        citations.extend(self._parse_knowledge_tool_citations(content_str))
                    tool_msgs.append(ToolMessage(content=content_str, tool_call_id=tid))

                messages = list(messages) + [response] + tool_msgs

            if step_result_text is None:
                step_result_text = "在工具调用轮数限制内未完成该步骤。"

            trace.append(
                {
                    "step_id": step.id,
                    "title": step.title,
                    "instruction": step.instruction,
                    "done_criteria": step.done_criteria,
                    "tool_calls": step_tool_calls,
                    "result": step_result_text,
                }
            )

        # ---- 最终合成 ----
        trace_context = "\n\n".join(
            [f"- {t['step_id']} {t['title']}: {t['result']}" for t in trace]
        )
        synthesis_prompt = MULTI_QUESTION_REASONING_TEMPLATE.format(
            context=f"执行过程记录（trace）：\n{trace_context}",
            question=plan.goal,
        )
        final_resp = self.llm.invoke(
            [
                {"role": "system", "content": "你是一位乐于助人的助手。请给出最终结果。"},
                {"role": "user", "content": synthesis_prompt},
            ]
        )
        final_answer = final_resp.content if hasattr(final_resp, "content") else str(final_resp)

        # 会话持久化（短期）
        if session_id:
            conversation_manager.add_turn(session_id, "user", goal)
            conversation_manager.add_turn(session_id, "assistant", final_answer)

        memory_written = False
        memory_summary = None
        if write_memory:
            # 为后续运行提炼关键内容并存入长期记忆。
            mem_prompt = (
                "为后续运行创建一段精炼的长期记忆条目。\n"
                "只返回纯文本（不要 JSON）；长度控制在 1200 字符以内。\n"
                "必须包含：\n"
                "- 已发现的关键事实\n"
                "- 做出的决策\n"
                "- 约束/偏好\n"
                "- 待确认问题/下一步（如有）\n\n"
                f"目标：{plan.goal}\n\n"
                "执行过程：\n"
                + "\n\n".join([f"{t['step_id']}: {t['result']}" for t in trace])
                + "\n\n最终回答：\n"
                + final_answer
            )
            mem_resp = self.llm.invoke(
                [
                    {"role": "system", "content": "你在为智能体写入可长期保存的记忆。"},
                    {"role": "user", "content": mem_prompt},
                ]
            )
            memory_summary = mem_resp.content if hasattr(mem_resp, "content") else str(mem_resp)
            try:
                add_memory(
                    session_id=session_id,
                    text=memory_summary,
                    source="agent_run",
                    extra_metadata={"namespace": namespace},
                )
                memory_written = True
            except Exception:
                memory_written = False

        out = {
            "answer": final_answer,
            "citations": citations,
            "plan": {
                "goal": plan.goal,
                "steps": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "instruction": s.instruction,
                        "done_criteria": s.done_criteria,
                    }
                    for s in plan.steps
                ],
            },
            "trace": trace,
            "memory": {
                "retrieved": memories,
                "written": memory_written,
                "summary": memory_summary,
            },
        }
        run_after_agent(self.middleware, out)
        return out

