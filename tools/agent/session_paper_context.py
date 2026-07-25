"""
会话级论文上下文：持久化「涉及哪些 arXiv、主要在聊什么」，下次同 session 启动时自动重新镜像到会话 namespace。

与 `data/conversations/{session}_paper_context.json` 同目录策略一致；不修改全局论文库。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONVERSATION_MAX_TURNS, SESSION_PAPER_CONTEXT_LLM_SUMMARY
from tools.agent.conversation import conversation_manager
from tools.agent.router import find_all_arxiv_ids
from tools.rag.knowledge import vector_store

_CONTEXT_SUFFIX = "_paper_context.json"
_PDF_ARXIV_RE = re.compile(
    r"papers[/\\](\d{4}\.\d{4,5})(?:v\d+)?\.pdf",
    re.IGNORECASE,
)


def _slug(session_id: str) -> str:
    return session_id.replace("/", "_").replace("\\", "_") or "default"


def context_file_path(session_id: str) -> Path | None:
    pd = conversation_manager.persist_dir
    if not pd:
        return None
    return Path(pd) / f"{_slug(session_id)}{_CONTEXT_SUFFIX}"


def _dialogue_snippet_from_messages(
    messages: list[dict],
    *,
    max_messages: int = 14,
    per_msg_chars: int = 240,
) -> str:
    lines: list[str] = []
    for m in messages[-max_messages:]:
        role = str((m or {}).get("role") or "")
        content = str((m or {}).get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content[:per_msg_chars]}")
    return "\n".join(lines)


def _llm_summarize_session_topic(
    *,
    arxiv_ids: list[str],
    dialogue_snippet: str,
    last_answer_preview: str,
) -> str | None:
    """用对话模型生成 2～4 句中文会话主题摘要；失败返回 None。"""
    if not dialogue_snippet.strip() and not last_answer_preview.strip():
        return None
    try:
        from langchain_core.messages import HumanMessage

        from models_qwen import qwen

        ids_txt = ", ".join(arxiv_ids) if arxiv_ids else "（未检测到 arXiv ID）"
        prompt = (
            "你是会话摘要助手。根据下列对话摘录，用中文写 2～4 句话，概括：\n"
            "1）用户主要在讨论什么；\n"
            "2）涉及哪些论文或技术主题（可与给出的 arXiv 列表对应）。\n"
            "要求：忠实于摘录，不要编造未出现的内容；不要列提纲，写成连贯段落。\n\n"
            f"涉及的 arXiv ID：{ids_txt}\n\n"
            f"对话摘录：\n{dialogue_snippet}\n\n"
            f"本轮助手回复开头：\n{last_answer_preview[:800]}"
        )
        resp = qwen.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=512,
            temperature=0.2,
        )
        text = (getattr(resp, "content", None) or str(resp) or "").strip()
        return text if text else None
    except Exception:
        return None


def persist_session_paper_context(
    session_id: str,
    namespace: str,
    *,
    question: str = "",
    answer: str = "",
    citations: list[dict[str, Any]] | None = None,
    topic_window_chars: int = 1200,
    use_llm_topic_summary: bool | None = None,
) -> Path | None:
    """根据当前轮参数 + 已持久化对话，写入 ``<session>_paper_context.json``。

    供中间件 ``after_agent`` 与 CLI ``session_finalize`` 共用。
    返回写入路径；无法写入时返回 None。
    """
    sid = (session_id or "").strip()
    ns = (namespace or "").strip()
    if not sid or not ns:
        return None
    path = context_file_path(sid)
    if path is None:
        return None

    use_llm = (
        SESSION_PAPER_CONTEXT_LLM_SUMMARY
        if use_llm_topic_summary is None
        else bool(use_llm_topic_summary)
    )
    tw = max(200, int(topic_window_chars))

    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    ids_ordered: list[str] = list(prev.get("arxiv_ids") or [])
    seen = set(ids_ordered)

    q = (question or "").strip()
    for aid in find_all_arxiv_ids(q):
        if aid not in seen:
            seen.add(aid)
            ids_ordered.append(aid)

    recent_list: list[dict] = []
    try:
        recent_list = list(conversation_manager.get_recent_messages(sid))
        tail = recent_list[-(CONVERSATION_MAX_TURNS * 4) :]
        for m in tail:
            text = str((m or {}).get("content") or "")
            for aid in find_all_arxiv_ids(text):
                if aid not in seen:
                    seen.add(aid)
                    ids_ordered.append(aid)
    except Exception:
        recent_list = []

    for cit in citations or []:
        if not isinstance(cit, dict):
            continue
        for aid in _arxiv_from_citation(cit):
            aid = aid.strip()
            if aid and aid not in seen:
                seen.add(aid)
                ids_ordered.append(aid)

    # 滚动条以「上一轮 rolling」为基底，避免把整段 LLM 摘要当用户行反复拼接
    old_roll = (prev.get("topic_summary_rolling") or "").strip()
    if not old_roll:
        old_roll = (prev.get("topic_summary") or "").strip()
    topic_line = f"用户：{q[:400]}" if q else ""
    if topic_line:
        merged = (old_roll + "\n" + topic_line).strip()
    else:
        merged = old_roll
    if len(merged) > tw:
        merged = merged[-tw:]

    topic_final = merged
    dialogue_snippet = _dialogue_snippet_from_messages(recent_list)

    ans_prev = (answer or "").strip()
    if not ans_prev and recent_list:
        for m in reversed(recent_list):
            if str((m or {}).get("role") or "") == "assistant":
                ans_prev = str((m or {}).get("content") or "").strip()
                break

    if use_llm:
        llm_txt = _llm_summarize_session_topic(
            arxiv_ids=ids_ordered,
            dialogue_snippet=dialogue_snippet or topic_line or "（无对话摘录）",
            last_answer_preview=ans_prev,
        )
        if llm_txt:
            topic_final = llm_txt

    payload = {
        "namespace": ns,
        "arxiv_ids": ids_ordered,
        "topic_summary": topic_final,
        "topic_summary_rolling": merged,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        return None


def finalize_session_paper_context(session_id: str, namespace: str) -> Path | None:
    """结束会话时调用：不写新用户句，仅按当前 jsonl 历史刷新摘要与 arXiv 列表。"""
    return persist_session_paper_context(
        session_id,
        namespace,
        question="",
        answer="",
        citations=None,
    )


def _arxiv_from_citation(cit: dict[str, Any]) -> list[str]:
    out: list[str] = []
    src = str(cit.get("source") or "")
    prev = str(cit.get("preview") or "")
    for chunk in (src, prev):
        for m in _PDF_ARXIV_RE.finditer(chunk):
            out.append(m.group(1))
        out.extend(find_all_arxiv_ids(chunk))
    return out


class SessionPaperContextMiddleware:
    """before_agent：同进程内首次对该 (session,namespace) 从快照重新镜像论文。
    after_agent：把本轮及历史中的 arXiv ID + 话题摘要写入快照。
    """

    def __init__(
        self,
        *,
        topic_window_chars: int = 1200,
        use_llm_topic_summary: bool | None = None,
    ) -> None:
        self.topic_window_chars = max(200, int(topic_window_chars))
        self.use_llm_topic_summary = (
            SESSION_PAPER_CONTEXT_LLM_SUMMARY
            if use_llm_topic_summary is None
            else bool(use_llm_topic_summary)
        )
        self._last_input: dict[str, Any] = {}
        self._mirrored_keys: set[str] = set()

    def before_agent(self, input: dict[str, Any]) -> None:
        self._last_input = dict(input)
        sid = (input.get("session_id") or "").strip()
        ns = (input.get("namespace") or "").strip()
        if not sid or not ns:
            return
        key = f"{sid}\n{ns}"
        if key in self._mirrored_keys:
            return
        path = context_file_path(sid)
        if path is None or not path.exists():
            self._mirrored_keys.add(key)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._mirrored_keys.add(key)
            return
        if (data.get("namespace") or "").strip() != ns:
            self._mirrored_keys.add(key)
            return
        ids = list(data.get("arxiv_ids") or [])
        if not ids:
            self._mirrored_keys.add(key)
            return
        from tools.agent.paper_session_mirror import mirror_local_paper_to_namespace

        for aid in ids:
            try:
                mirror_local_paper_to_namespace(
                    vector_store, ns, str(aid), replace=True
                )
            except Exception:
                pass
        self._mirrored_keys.add(key)

    def before_model(self, messages: list, **kwargs: Any) -> None:
        return

    def after_model(self, response: Any, **kwargs: Any) -> None:
        return

    def after_agent(self, output: dict[str, Any]) -> None:
        inp = self._last_input or {}
        sid = (inp.get("session_id") or "").strip()
        ns = (inp.get("namespace") or "").strip()
        if not sid or not ns:
            return
        cit = output.get("citations") or []
        if not isinstance(cit, list):
            cit = []
        persist_session_paper_context(
            sid,
            ns,
            question=str(inp.get("question") or ""),
            answer=str(output.get("answer") or ""),
            citations=[c for c in cit if isinstance(c, dict)],
            topic_window_chars=self.topic_window_chars,
            use_llm_topic_summary=self.use_llm_topic_summary,
        )
