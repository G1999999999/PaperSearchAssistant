"""
Agent 中间件（Middleware）层。

在 agent 执行的关键节点插入逻辑：before_agent / before_model / after_model / after_agent。
可与 LangChain 的 CallbackHandler 配合，实现日志、鉴权、限流、统计等。

业务中间件说明：
- InputValidationMiddleware：校验/清洗输入，空问题则终止并返回提示。
- CallLimitMiddleware：按 session 或全局限制 LLM 调用次数，超限则终止。
- PIIMaskingMiddleware：对问题中的手机号/身份证/邮箱做占位脱敏后再送 LLM。
- UsageStatsMiddleware：统计请求次数并在输出中附带本次会话统计（可选落盘）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol
from config import RAG_TRACE_ENABLED, RAG_TRACE_LOG_FILE

try:
    from langchain.agents.middleware import SummarizationMiddleware as _LCSummarizationMiddleware
except Exception:
    _LCSummarizationMiddleware = None


class AgentMiddleware(Protocol):
    """Agent 中间件协议：在关键节点被调用。"""

    def before_agent(self, input: dict[str, Any]) -> None:
        """Agent 开始处理前调用（仅一次）。可修改 input 或设置 input["_abort"] 提前终止。"""
        ...

    def before_model(self, messages: list, **kwargs: Any) -> None:
        """每次调用 LLM 前调用。"""
        ...

    def after_model(self, response: Any, **kwargs: Any) -> None:
        """每次 LLM 返回后调用。"""
        ...

    def after_agent(self, output: dict[str, Any]) -> None:
        """Agent 返回最终结果后调用（仅一次）。可修改 output 附加统计等。"""
        ...


# ---------------------------------------------------------------------------
# 业务中间件
# ---------------------------------------------------------------------------


class InputValidationMiddleware:
    """输入校验：去首尾空白、长度限制；空问题则设置 _abort 由 agent 直接返回。"""

    def __init__(self, max_question_length: int = 2000, empty_message: str = "请输入有效问题。") -> None:
        self.max_question_length = max_question_length
        self.empty_message = empty_message

    def before_agent(self, input: dict[str, Any]) -> None:
        q = (input.get("question") or "").strip()
        if not q:
            input["_abort"] = {"answer": self.empty_message, "citations": []}
            return
        if len(q) > self.max_question_length:
            q = q[: self.max_question_length]
        input["question"] = q


class CallLimitMiddleware:
    """调用次数限制：同一 session 超过 N 次则拒绝。可选 persist_file 持久化计数，重启后仍生效。"""

    def __init__(
        self,
        max_calls: int = 20,
        window_seconds: int | None = None,
        persist_file: str | Path | None = "data/call_limits.json",
    ) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.persist_file = Path(persist_file) if persist_file else None
        self._counts: dict[str, int] = defaultdict(int)
        if self.persist_file and self.persist_file.exists():
            try:
                import json as _json
                data = _json.loads(self.persist_file.read_text(encoding="utf-8"))
                self._counts.update(data)
            except Exception:
                pass

    def _save(self) -> None:
        if not self.persist_file:
            return
        try:
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            self.persist_file.write_text(
                _json.dumps(dict(self._counts), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def before_agent(self, input: dict[str, Any]) -> None:
        key = input.get("session_id") or "default"
        self._counts[key] += 1
        self._save()
        if self._counts[key] > self.max_calls:
            input["_abort"] = {
                "answer": f"本会话调用已达上限（{self.max_calls} 次），请稍后再试或更换 session。",
                "citations": [],
            }


class PIIMaskingMiddleware:
    """PII 脱敏：在发送给 LLM 前将问题中的手机号、身份证号、邮箱替换为占位符。"""

    # 简单正则，仅作演示
    _PHONE = re.compile(r"1[3-9]\d{9}")
    _ID_CARD = re.compile(r"\b\d{17}[\dXx]\b")
    _EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def before_agent(self, input: dict[str, Any]) -> None:
        q = input.get("question") or ""
        q = self._PHONE.sub("[PHONE]", q)
        q = self._ID_CARD.sub("[ID_CARD]", q)
        q = self._EMAIL.sub("[EMAIL]", q)
        input["question"] = q


class UsageStatsMiddleware:
    """请求统计：在 after_agent 中记录本次请求并可选写入文件；在 output 中附带 _stats。"""

    def __init__(self, stats_file: str | Path | None = None) -> None:
        self.stats_file = Path(stats_file) if stats_file else None
        self._request_count = 0
        self._session_counts: dict[str, int] = defaultdict(int)
        self._last_session = "default"

    def before_agent(self, input: dict[str, Any]) -> None:
        self._request_count += 1
        self._last_session = input.get("session_id") or "default"
        self._session_counts[self._last_session] += 1

    def after_agent(self, output: dict[str, Any]) -> None:
        output["_stats"] = {
            "total_requests": self._request_count,
            "session_requests": self._session_counts.get(self._last_session, 0),
        }
        if self.stats_file:
            try:
                self.stats_file.parent.mkdir(parents=True, exist_ok=True)
                with self.stats_file.open("a", encoding="utf-8") as f:
                    f.write(f"session={self._last_session}, total={self._request_count}\n")
            except Exception:
                pass


class SummarizationBridgeMiddleware:
    """LangChain SummarizationMiddleware 的桥接层。

    本项目使用自定义的中间件生命周期；不同版本的 LangChain middleware
    API 可能存在差异，因此该桥接层会尽最大努力调用可用的钩子；
    当钩子不可用时会降级为 no-op。
    """

    def __init__(self, enabled: bool = True, **kwargs: Any) -> None:
        self._enabled = bool(enabled)
        self._inner = None
        if not self._enabled or _LCSummarizationMiddleware is None:
            return
        try:
            self._inner = _LCSummarizationMiddleware(**kwargs)
        except Exception:
            # 构造函数签名不匹配时，尽量保持运行稳定。
            self._inner = None

    def before_model(self, messages: list, **kwargs: Any) -> None:
        if self._inner is None:
            return
        # 若存在同名钩子，则优先调用。
        if hasattr(self._inner, "before_model"):
            try:
                self._inner.before_model(messages, **kwargs)
                return
            except Exception:
                return
        # 兜底：当 API 期待 state dict 时使用该参数形式。
        if hasattr(self._inner, "before_agent"):
            try:
                self._inner.before_agent({"messages": messages})
            except Exception:
                return


class HumanApprovalMiddleware:
    """需要人工在场的工具审批。

    通过 interrupt_on 配置哪些工具需要审批：
    - {"tool_delete_file": True} => 允许 approve/edit/reject
    - {"tool_read_file": False} => 不需要审批
    - {"tool_send_email": {"allowed_decisions": ["approve","reject"]}}
    """

    def __init__(self, interrupt_on: dict[str, Any] | None = None) -> None:
        self.interrupt_on = interrupt_on or {}

    def before_tool(self, tool_name: str, tool_args: dict[str, Any], input: dict[str, Any]) -> None:
        cfg = self.interrupt_on.get(tool_name, False)
        if not cfg:
            return

        allowed = ["approve", "edit", "reject"]
        if isinstance(cfg, dict) and cfg.get("allowed_decisions"):
            allowed = list(cfg["allowed_decisions"])

        from tools.agent.approvals import create_approval

        item = create_approval(
            session_id=input.get("session_id"),
            tool_name=tool_name,
            tool_args=tool_args,
            allowed_decisions=allowed,
        )
        input["_abort"] = {
            "answer": (
                "需要人工确认后才能执行该操作。\n"
                f"- approval_id: {item.approval_id}\n"
                f"- tool: {tool_name}\n"
                f"- args: {tool_args}\n"
                f"- allowed_decisions: {allowed}\n"
            ),
            "citations": [],
            "approval_required": {
                "approval_id": item.approval_id,
                "tool": tool_name,
                "args": tool_args,
                "allowed_decisions": allowed,
            },
        }


class LoggingMiddleware:
    """简单日志中间件：在控制台打印各节点（便于调试与面试演示）。"""

    def before_agent(self, input: dict[str, Any]) -> None:
        q = input.get("question", input.get("input", ""))[:80]
        print(f"[Middleware] before_agent: question={q}...")

    def before_model(self, messages: list, **kwargs: Any) -> None:
        print("[Middleware] before_model: invoking LLM")

    def after_model(self, response: Any, **kwargs: Any) -> None:
        content = getattr(response, "content", str(response))[:60]
        print(f"[Middleware] after_model: response={content}...")

    def after_agent(self, output: dict[str, Any]) -> None:
        ans = (output.get("answer") or "")[:60]
        print(f"[Middleware] after_agent: answer={ans}...")


def _safe_preview(v: Any, max_len: int = 400) -> Any:
    """把复杂对象转换为短预览，避免日志爆炸。"""
    try:
        if isinstance(v, (str, int, float, bool)) or v is None:
            s = str(v)
            return s[:max_len] + ("..." if len(s) > max_len else "")
        if isinstance(v, dict):
            out: dict[str, Any] = {}
            for k, val in list(v.items())[:20]:
                out[str(k)] = _safe_preview(val, max_len=max_len)
            return out
        if isinstance(v, list):
            return [_safe_preview(x, max_len=max_len) for x in v[:20]]
        s = str(v)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    except Exception:
        return "<unserializable>"


def trace_event(event: str, payload: dict[str, Any] | None = None) -> None:
    """统一流程日志：控制台 + jsonl 文件（可通过 RAG_TRACE_ENABLED 开关）。"""
    if not RAG_TRACE_ENABLED:
        return
    ts = datetime.now(timezone.utc).isoformat()
    rec = {
        "ts": ts,
        "event": str(event or "").strip() or "unknown",
        "payload": _safe_preview(payload or {}),
    }
    try:
        print(f"[RAG_TRACE] {rec['event']} | {rec['payload']}")
    except Exception:
        pass
    try:
        p = Path(RAG_TRACE_LOG_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_before_agent(middlewares: list[AgentMiddleware], input: dict[str, Any]) -> None:
    trace_event(
        "before_agent",
        {
            "question": str(input.get("question") or "")[:300],
            "namespace": input.get("namespace"),
            "session_id": input.get("session_id"),
        },
    )
    for m in middlewares:
        if hasattr(m, "before_agent"):
            m.before_agent(input)


def run_before_model(middlewares: list[AgentMiddleware], messages: list, **kwargs: Any) -> None:
    trace_event(
        "before_model",
        {
            "messages_count": len(messages or []),
            "last_role": (messages[-1].get("role") if messages and isinstance(messages[-1], dict) else None),
        },
    )
    for m in middlewares:
        if hasattr(m, "before_model"):
            m.before_model(messages, **kwargs)


def run_after_model(middlewares: list[AgentMiddleware], response: Any, **kwargs: Any) -> None:
    content = getattr(response, "content", "")
    trace_event(
        "after_model",
        {
            "has_tool_calls": bool(getattr(response, "tool_calls", None)),
            "content_preview": str(content)[:300],
        },
    )
    for m in middlewares:
        if hasattr(m, "after_model"):
            m.after_model(response, **kwargs)


def run_after_agent(middlewares: list[AgentMiddleware], output: dict[str, Any]) -> None:
    trace_event(
        "after_agent",
        {
            "answer_preview": str(output.get("answer") or "")[:300],
            "citations_count": len(output.get("citations") or []),
            "web_fallback": output.get("web_fallback"),
            "web_supplement": output.get("web_supplement"),
        },
    )
    for m in middlewares:
        if hasattr(m, "after_agent"):
            m.after_agent(output)


def default_business_middleware(
    *,
    validate_input: bool = True,
    max_question_length: int = 2000,
    call_limit: int | None = 20,
    call_limit_persist_file: str | Path | None = "data/call_limits.json",
    pii_mask: bool = True,
    usage_stats: bool = True,
    stats_file: str | Path | None = "data/logs/agent_stats.txt",
    summarization: bool = False,
    summarization_kwargs: dict[str, Any] | None = None,
    session_paper_context: bool = False,
    require_human_approval: bool = True,
    approval_interrupt_on: dict[str, Any] | None = None,
) -> list[AgentMiddleware]:
    """返回一组合适的业务中间件，便于一键启用。"""
    out: list[AgentMiddleware] = []
    if validate_input:
        out.append(InputValidationMiddleware(max_question_length=max_question_length))
    if call_limit is not None:
        out.append(
            CallLimitMiddleware(max_calls=call_limit, persist_file=call_limit_persist_file)
        )
    if pii_mask:
        out.append(PIIMaskingMiddleware())
    if summarization:
        # LangChain SummarizationMiddleware 需要一个摘要模型。
        skw = dict(summarization_kwargs or {})
        if "model" not in skw:
            try:
                from models_qwen import qwen as _default_summary_model

                skw["model"] = _default_summary_model
            except Exception:
                # 如果没有可用模型，则保持禁用（空操作桥接）。
                skw["enabled"] = False
        out.append(SummarizationBridgeMiddleware(**skw))
    if session_paper_context:
        try:
            from tools.agent.session_paper_context import SessionPaperContextMiddleware

            out.append(SessionPaperContextMiddleware())
        except Exception:
            pass
    if require_human_approval:
        out.append(
            HumanApprovalMiddleware(
                interrupt_on=(
                    approval_interrupt_on
                    if approval_interrupt_on is not None
                    else {"tool_delete_file": True}
                )
            )
        )
    if usage_stats:
        out.append(UsageStatsMiddleware(stats_file=stats_file))
    return out


try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # type: ignore[misc, assignment]


class AgentCallbackHandler(BaseCallbackHandler):
    """LangChain 回调：在 LLM 开始/结束时调用中间件的 before_model / after_model。

    传入 invoke(..., config={"callbacks": [AgentCallbackHandler(middlewares)]}) 即可。
    """

    def __init__(self, middlewares: list[AgentMiddleware]) -> None:
        super().__init__()
        self.middlewares = middlewares

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        run_before_model(self.middlewares, prompts, **kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_after_model(self.middlewares, response, **kwargs)
