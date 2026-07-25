from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


class McpRuntimeError(RuntimeError):
    pass


@dataclass
class McpServerConfig:
    name: str
    kind: Literal["stdio", "streamable_http"] = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str = ""
    headers: dict[str, str] | None = None


def _env_list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _streamable_headers_from_env() -> dict[str, str] | None:
    """可选 HTTP 头：JSON 合并后，再写入 Bearer。"""
    h: dict[str, str] = {}
    raw = (os.getenv("MCP_STREAMABLE_HEADERS_JSON") or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k and v is not None:
                        h[str(k)] = str(v)
        except json.JSONDecodeError:
            pass
    bearer = (os.getenv("MCP_STREAMABLE_BEARER") or "").strip()
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h or None


def _streamable_server_config() -> McpServerConfig | None:
    """mcpmarket / Bright Data 等 Streamable HTTP MCP（需在 MCP_ENABLED 含对应别名，默认 mcpmarket）。"""
    enabled = set(_env_list("MCP_ENABLED", "filesystem,browser"))
    alias = (os.getenv("MCP_STREAMABLE_SERVER_NAME") or "mcpmarket").strip() or "mcpmarket"
    if alias not in enabled:
        return None
    url = (
        os.getenv("MCP_STREAMABLE_URL") or os.getenv("MCP_MCPMARKET_URL") or ""
    ).strip()
    if not url:
        return None
    return McpServerConfig(
        name=alias,
        kind="streamable_http",
        url=url,
        headers=_streamable_headers_from_env(),
    )


def default_server_configs() -> list[McpServerConfig]:
    enabled = set(_env_list("MCP_ENABLED", "filesystem,browser"))
    cfgs: list[McpServerConfig] = []

    # Brave Search MCP：网页/新闻检索（需 API Key；见 README_MCP.md）
    if "search" in enabled:
        cmd = os.getenv("MCP_SEARCH_CMD", "npx")
        args = _env_list("MCP_SEARCH_ARGS", "")
        use_default_brave = not args
        if use_default_brave:
            args = ["-y", "@modelcontextprotocol/server-brave-search"]
        brave = (os.getenv("BRAVE_API_KEY") or os.getenv("MCP_BRAVE_API_KEY") or "").strip()
        env_merge = os.environ.copy()
        if brave:
            env_merge["BRAVE_API_KEY"] = brave
        if use_default_brave and not brave:
            # 官方 Brave MCP 无 Key 无法工作；避免启动即失败的子进程
            pass
        else:
            cfgs.append(
                McpServerConfig(
                    name="search",
                    kind="stdio",
                    command=cmd,
                    args=args,
                    env=env_merge,
                    cwd=os.getenv("MCP_SEARCH_CWD"),
                )
            )

    if "filesystem" in enabled:
        allowed_dirs = _env_list(
            "MCP_FILESYSTEM_ALLOWED_DIRS",
            "data,data/papers,data/conversations",
        )
        cmd = os.getenv("MCP_FILESYSTEM_CMD", "npx")
        args = _env_list("MCP_FILESYSTEM_ARGS", "")
        if not args:
            args = ["-y", "@modelcontextprotocol/server-filesystem", *allowed_dirs]
        cfgs.append(
            McpServerConfig(
                name="filesystem",
                kind="stdio",
                command=cmd,
                args=args,
                env=None,
                cwd=os.getenv("MCP_FILESYSTEM_CWD"),
            )
        )

    if "browser" in enabled:
        cmd = os.getenv("MCP_BROWSER_CMD", "npx")
        args = _env_list("MCP_BROWSER_ARGS", "")
        if not args:
            args = ["-y", "@playwright/mcp@latest"]
        cfgs.append(
            McpServerConfig(
                name="browser",
                kind="stdio",
                command=cmd,
                args=args,
                env=None,
                cwd=os.getenv("MCP_BROWSER_CWD"),
            )
        )

    # 示例：Python FastMCP 天气服务（独立子进程，与进程内 LangChain tool_weather 对照）
    if "weather" in enabled:
        pkg_root = Path(__file__).resolve().parent.parent.parent
        cmd = os.getenv("MCP_WEATHER_CMD", sys.executable)
        args = _env_list("MCP_WEATHER_ARGS", "")
        if not args:
            script = os.getenv("MCP_WEATHER_SCRIPT", "").strip()
            path = Path(script) if script else (pkg_root / "mcp_servers" / "weather_fastmcp.py")
            args = [str(path)]
        cfgs.append(
            McpServerConfig(
                name="weather",
                kind="stdio",
                command=cmd,
                args=args,
                env=None,
                cwd=os.getenv("MCP_WEATHER_CWD") or str(pkg_root),
            )
        )

    sc = _streamable_server_config()
    if sc:
        cfgs.append(sc)

    return cfgs


class _ServerHandle:
    def __init__(self, cfg: McpServerConfig):
        self.cfg = cfg
        self._initialized = False
        self._session = None
        self._exit_stack = None
        self._server_info = None

    async def ensure_connected(self) -> None:
        if self._initialized:
            return
        if self.cfg.kind == "streamable_http":
            await self._ensure_connected_streamable_http()
        else:
            await self._ensure_connected_stdio()

    async def _ensure_connected_stdio(self) -> None:
        try:
            from contextlib import AsyncExitStack

            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except Exception as exc:  # pragma: no cover
            raise McpRuntimeError(
                "MCP python SDK not available. Please `pip install mcp>=1.26.0`."
            ) from exc

        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self.cfg.command,
                args=self.cfg.args,
                env=self.cfg.env,
                cwd=self.cfg.cwd,
            )
            _streams = await stack.enter_async_context(stdio_client(params))
            read, write = _streams[0], _streams[1]
            session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
            init = await session.initialize()
            self._server_info = init.serverInfo if hasattr(init, "serverInfo") else None
            self._session = session
            self._exit_stack = stack
            self._initialized = True
        except Exception:
            await stack.aclose()
            raise

    async def _ensure_connected_streamable_http(self) -> None:
        try:
            from contextlib import AsyncExitStack

            import httpx
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except Exception as exc:  # pragma: no cover
            raise McpRuntimeError(
                "MCP Streamable HTTP 需要 mcp>=1.26 与 httpx。请 `pip install mcp httpx httpx-sse`。",
            ) from exc

        url = (self.cfg.url or "").strip()
        if not url:
            raise McpRuntimeError("streamable_http MCP：url 为空")

        base_headers: dict[str, str] = {
            "Accept": "application/json, text/event-stream",
        }
        if self.cfg.headers:
            base_headers.update(self.cfg.headers)

        timeout_sec = float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "60"))
        timeout = httpx.Timeout(
            connect=30.0,
            read=max(30.0, timeout_sec),
            write=max(30.0, timeout_sec),
            pool=10.0,
        )

        stack = AsyncExitStack()
        try:
            client = httpx.AsyncClient(headers=base_headers, timeout=timeout, follow_redirects=True)
            await stack.enter_async_context(client)
            transport_cm = streamable_http_client(
                url,
                http_client=client,
                terminate_on_close=True,
            )
            streams = await stack.enter_async_context(transport_cm)
            read, write = streams[0], streams[1]
            session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
            init = await session.initialize()
            self._server_info = init.serverInfo if hasattr(init, "serverInfo") else None
            self._session = session
            self._exit_stack = stack
            self._initialized = True
        except Exception:
            await stack.aclose()
            raise

    async def list_tools(self) -> list[dict]:
        await self.ensure_connected()
        assert self._session is not None
        res = await self._session.list_tools()
        tools = getattr(res, "tools", None) or []
        out: list[dict] = []
        for t in tools:
            out.append(
                {
                    "name": getattr(t, "name", None),
                    "description": getattr(t, "description", None),
                    "inputSchema": getattr(t, "inputSchema", None),
                }
            )
        return out

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        await self.ensure_connected()
        assert self._session is not None
        args = arguments or {}
        res = await self._session.call_tool(tool_name, args)
        # res.content 通常是文本/图片内容等条目的列表
        return res

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._initialized = False
        self._session = None
        self._exit_stack = None
        self._server_info = None


class McpRuntime:
    """对 MCP 客户端的同步友好封装。

    - stdio：启动 MCP servers 子进程（npx / python ...）。
    - streamable_http：连接远程 MCP（如 mcpmarket Streamable HTTP），与 stdio 可并存。
    - 在后台线程中维护事件循环，供 LangChain 工具同步调用。
    """

    def __init__(self, server_configs: list[McpServerConfig] | None = None) -> None:
        self._cfgs = server_configs or default_server_configs()
        self._servers: dict[str, _ServerHandle] = {c.name: _ServerHandle(c) for c in self._cfgs}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop and self._loop.is_running():
                return self._loop

            loop = asyncio.new_event_loop()

            def _run():
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_run, name="mcp-runtime-loop", daemon=True)
            t.start()
            self._loop = loop
            self._thread = t
            return loop

    def _run(self, coro):
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "60")))

    def enabled_servers(self) -> list[str]:
        return list(self._servers.keys())

    def list_tools(self, server: str) -> list[dict]:
        h = self._servers.get(server)
        if not h:
            raise McpRuntimeError(f"Unknown MCP server: {server}")
        return self._run(h.list_tools())

    def call_tool(self, server: str, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        h = self._servers.get(server)
        if not h:
            raise McpRuntimeError(f"Unknown MCP server: {server}")
        return self._run(h.call_tool(tool_name, arguments))

    def close(self) -> None:
        for h in self._servers.values():
            try:
                self._run(h.close())
            except Exception:
                pass


mcp_runtime = McpRuntime()

