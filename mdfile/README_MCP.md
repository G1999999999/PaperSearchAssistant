# PaperSearchAssistant MCP 集成（Server 部署版）

本项目已支持在 **FastAPI 服务进程内**启动/连接 MCP servers，并将 MCP tools 封装成 LangChain `@tool`，供 `/chat_answer?use_tools=true` 直接调用。

## 依赖

- **Python**: 3.10+
- **Node.js**: 18+
- **npx**: 随 Node 安装
- **Playwright 运行依赖**（仅 browser MCP）：
  - 首次运行建议执行：`npx playwright install --with-deps`

Python 依赖（已写入 `requirements.txt`）：

- `mcp>=1.26.0`

## 默认启用的 MCP servers

- **filesystem**：`@modelcontextprotocol/server-filesystem`
- **browser**：`@playwright/mcp`
- **search**（可选）：`@modelcontextprotocol/server-brave-search` — **开放网页/新闻检索**，与本地论文库互补
- **weather**（可选）：本仓库自带 **Python FastMCP** 示例 `mcp_servers/weather_fastmcp.py`（Open-Meteo，无 Key）— 与进程内 LangChain `tool_weather` **对照**，用于演示 MCP 子进程边界

它们都以 **stdio 子进程**方式运行（由 Python MCP client 拉起）。

另有 **Streamable HTTP** MCP（无需 npx 子进程）：在 `MCP_ENABLED` 中加入 **`mcpmarket`**（或与 `MCP_STREAMABLE_SERVER_NAME` 一致），并设置 **`MCP_STREAMABLE_URL`**（例如 mcpmarket 控制台给出的 `https://mcpmarket.cn/mcp/...` 端点）。实现见 [`tools/agent/mcp_runtime.py`](tools/agent/mcp_runtime.py)（`kind=streamable_http`）与 Python SDK `streamable_http_client`。可选 **`MCP_STREAMABLE_BEARER`** / **`MCP_STREAMABLE_HEADERS_JSON`** 传入平台要求的鉴权头。

### RAG 路径与 MCP 搜索的自动择优

当 **`RAG_WEB_BACKEND=auto`（默认）** 时，**普通 RAG 联网兜底**（`answer()`，非 `--use-tools`）在 `tools/agent/web_search.py` 的 `search_web_snippets` 中顺序为：

1. 若 **`RAG_WEB_STREAMABLE_FIRST=1`（默认）** 且已注册 Streamable HTTP server（`MCP_STREAMABLE_URL` + `MCP_ENABLED` 含对应别名），则**先**调用远程搜索工具（默认 **`search_engine`**，可用 `MCP_STREAMABLE_TOOL_NAME` 覆盖）。
2. 若 **`MCP_ENABLED` 包含 `search`** 并已配置 Brave Key，再调用 MCP stdio 的 `brave_web_search` 等。
3. 再回退 **DuckDuckGo / SearXNG / `RAG_WEB_BRAVE_API_KEY` 直连**。

若希望 RAG **不要**走 **Streamable**，设 **`RAG_WEB_STREAMABLE_FIRST=0`**。若希望完全不走 **任何 MCP**，设 **`RAG_WEB_BACKEND=ddg_first`**。

### search（Brave Search）说明

- 在 `MCP_ENABLED` 中加入 **`search`**，并设置 **`BRAVE_API_KEY`**（或 **`MCP_BRAVE_API_KEY`**，会写入子进程环境）。  
  申请 Key：<https://brave.com/search/api/>  
- 未配置 Key 时，使用默认 Brave 包**不会**注册 `search` server（避免 npx 子进程启动失败）。  
- 若改用其它搜索类 MCP，可设 **`MCP_SEARCH_ARGS`**（逗号分隔）覆盖启动参数，并自行在环境中提供该 server 所需的变量。

## 环境变量配置

### 总开关

- `MCP_ENABLED`: 启用哪些 server，逗号分隔。默认 `filesystem,browser`
  - 示例：`MCP_ENABLED=filesystem`
  - 网页搜索（stdio Brave）：`MCP_ENABLED=filesystem,browser,search` 且设置 `BRAVE_API_KEY`
  - 网页搜索（Streamable HTTP，如 mcpmarket）：`MCP_ENABLED=filesystem,browser,mcpmarket` 且设置 `MCP_STREAMABLE_URL`
  - 天气 MCP 示例：`MCP_ENABLED=filesystem,browser,weather`（会拉起 `weather_fastmcp.py`，并注册 `tool_mcp_weather_*`）

Streamable HTTP 补充：

- `MCP_STREAMABLE_URL`：远程 MCP 端点 URL（**路径常含密钥，勿提交仓库**）。
- `MCP_MCPMARKET_URL`：与上一项等价别名。
- `MCP_STREAMABLE_SERVER_NAME`：注册名，默认 `mcpmarket`，须出现在 `MCP_ENABLED` 列表中。
- `MCP_STREAMABLE_TOOL_NAME`：RAG 搜索优先调用的工具名，默认 `search_engine`（Bright Data / mcpmarket 常见）。
- `MCP_STREAMABLE_BEARER` / `MCP_STREAMABLE_HEADERS_JSON`：可选 HTTP 鉴权。

### filesystem server

- `MCP_FILESYSTEM_CMD`: 默认 `npx`
- `MCP_FILESYSTEM_ARGS`: 可覆盖完整 args（逗号分隔）
  - 默认等价于：`-y,@modelcontextprotocol/server-filesystem,<allowed_dirs...>`
- `MCP_FILESYSTEM_ALLOWED_DIRS`: 允许访问的目录白名单（逗号分隔）
  - 默认：`data,data/papers,data/conversations`
- `MCP_FILESYSTEM_CWD`: 可选，子进程工作目录

### browser server

- `MCP_BROWSER_CMD`: 默认 `npx`
- `MCP_BROWSER_ARGS`: 可覆盖完整 args（逗号分隔）
  - 默认：`-y,@playwright/mcp@latest`
- `MCP_BROWSER_CWD`: 可选，子进程工作目录

### search server（Brave Search）

- `BRAVE_API_KEY` 或 `MCP_BRAVE_API_KEY`: Brave Search API Key（使用默认 npm 包时**必填**）
- `MCP_SEARCH_CMD`: 默认 `npx`
- `MCP_SEARCH_ARGS`: 逗号分隔；留空则等价于 `-y,@modelcontextprotocol/server-brave-search`
- `MCP_SEARCH_CWD`: 可选，子进程工作目录

### weather server（本仓库 FastMCP 示例）

- 在 `MCP_ENABLED` 中加入 **`weather`** 即可启用。
- `MCP_WEATHER_CMD`: 默认当前解释器（`sys.executable`）
- `MCP_WEATHER_SCRIPT`: 可选，覆盖天气 MCP 脚本路径（默认：`mcp_servers/weather_fastmcp.py`）
- `MCP_WEATHER_ARGS`: 逗号分隔的完整 argv；**若设置则覆盖默认脚本路径**（高级用法）
- `MCP_WEATHER_CWD`: 子进程工作目录，默认 **PaperSearchAssistant 项目根**（便于 `import tools...`）

暴露的 MCP tool 名：`query_weather`、`season_weather_tips`。LangChain 侧对应薄封装：`tool_mcp_weather_query`、`tool_mcp_weather_season_tips`（与进程内 `tool_weather` 区分）。

### 调用超时

- `MCP_CALL_TIMEOUT_SECONDS`: 默认 `60`

## 已提供的 LangChain Tools

在 `tools/agent_tools.py` 中新增并注册：

- `tool_mcp_list_tools(server="filesystem")`
- `tool_mcp_call(server, tool_name, arguments_json="{}")`（通用入口，建议优先使用）
- `tool_mcp_fs_list_directory(path="data")`
- `tool_mcp_fs_read_text_file(path, head=None, tail=None)`
- `tool_mcp_fs_write_file(path, content)`（会覆盖）
- `tool_mcp_browser_open(url)`
- `tool_mcp_browser_get_title(url="")`
- `tool_mcp_browser_get_text(url="", max_chars=4000)`
- `tool_mcp_browser_screenshot(url="", file_path="data/screenshots/mcp_page.png", full_page=True)`
- `tool_mcp_brave_web_search(query, count=8, offset=0)`（需 `search` server）
- `tool_mcp_brave_news_search(query, count=8)`（优先新闻类 tool，无则回退网页检索）
- `tool_mcp_weather_query(city)`（需 `weather` server）
- `tool_mcp_weather_season_tips(season)`（需 `weather` server）

说明：

- browser 专用工具内部做了常见工具名自动适配（不同 MCP server 命名差异），优先走常见 `navigate/evaluate/screenshot` 族。
- 若适配失败，仍可回退到 `tool_mcp_call` 手动指定 `tool_name + arguments_json`。

## 运行与验证（最小冒烟）

1) 安装依赖

```bash
pip install -r requirements.txt
```

2)（可选）安装 Playwright 依赖（browser MCP）

```bash
npx playwright install --with-deps
```

3) 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

4) 验证 filesystem MCP（列目录）

```bash
curl -s http://127.0.0.1:9000/chat_answer \
  -H 'content-type: application/json' \
  -d '{"question":"请调用 tool_mcp_fs_list_directory 列出 data/papers 下有哪些文件","use_tools":true}' | jq .
```

5) 验证 browser MCP（先列 tools，再调用）

```bash
curl -s http://127.0.0.1:9000/chat_answer \
  -H 'content-type: application/json' \
  -d '{"question":"先调用 tool_mcp_list_tools(server=\"browser\")，然后用 tool_mcp_call 打开 https://example.com 并提取页面标题（如果 browser MCP 提供相应 tool）","use_tools":true}' | jq .
```

## 安全提示

- filesystem server 只能访问 `MCP_FILESYSTEM_ALLOWED_DIRS` 指定的目录；部署时请严格收敛白名单。
- `tool_mcp_fs_write_file` 具备覆盖写能力，建议仅在受控环境开启或在上层加权限/审计。

