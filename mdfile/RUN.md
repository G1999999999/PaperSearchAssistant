# 论文检索助手 运行说明

## 一、环境要求

- Python 3.10+
- 通义千问（Qwen）API Key（或任意 OpenAI 兼容接口的 Key）

---

## 二、安装

```bash
cd SmartSearchAssistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 三、环境变量

推荐使用集中配置文件（优先级高于 `.env`）：

```bash
# 方式一：一键切换模式并加载到当前 shell（推荐）
source scripts/use_mode.sh online
# 或
source scripts/use_mode.sh offline

# 方式二：手动编辑项目根目录 .env.runtime
```

说明：
- `cli.py` 与 `main.py` 都会自动加载：`.env.runtime` -> `.env`
- shell 中已 `export` 的变量优先级最高（可临时覆盖文件配置）

在项目根目录创建 `.env`，或在本机/终端中设置：

```bash
# 必填：通义千问兼容 OpenAI 协议的 Key（对话 + 嵌入都用它）
export QWEN_API_KEY="sk-xxxx"

# 可选：兼容模式 base_url，默认如下
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 可选：单次回复最大 token（默认 4096）。若回答总在「步骤 2」等处突然截断，多半是旧默认 1000 过小，可加大：
# export QWEN_MAX_TOKENS=8192
# export QWEN_TIMEOUT=180

# 可选：本地向量库已有命中时，用 LLM 给检索充分性打分（0-10）；分数低于阈值或模型判 insufficient 时，按模型给出的查询联网补充并与本地合并
# RAG_LLM_CONTEXT_SCORE：on=每轮都评判 | off=关闭 | auto（默认）=由路由决定本轮是否评判（如「仅本地」类表述会跳过）
# export RAG_LLM_CONTEXT_SCORE=auto
# export RAG_LLM_CONTEXT_SCORE=on
# export RAG_CONTEXT_SCORE_MIN=6
# 在 auto 下若希望再加一次「轻量 LLM」判断是否值得跑评判：export RAG_AUTO_JUDGE_USE_LLM=1
# 需同时保持联网兜底开启（RAG_WEB_FALLBACK 未关）。paper: 命名空间不会走此逻辑。

# 用户**明确要求联网**时（如「请联网搜索」「联网检索一下」「search the web」），即使本地已有高相关片段也会再搜网并合并；
# 系统会在 system 中要求模型必须使用网络摘要，避免回答「无法联网」。见 `router.user_requests_forced_web_search`。

# 可选：关闭「检索前 LLM Query 改写」（默认是开启的，见 config.RAG_LLM_QUERY_REWRITE）
# export RAG_LLM_QUERY_REWRITE=0

# 可选：关闭「一次输入多问题时拆成子问题再检索」（默认开启，见 config.RAG_SUBQUESTION_SPLIT）
# export RAG_SUBQUESTION_SPLIT=0
# 可选：子问题数量上限（默认 8）
# export RAG_MAX_SUBQUESTIONS=8

# 可选：关闭 rerank（hybrid_rerank / rerank 会退化为不重排）
# export RAG_ENABLE_RERANK=0

# 可选：指定 reranker 模型（默认 BAAI/bge-reranker-v2-m3）
# export RAG_RERANK_MODEL="BAAI/bge-reranker-v2-m3"

# 论文下载工具默认是否「PDF+SQLite 后再全文向量入库」（论文阅读助手建议保持 1）
# export RAG_ARXIV_DOWNLOAD_DEFAULT_EMBED=0   # 仅 PDF+SQLite，不自动 embed；需全文 RAG 时用 tool_ingest_arxiv_paper

# 可选：关闭「Chroma 无命中后联网摘要兜底」（需 pip install duckduckgo-search）
# export RAG_WEB_FALLBACK=0
# export RAG_WEB_MAX_RESULTS=5
#
# 若本地无命中且联网仍无摘要：CLI 会打印 [Note] 说明原因（未安装依赖 / 网络或代理 /
# DDG 限流等）。国内环境常需 export https_proxy=... 后再运行。
#
# 务必在与 `python cli.py` 相同的解释器里安装依赖，例如：
#   python -m pip install -r requirements.txt
#   # 或仅补装联网检索：
#   python -m pip install 'duckduckgo-search>=6.0.0' 'lxml>=5.0.0'
# 若仍提示「未安装 duckduckgo-search」，说明当前 `python` 与 `pip` 不是同一环境，请用 `which python` / `python -m pip` 对齐。
#
# DuckDuckGo 出现 202 Ratelimit 时：
# - 默认已按 lite → html → api 依次尝试（见 config.RAG_WEB_DDG_BACKENDS，可用环境变量覆盖）
# - 可选自建 SearXNG：export RAG_WEB_SEARXNG_URL='https://你的实例域名'（无尾斜杠）
# - 可选 Brave Search API：export RAG_WEB_BRAVE_API_KEY='...'（见 https://brave.com/search/api/ ）
#
# RAG 联网摘要统一入口（search_web_snippets）默认会「自动择优」：
# - export RAG_WEB_BACKEND=auto（默认）：
#   ① 若已配置 **Streamable HTTP MCP**（如 mcpmarket：MCP_ENABLED 含 **mcpmarket** + MCP_STREAMABLE_URL），
#     且 RAG_WEB_STREAMABLE_FIRST=1（默认），则**最先**走远程 MCP（默认工具名 search_engine，见 MCP_STREAMABLE_TOOL_NAME）；
#   ② 再尝试 MCP Brave **stdio**（MCP_ENABLED 含 search + BRAVE_API_KEY）；
#   ③ 再 DuckDuckGo → 可选 SearXNG → 可选进程内 Brave REST。
# - export RAG_WEB_STREAMABLE_FIRST=0：跳过 ①，仍可走 ②③…。
# - export RAG_WEB_BACKEND=ddg_first：不经过上述两类 MCP，保持旧顺序（先 DDG 再其它）。
# - export RAG_WEB_BACKEND=ddg_only：仅 DuckDuckGo。
#
# Streamable HTTP（mcpmarket / Bright Data 等）常用变量（勿将含密钥的 URL 提交到仓库）：
# - MCP_STREAMABLE_URL=https://mcpmarket.cn/mcp/<你的路径>
# - MCP_ENABLED=filesystem,browser,mcpmarket（需在列表中含 mcpmarket，或与 MCP_STREAMABLE_SERVER_NAME 一致）
# - 可选 MCP_STREAMABLE_BEARER=... 或 MCP_STREAMABLE_HEADERS_JSON='{"Authorization":"Bearer ..."}'
# - 可选 MCP_STREAMABLE_TOOL_NAME=search_engine
# - 可选 MCP_STREAMABLE_SERVER_NAME=mcpmarket（与 MCP_ENABLED 中 token 一致）
#
# 「今天有没有 NBA 比赛」类问题：
# - 联网检索会自动把「今天」展开成具体年月日再搜，并在 system 里注入当前日期（时区可用 RAG_DISPLAY_TIMEZONE=Asia/Shanghai）。
# - 对「相对时间 + 赛程/比赛」类问句，默认会尝试抓取首条搜索结果 URL 的正文（补充摘要）；可设 RAG_WEB_FETCH_FIRST_PAGE=off 关闭，或 =always 每次都抓。
#
# 联网兜底 + 子问题（与本地 RAG 一致，默认 RAG_SUBQUESTION_SPLIT=1 时生效）：
# - 对每个子问题分别发起搜索，按 URL 去重合并进同一次对话上下文，片段前会标注「针对子问题：…」。
# - export RAG_WEB_MERGED_MAX_RESULTS=12   # 合并后最多保留几条摘要
# - export RAG_WEB_SUBQUERY_DELAY_SEC=0.45 # 子问题之间的间隔，缓解限流；设为 0 关闭
# - export RAG_WEB_SUBQUESTION_SPLIT=0     # 强制整句只搜一次（与子问题拆分无关时）
```

质检子问题拆分与检索 query（不访问向量库）：

```bash
python cli.py preview_subquestions --question "RAG 是什么？如何评估检索效果？"
```

说明：对话模型与向量嵌入均使用上述配置，见 `models_qwen.py`。

默认检索为 **向量 + BM25（RRF）+ rerank**：`config.DEFAULT_RETRIEVAL_STRATEGY = hybrid_rerank`，且默认 `RAG_ENABLE_RERANK=1`。若只要纯向量、更快，可在 CLI 传 `--strategy default` 或设 `RAG_ENABLE_RERANK=0`。

### 论文阅读助手：下载后保存到哪？

**可以一次落到三处**（实现见 `tools/agent/paper_ingest.py` → `ingest_arxiv_paper_full_pipeline`）：

| 位置 | 说明 |
|------|------|
| `data/papers/<arxiv_id>.pdf` | 论文 PDF 文件 |
| `data/papers/papers.db`（SQLite） | 标题、作者、摘要、`namespaces` 等元数据 |
| 向量库（Chroma） | 默认 `paper:<id>:full`；若未关 `RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC`，还会写入 `RAG_PUBLIC_NAMESPACE`（常与 `default` 一致），方便在任意会话 namespace 里检索 |

- **普通对话**（`python cli.py chat`，不带 `--use-tools`）：句子里带 **下载 / 保存 / 入库 / embed / 全文入库** 等，且本句或**同一会话历史**里有 **arXiv ID**，会自动调用完整入库（含向量），见 `agent.py` 对 `ingest_arxiv_paper_full_pipeline` 的调用。
- **工具模式**（`--use-tools`）：可用 `tool_ingest_arxiv_paper`；`tool_download_arxiv_pdf` 在默认配置下也会走同一套全文入库。
- **CLI**：`download_paper`、`embed_paper_full` 等（`python cli.py --help`）。
- **HTTP API**：`/papers/download`、`/papers/embed` 等（见 `main.py`）。

---

## 四、启动方式

### 1. HTTP 服务（推荐面试演示）

```bash
cd KnowledgeAssistant
uvicorn main:app --reload --port 9000
```

或直接运行：

```bash
python main.py
```

- 健康检查：<http://localhost:9000/health>
- API 文档：<http://localhost:9000/docs>
- 启动时会自动从 `data/chroma` 加载已持久化的向量库（若存在）。

### 2. 仅用命令行（CLI）

不启动服务，在项目根目录执行：

```bash
python cli.py <子命令> [参数...]
```

---

## 五、常用操作

### 5.1 嵌入本地文件到向量库

```bash
python cli.py embed --namespace default --file notes.txt
# 指定分块参数
python cli.py embed --namespace work --file report.md --chunk-size 500 --chunk-overlap 50
```

### 5.2 向量检索（不调用 LLM）

```bash
python cli.py search --namespace default --query "核心结论是什么" --k 4
# 使用混合检索或重排
python cli.py search --namespace default --query "..." --strategy hybrid --score-threshold 0.5
```

### 5.3 RAG 问答（检索 + LLM 生成）

```bash
python cli.py chat --namespace default --question "总结一下知识库里的要点"
# 多轮对话（同一 session）
python cli.py chat --namespace default --question "第一问" --session-id user1
python cli.py chat --namespace default --question "接着上一句" --session-id user1
# 启用 Tool Calling（由模型选天气/论文/知识库）
python cli.py chat --question "北京天气怎么样" --use-tools
# 启用业务中间件（输入校验、限流、PII 脱敏、统计）
python cli.py chat --question "..." --business-middleware --session-id demo
```

### 5.4 arXiv 论文

```bash
# 只检索，不写入向量库
python cli.py papers --query "retrieval augmented generation" --max-results 5

# 检索后选一篇，把摘要写入向量库
python cli.py embed_paper --query "RAG survey" --index 1 --namespace paper:rag_survey

# 按 arXiv ID 下载 PDF 并把全文写入向量库
python cli.py embed_paper_full --arxiv-id 2401.12345

# 列出本地已入库的论文 collection
python cli.py list_papers

# 对某篇已入库论文提问
python cli.py chat_paper --arxiv-id 2401.12345 --full --question "方法部分讲了什么"
```

### 5.5 多格式文件解读

```bash
python cli.py explain_file --file report.docx --question "请用中文总结重点"
```

### 5.6 最近 N 天知识快照

```bash
python cli.py daily_summary --namespace default --days 1 --k 4
```

---

## 六、HTTP API 示例

以下均在「已启动 uvicorn」的前提下使用。

### 健康检查

```bash
curl http://localhost:9000/health
```

### 嵌入文本

```bash
curl -X POST http://localhost:9000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "这是一段要记住的内容", "namespace": "default"}'
```

### 检索

```bash
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "要点总结", "namespace": "default", "k": 4, "strategy": "default"}'
```

### RAG 问答

```bash
curl -X POST http://localhost:9000/chat_answer \
  -H "Content-Type: application/json" \
  -d '{"question": "知识库里有啥重点？", "namespace": "default", "k": 4}'
```

### 启用业务中间件 + 统计

```bash
curl -X POST http://localhost:9000/chat_answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "总结一下",
    "namespace": "default",
    "use_business_middleware": true,
    "session_id": "api-demo",
    "call_limit": 20
  }'
```

响应中会多出 `stats` 字段（如 `total_requests`、`session_requests`）。

### arXiv 论文检索

```bash
curl -X POST http://localhost:9000/search_papers \
  -H "Content-Type: application/json" \
  -d '{"query": "large language models", "max_results": 5}'
```

---

## 七、数据与目录

| 路径 | 说明 |
|------|------|
| `data/chroma` | Chroma 向量库持久化目录（按 namespace 分 collection） |
| `data/conversations` | 对话历史按 session 的 jsonl 持久化 |
| `data/call_limits.json` | 启用业务中间件时的每 session 调用次数 |
| `data/logs/agent_stats.txt` | 启用业务中间件时的请求统计落盘（可选） |
| `data/papers` | `embed_paper_full` 下载的 PDF 存放目录 |
| `logs/chat_history.jsonl` | CLI 问答的聊天记录（每行一条 JSON） |

---

## 八、常见问题

1. **报错找不到 QWEN_API_KEY**  
   确保已设置环境变量或在项目根目录配置 `.env`，且 `main.py` 顶部已执行 `load_dotenv()`。

2. **Chroma 报错**  
   确认已安装：`pip install chromadb langchain-chroma`。首次运行会自动创建 `data/chroma`。

3. **重排序很慢或报错**  
   重排使用 sentence-transformers，首次会拉取模型。若不需要可改用 `strategy=default` 或 `hybrid`（不做 CrossEncoder 重排）。

4. **想恢复 FAISS**  
   当前默认使用 Chroma；若需改回 FAISS，需改回 `tools/knowledge.py` 的旧实现并安装 `faiss-cpu`。
