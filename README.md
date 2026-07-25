# PaperSearchAssistant

面向论文的检索与问答助手：本地论文库 + 混合 RAG + arXiv 入库，提供 FastAPI 服务与 CLI。支持在线/离线双模式，可选 PostgreSQL、Redis、联网补充与 LangGraph 编排。

## 架构

```text
                    ┌─────────────────────────────────────┐
  CLI / HTTP / Web  │  cli.py  ·  main.py (FastAPI)       │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  agent.py  ·  LangGraph（可选）       │
                    │  路由 → 检索 / 工具 → 生成回答         │
                    └─────────────────┬───────────────────┘
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
   tools/agent/              tools/retrieval/              tools/rag/
   arXiv / 联网 / MCP         论文分层检索与问答              切块 · BM25 · 重排
   会话 / 中间件               查询理解 · 融合 · 图表          检索充分性评判
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  tools/storage/                      │
                    │  Chroma 向量库 · SQLite 论文元数据     │
                    │  可选 PostgreSQL / Redis              │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    models_qwen.py（Chat + 本地 Embedding）
```

**请求主路径：** 问题 → 路由/查询理解 → 混合检索（向量 + BM25 + Rerank）→ 上下文不足时可联网 → Prompt 组装 → LLM 生成（带引用与会话记忆）。

| 目录 / 文件 | 职责 |
|-------------|------|
| `main.py` | FastAPI：`/health`、`/embed`、`/search`、`/chat_answer`、论文相关接口 |
| `cli.py` | 命令行：嵌入、检索、问答、论文下载入库 |
| `agent.py` | RAG Agent、工具循环、业务中间件 |
| `config.py` / `runtime_settings.py` | 配置与 `.env.runtime` 加载 |
| `tools/retrieval/` | 论文场景检索、路由、问答组装 |
| `tools/rag/` | 通用 RAG：切块、BM25、重排、评判 |
| `tools/storage/` | Chroma / SQLite / Redis / PostgreSQL |
| `tools/agent/` | arXiv、联网、MCP、会话与中间件 |
| `eval/`、`tests/` | 分层评测（可选） |
| `web/index.html` | 简易前端演示页 |

**运行模式**

- `online`：远程 Chat（如 DashScope）+ 本地 Embedding
- `offline`：本地 OpenAI 兼容 Chat + 本地 Embedding

切换：`source scripts/use_mode.sh online|offline`（生成本地 `.env.runtime`，不入库）。

## 部署

### 环境要求

- Python 3.10+
- 对话 API Key（通义千问兼容 OpenAI 协议，或其他兼容端点）
- 本地 Embedding（默认 `BAAI/bge-m3`，也可改为本机模型路径）
- 可选：Docker（PostgreSQL）、Redis、CUDA

### 1. 安装

```bash
git clone git@github.com:G1999999999/PaperSearchAssistant.git
cd PaperSearchAssistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

```bash
source scripts/use_mode.sh online
```

编辑 `.env.runtime`：

```bash
QWEN_CHAT_API_KEY=你的密钥
QWEN_CHAT_MODEL=qwen3.5-plus
LOCAL_EMBED_MODEL=BAAI/bge-m3
LOCAL_EMBED_DEVICE=cpu          # GPU: cuda 或 cuda:0
```

加载顺序：`.env.runtime` → `.env`；已 `export` 的变量优先。完整可选项见 `.env.online.example` / `.env.offline.example`。

### 3.（可选）PostgreSQL

```bash
docker compose -f docker-compose.postgres.yml up -d
# 端口 15432，账号见 compose 文件

# 写入 .env.runtime：
# DATABASE_URL=postgresql+psycopg://papersearch:papersearch_dev@127.0.0.1:15432/papersearch
# RAG_PG_SYNC_ON_INGEST=1

python scripts/init_pg_schema.py
```

### 4. 启动

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 9000
# 或：python main.py
```

- 健康检查：http://localhost:9000/health
- API 文档：http://localhost:9000/docs
- 前端：打开 `web/index.html`（需服务已启动）

### 5. 常用 CLI

```bash
python cli.py papers --query "retrieval augmented generation" --max-results 5
python cli.py embed_paper_full --arxiv-id 2401.12345
python cli.py chat --namespace default --question "总结本地库里关于 RAG 的要点"
python cli.py chat_paper --arxiv-id 2401.12345 --full --question "方法部分讲了什么"
```

### 6. API 示例

```bash
curl http://localhost:9000/health

curl -X POST http://localhost:9000/chat_answer \
  -H "Content-Type: application/json" \
  -d '{"question": "知识库里有什么重点？", "namespace": "default", "k": 4}'
```

## 本地数据（不提交）

运行后生成于 `data/`（Chroma、PDF、会话等）与 `logs/`，已由 `.gitignore` 忽略。密钥只放在 `.env` / `.env.runtime`，勿提交仓库。
