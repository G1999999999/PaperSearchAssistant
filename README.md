# PaperSearchAssistant

面向论文场景的检索与问答助手：支持本地论文库管理、混合 RAG、arXiv 检索/入库、多轮对话与可选联网补充。提供 FastAPI HTTP 服务与 CLI，并附带分层评测（routing / query understanding / fusion / BM25 / recall）。

## 功能概览

- **论文库**：按 arXiv ID 下载 PDF，写入 SQLite / 可选 PostgreSQL，全文切块后嵌入 Chroma
- **混合检索**：向量 + BM25（RRF）+ CrossEncoder 重排；论文场景支持章节预筛、图表子检索、查询理解与子问题拆分
- **智能路由**：规则 / 混合路由判断本地论文优先、是否联网、工具调用等
- **双模式运行**：`online`（远程 Chat + 本地 Embedding）或 `offline`（本地 OpenAI 兼容 Chat + 本地 Embedding）
- **可选组件**：Redis 会话缓存、PostgreSQL 结构化存储、LangGraph 编排、MCP 联网检索
- **评测**：`eval/` + `tests/layers/` 分层黄金用例，默认离线可跑；集成层需显式开启

## 架构（简图）

```text
用户问题
   │
   ▼
路由 / 查询理解 ──► 本地论文匹配 / 工具意图
   │
   ▼
混合检索（Chroma + BM25 + Rerank）
   │                    ┌─ 不足时可选联网 / MCP
   ▼                    │
组装上下文 + Prompt ───┘
   │
   ▼
LLM 生成（带引用 / 会话记忆）
```

主要入口：

| 路径 | 说明 |
|------|------|
| `main.py` | FastAPI 服务 |
| `cli.py` | 命令行：嵌入、检索、问答、论文入库等 |
| `agent.py` | RAG Agent 与工具循环 |
| `tools/retrieval/` | 论文分层检索与问答 |
| `tools/rag/` | 通用 RAG（切块、BM25、重排、评判） |
| `tools/storage/` | Chroma / SQLite / Redis / PostgreSQL |
| `eval/`、`tests/layers/` | 分层评测 |

更细的运行参数与 FAQ 见 [`mdfile/RUN.md`](mdfile/RUN.md)；评测层级见 [`eval/LAYERS.md`](eval/LAYERS.md)。

## 环境要求

- Python **3.10+**
- 对话模型 API Key（通义千问 DashScope 兼容 OpenAI 协议，或其他兼容端点）
- 本地 Embedding 模型（默认 HuggingFace `BAAI/bge-m3`，也可改为本机路径上的 Qwen Embedding 等）
- （可选）Docker：PostgreSQL；本机 Redis；CUDA（加速 Embedding / Rerank）

## 快速部署

### 1. 克隆与依赖

```bash
git clone git@github.com:G1999999999/PaperSearchAssistant.git
cd PaperSearchAssistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

推荐用模板生成运行时配置（会写入本地 `.env.runtime`，已在 `.gitignore` 中，**不会入库**）：

```bash
# 远程 Chat（DashScope 等）
source scripts/use_mode.sh online

# 或本地 Chat 服务
# source scripts/use_mode.sh offline
```

然后编辑 `.env.runtime`，至少填写：

```bash
QWEN_CHAT_API_KEY=你的密钥
QWEN_CHAT_MODEL=qwen3.5-plus          # 按实际模型名修改
LOCAL_EMBED_MODEL=BAAI/bge-m3         # 或本机模型目录
LOCAL_EMBED_DEVICE=cpu                # 有 GPU 可改为 cuda / cuda:0
```

也可用项目根目录 `.env`；加载顺序为 **`.env.runtime` → `.env`**，已 `export` 的环境变量优先级最高。

### 3.（可选）PostgreSQL

```bash
docker compose -f docker-compose.postgres.yml up -d
# 默认映射 15432 → 5432，账号见 compose 文件

export DATABASE_URL=postgresql+psycopg://papersearch:papersearch_dev@127.0.0.1:15432/papersearch
python scripts/init_pg_schema.py
```

在 `.env.runtime` 中设置 `DATABASE_URL` 与 `RAG_PG_SYNC_ON_INGEST=1` 后，入库会同步 papers / chunks 等表。

### 4. 启动服务

```bash
# HTTP（推荐）
uvicorn main:app --reload --host 0.0.0.0 --port 9000
# 或
python main.py
```

- 健康检查：<http://localhost:9000/health>
- API 文档：<http://localhost:9000/docs>
- 简易前端：打开 `web/index.html`（需服务已启动）

### 5. CLI 常用命令

```bash
# 检索 arXiv
python cli.py papers --query "retrieval augmented generation" --max-results 5

# 按 ID 下载并全文入库（PDF + SQLite + Chroma）
python cli.py embed_paper_full --arxiv-id 2401.12345

# RAG 问答
python cli.py chat --namespace default --question "总结本地库里关于 RAG 的要点"

# 对已入库论文提问
python cli.py chat_paper --arxiv-id 2401.12345 --full --question "方法部分讲了什么"
```

## HTTP API 示例

```bash
curl http://localhost:9000/health

curl -X POST http://localhost:9000/chat_answer \
  -H "Content-Type: application/json" \
  -d '{"question": "知识库里有什么重点？", "namespace": "default", "k": 4}'

curl -X POST http://localhost:9000/search_papers \
  -H "Content-Type: application/json" \
  -d '{"query": "large language models", "max_results": 5}'
```

## 评测

默认只跑可离线、确定性的层（不依赖向量库 / LLM）：

```bash
pytest tests/layers -v
```

需要 Chroma + Embedding 的召回评测：

```bash
export RUN_LAYER_INTEGRATION=1
python -m eval.retrieval_recall_eval
# 或
pytest tests/layers -m integration -v
```

说明与黄金数据格式见 [`eval/LAYERS.md`](eval/LAYERS.md)。

## 数据目录（本地生成，默认不提交）

| 路径 | 说明 |
|------|------|
| `data/chroma` | Chroma 持久化向量库 |
| `data/papers` | PDF 与 `papers.db` |
| `data/conversations` | 会话 JSONL |
| `data/logs` | 链路 trace 等 |
| `logs/` | CLI 聊天记录等 |

首次运行会自动创建所需目录。

## 安全与隐私

上传 / 开源前请确认：

- **不要提交** `.env`、`.env.runtime` 或任何含真实 API Key 的文件（已由 `.gitignore` 忽略）
- 示例文件 `.env.online.example` / `.env.offline.example` 仅含占位符
- **不要提交** `data/`（对话历史、本地 PDF、向量库）与 `logs/`
- 代码中不硬编码个人机器路径或密钥；Embedding 默认使用公开模型名 `BAAI/bge-m3`

## 相关文档

| 文档 | 内容 |
|------|------|
| [`mdfile/RUN.md`](mdfile/RUN.md) | 完整运行说明与环境变量 |
| [`mdfile/PROJECT_MANUAL.md`](mdfile/PROJECT_MANUAL.md) | 项目手册 |
| [`mdfile/技术与算法说明.md`](mdfile/技术与算法说明.md) | 技术与算法说明 |
| [`eval/LAYERS.md`](eval/LAYERS.md) | 分层评测说明 |
| [`INTERVIEW_QA.md`](INTERVIEW_QA.md) | 面试问答整理 |

## License

本仓库用于学习与演示；第三方模型与 API 的使用请遵循各自服务条款。
