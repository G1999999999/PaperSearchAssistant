# PaperSearchAssistant

一个**论文检索与问答**后端：你可以把 arXiv 论文下载到本地，切块后写入向量库，再用自然语言提问；也可以把普通文档放进知识库做 RAG 问答。提供 HTTP API 和命令行两种用法。

适合：想本地跑通「论文入库 → 检索 → 问答」链路的开发者或研究者。

---

## 你能用它做什么

| 能力 | 说明 |
|------|------|
| 搜论文 | 按关键词检索 arXiv，查看标题、摘要、链接 |
| 入库 | 按 arXiv ID 下载 PDF，保存元数据，并把全文嵌入向量库 |
| 问论文 | 对已入库的某篇论文提问（例如「方法部分讲了什么」） |
| 通用 RAG | 把本地 txt/pdf/docx 等嵌入知识库后问答，返回答案与引用 |
| 可选联网 | 本地检索不够时，可自动补充网页摘要（需网络） |

**最小可用路径（建议第一次按这个走）：**  
配置 API Key → 启动服务 → 用 CLI 下载一篇论文 → 对这篇论文提问。

---

## 系统如何工作（读懂再部署）

可以把整系统想成三条线：**入口 → 大脑 → 存储**。

```text
你（浏览器 / curl / CLI）
        │
        ▼
┌───────────────────┐
│  HTTP: main.py    │  对外提供 REST API（默认端口 9000）
│  CLI:  cli.py     │  不启动服务也能做入库 / 问答
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Agent / 路由      │  判断：本地论文？知识库？要不要联网？要不要调工具？
│  混合检索 + LLM    │  向量检索 + 关键词（BM25）+ 重排 → 拼上下文 → 大模型回答
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  本地数据          │  data/chroma（向量）  data/papers（PDF + SQLite）
│  （可选）PG / Redis │  结构化论文表 / 会话缓存，没有也能跑
└───────────────────┘
```

**模型怎么配：**

- **对话（Chat）**：默认走远程 OpenAI 兼容接口（例如阿里云通义 DashScope）。也支持你自己在本机起的兼容服务（`offline` 模式）。
- **向量（Embedding）**：始终在**本机**加载（默认 HuggingFace 模型名 `BAAI/bge-m3`）。首次运行会下载模型，体积较大，请预留磁盘与时间；有 GPU 可把设备改成 `cuda`。

**配置文件：**  
项目通过 `.env.runtime`（或 `.env`）读环境变量。仓库里只有示例模板，**不会**提交你的密钥。

---

## 部署前准备

请确认本机具备：

1. **Python 3.10+**
2. **对话 API Key**（通义千问 / 其他 OpenAI 兼容服务均可）
3. 能访问 HuggingFace（或已有本地 Embedding 模型目录）；国内网络可能需要代理
4. （可选）NVIDIA GPU + CUDA，用于加速 Embedding / 重排
5. （可选）Docker，若要用 PostgreSQL

磁盘：Embedding 模型 + 依赖，建议预留数 GB 以上空间。

---

## 部署步骤

### 第 1 步：获取代码并安装依赖

```bash
git clone git@github.com:G1999999999/PaperSearchAssistant.git
cd PaperSearchAssistant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

若 `pip` 下载 HuggingFace / PyPI 很慢，请先配置好本机代理或镜像，再继续。

### 第 2 步：生成配置并填写密钥

推荐用脚本从模板生成运行配置（会覆盖写入项目根目录的 `.env.runtime`）：

```bash
# 远程对话模型（最常见）
source scripts/use_mode.sh online

# 若 Chat 也跑在本机 OpenAI 兼容服务上，改用：
# source scripts/use_mode.sh offline
```

用编辑器打开 `.env.runtime`，至少改这几项：

```bash
# 必填：你的对话 API Key
QWEN_CHAT_API_KEY=sk-xxxxxxxx

# 对话模型名（按你账号实际可用的模型填写）
QWEN_CHAT_MODEL=qwen3.5-plus

# 对话服务地址（通义兼容模式示例；换服务商就改这里）
QWEN_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 本机向量模型：可用 HuggingFace 模型名，或本机绝对路径
LOCAL_EMBED_MODEL=BAAI/bge-m3
LOCAL_EMBED_DEVICE=cpu             # 有 GPU 时改为 cuda 或 cuda:0
```

说明：

- `source scripts/use_mode.sh ...` 必须用 `source`（或 `.`），这样变量才会进当前终端。
- 更完整的开关（联网、子问题拆分、LangGraph 等）都在 `.env.online.example` / `.env.offline.example` 里，按需抄到 `.env.runtime`。
- **不要把填好密钥的 `.env.runtime` 提交到 Git。**

### 第 3 步：启动 HTTP 服务

```bash
# 确保仍在虚拟环境中
uvicorn main:app --host 0.0.0.0 --port 9000
```

浏览器打开：

- 健康检查：http://localhost:9000/health  
- 交互式 API 文档：http://localhost:9000/docs  
- 简易网页：用浏览器打开仓库里的 `web/index.html`（需上面服务已启动）

命令行自检：

```bash
curl http://localhost:9000/health
```

### 第 4 步：完成一次端到端试用（CLI）

另开一个终端，进入项目目录并 `source .venv/bin/activate`，然后：

```bash
# 1）在 arXiv 上搜论文（不入库）
python cli.py papers --query "retrieval augmented generation" --max-results 5

# 2）按 arXiv ID 下载 PDF + 全文写入向量库（首次会较慢）
python cli.py embed_paper_full --arxiv-id 2401.12345

# 3）针对这篇论文提问
python cli.py chat_paper --arxiv-id 2401.12345 --full --question "这篇论文的主要贡献是什么？"

# 4）对通用知识库提问（需你先 embed 过内容）
python cli.py chat --namespace default --question "总结一下知识库里的要点"
```

把 `2401.12345` 换成你真实感兴趣的 arXiv ID。

HTTP 问答示例：

```bash
curl -X POST http://localhost:9000/chat_answer \
  -H "Content-Type: application/json" \
  -d '{"question": "知识库里有什么重点？", "namespace": "default", "k": 4}'
```

---

## 可选：PostgreSQL（增强结构化存储）

**不配也能用。** 配置后，论文分块/章节等可同步进 Postgres，便于更复杂的论文检索。

```bash
docker compose -f docker-compose.postgres.yml up -d
```

在 `.env.runtime` 增加（注意 compose 默认把容器 5432 映射到宿主机 **15432**）：

```bash
DATABASE_URL=postgresql+psycopg://papersearch:papersearch_dev@127.0.0.1:15432/papersearch
RAG_PG_SYNC_ON_INGEST=1
```

然后建表：

```bash
python scripts/init_pg_schema.py
```

Redis 同理：未配置时用本地 JSONL 存会话；需要时再设置 `REDIS_URL`。

---

## 目录速览（方便对照代码）

| 路径 | 第三方关心什么 |
|------|----------------|
| `main.py` | HTTP 服务入口 |
| `cli.py` | 命令行入口 |
| `agent.py` | 问答与工具调用主逻辑 |
| `.env.*.example` | 配置模板（复制思路，勿提交真实密钥） |
| `scripts/use_mode.sh` | 一键生成 `.env.runtime` |
| `tools/rag/` | 通用检索与重排 |
| `tools/retrieval/` | 论文场景检索 |
| `tools/storage/` | 向量库与数据库 |
| `data/` | **运行后自动生成**，含向量库与 PDF，默认不进 Git |
| `web/index.html` | 简单前端演示 |

---

## 常见问题

**1. 提示缺少 API Key**  
检查是否已 `source scripts/use_mode.sh online`，且 `.env.runtime` 里 `QWEN_CHAT_API_KEY` 不是占位符；在**同一个终端**里启动 `uvicorn` / `cli.py`。

**2. 第一次问答很慢 / 卡住**  
多半在下载 Embedding 或 Rerank 模型。看终端是否在拉 HuggingFace 权重；网络不通时请配置代理，或把 `LOCAL_EMBED_MODEL` 改成本机已有模型目录。

**3. Embedding / CUDA 报错**  
先把 `LOCAL_EMBED_DEVICE=cpu` 跑通，再改 `cuda`。多卡时注意 `CUDA_VISIBLE_DEVICES` 与 `cuda:0` 的对应关系。

**4. 论文问答说库里没有**  
先确认 `embed_paper_full` 成功结束；再用 `python cli.py list_papers` 看本地是否已有该论文命名空间。

**5. 只要最简部署，要不要 Postgres / Redis / MCP？**  
不要。按上文第 1～4 步即可；高级组件都是可选增强。

---

## 许可与第三方服务

本项目用于学习与演示。使用通义、HuggingFace、arXiv 等服务时，请遵守各服务的条款与配额限制。
