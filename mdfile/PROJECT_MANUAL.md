# PaperSearchAssistant2 项目说明书（含技术说明）

## 1. 项目概述

`PaperSearchAssistant2` 是一个面向论文检索与问答的 RAG/Agent 系统，提供：

- 本地知识入库（文本、文件、论文 PDF）
- 向量检索（Chroma，支持多种检索策略）
- 多轮对话与会话持久化
- 工具调用（天气、arXiv、知识库检索等）
- 联网补充检索（含子问题拆分与融合）
- 多模态问答（文本 + 图片）

典型用途：

- 论文阅读助手
- 个人知识库问答
- “本地优先，必要时联网”的智能检索问答

---

## 2. 技术栈

- 后端框架：`FastAPI`
- 模型接入：`langchain_openai`（Qwen OpenAI-compatible API）
- 向量存储：`Chroma`（本地持久化）
- 文档处理：`langchain` 文档/切块组件 + 自定义 loader
- 检索策略：语义检索、BM25、RRF 融合、rerank
- 工具与编排：LangChain Tool Calling + 自定义 `RAGAgent`
- 前端：`web/index.html`（内置轻量聊天页）

---

## 3. 系统架构与目录职责

### 3.1 主要目录

- `main.py`：FastAPI 入口，HTTP API、请求模型、会话接口、文件上传与入库接口
- `agent.py`：核心智能体（路由、检索、fallback、工具调用、回答生成）
- `models_qwen.py`：Qwen 对话模型与 embedding 模型初始化
- `config.py`：全局配置与环境变量解析
- `prompts.py`：系统提示词、RAG 消息构造、多模态历史拼接规则
- `tools/rag/knowledge.py`：向量库封装、检索策略实现
- `tools/rag/retrieval_merge.py`：主分区与公共分区检索融合
- `tools/agent/agent_tools.py`：工具定义（天气、arXiv、知识库等）
- `tools/agent/conversation.py`：会话历史管理与 JSONL 持久化
- `tools/agent/middleware.py`：中间件机制与流程日志
- `tools/agent/session_file_embed.py`：会话文件入库流程
- `web/index.html`：聊天 UI（含图片上传、文件上传、流式显示）

### 3.2 数据落盘位置

- `data/chroma`：向量库数据
- `data/conversations/*.jsonl`：会话历史
- `data/uploads/chat_images`：聊天上传图片
- `data/uploads/session_scratch`：会话文件上传临时文件
- `data/papers`：论文 PDF 与论文数据
- `data/logs/*.txt|*.jsonl`：统计与流程日志

---

## 4. 核心流程说明（技术重点）

## 4.1 入库流程

### 文本入库（`POST /embed`）

1. 接收文本与 namespace
2. 切块（`chunk_size` / `chunk_overlap`）
3. 生成 embedding
4. 写入 Chroma 对应 collection（namespace 对应 collection）

### 文件入库（`POST /session/embed_file`）

1. 上传文件保存到临时目录
2. 读取并抽取文本
3. 生成 `session_ingest_id` 并写入每个 chunk 的 metadata
4. 写入指定 namespace
5. 将 `session_embed` 索引信息写入会话 JSONL（含 ingest_id、namespace、filename）

这样可实现“下一轮聊天优先检索当前会话刚上传文件”的能力。

## 4.2 问答流程（`POST /chat_answer`）

1. 解析请求（含 `images_base64`、`user_image_paths`、`session_id`）
2. 调用 `RAGAgent.answer()`
3. 路由判定（天气/论文/RAG/历史等）
4. 本地检索（可带 session ingest 过滤）
5. 可选：历史上下文充分性评估（LLM judge）
6. 可选：联网搜索补充或兜底
7. 构建 Prompt（含历史、多模态、来源提示）
8. 调用模型生成回答
9. 返回 `answer + citations + web_fallback 等标记`
10. 持久化用户/助手消息到会话 JSONL

## 4.3 路由策略（当前实现）

- 论文相关问题：优先本地检索；本地无命中时调用 arXiv 在线检索
- 非论文问题：优先历史记忆；不足则联网补充
- 天气问题：工具获取结构化天气后，模型改写成自然语言建议
- 含图片问题：走多模态路径，并做“图片与数据库上下文相关性判定”

---

## 5. 检索与排序策略说明

配置入口在 `config.py`，核心实现在 `tools/rag/knowledge.py`。

支持策略：

- `default`（兼容映射到 `hybrid_rerank`）
- `hybrid`：语义检索 + BM25，RRF 融合
- `hybrid_rerank`：混合后再 rerank
- `rerank`：语义初检 + rerank

关键能力：

- 多 query 检索并行（query rewrite / sub-questions）
- score threshold 过滤
- 邻接 chunk 扩展（提升上下文连续性）
- 主分区 + 公共分区融合检索
- 会话上传文件的 ingest_id 过滤优先检索

---

## 6. 多模态与会话机制

## 6.1 图片处理

- 前端将图片以 base64 发送或传路径
- 后端统一落盘为图片文件
- 会话历史只保存 `image_paths`，不保存原始 base64
- Prompt 构造时按策略重放历史图片（可限制轮数）

## 6.2 会话持久化

- `ConversationContextManager` 按 session 管理最近 N 轮
- 同步写入 `data/conversations/<session>.jsonl`
- 支持记录 `session_embed` 索引事件（用于后续检索过滤）

---

## 7. Agent 与 Tool Calling

`RAGAgent.answer_with_tools()` 提供工具调用循环：

1. LLM 判断是否需要调用工具
2. 执行工具（天气、arXiv、知识库、MCP 工具等）
3. 将工具输出回填消息上下文
4. 直到无 tool calls，再进行最终答案合成

同时支持中间件拦截（例如删除工具审批、人机确认）。

---

## 8. 日志与可观测性

已支持流程级 trace 日志（RAG + Agent）：

- 开关：`RAG_TRACE_ENABLED=1`
- 文件：`RAG_TRACE_LOG_FILE=data/logs/rag_agent_trace.jsonl`

日志事件包括：

- `before_agent` / `after_agent`
- `before_model` / `after_model`
- `route_decision`
- `retrieval_done`
- `history_context_judge`
- `image_relevance_decision`
- `paper_fallback_arxiv`
- `tool_call_start` / `tool_call_end`

便于定位“为何走这条路径、命中了什么、是否联网、调用了哪些工具”。

---

## 9. 关键配置项（建议重点关注）

- 模型与网络模式：
  - `RAG_NETWORK_MODE`
  - `QWEN_CHAT_MODEL`
  - `QWEN_MAX_TOKENS`
- 检索：
  - `DEFAULT_RETRIEVAL_STRATEGY`
  - `RAG_LLM_QUERY_REWRITE`
  - `RAG_SUBQUESTION_SPLIT`
  - `RAG_CONTEXT_SCORE_MIN`
  - `RAG_WEB_FALLBACK`
- 多模态：
  - `RAG_MAX_IMAGES_PER_MESSAGE`
  - `RAG_USER_UPLOAD_MAX_IMAGES`
  - `RAG_IMAGE_RELEVANCE_JUDGE_*`
- 会话与日志：
  - `CONVERSATION_MAX_TURNS`
  - `RAG_TRACE_ENABLED`
  - `RAG_TRACE_LOG_FILE`

---

## 10. API 快速索引

- `GET /health`：健康检查
- `POST /embed`：文本入库
- `POST /search`：检索
- `POST /chat_answer`：RAG 问答
- `POST /chat_answer/stream`：流式问答
- `POST /session/embed_file`：会话文件入库
- `GET /sessions`：会话列表
- `GET /sessions/{session_id}`：会话详情
- `POST /search_papers`：arXiv 检索

---

## 11. 运行方式（简版）

1. 配置 `.env.runtime`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动：`uvicorn main:app --host 0.0.0.0 --port 9000`
4. 访问：`/docs` 或内置页面

---

## 12. 常见问题与排障

- 无法回答且提示无上下文：检查 namespace 是否正确、是否已入库、阈值是否过严
- 联网补充未触发：检查 `RAG_WEB_FALLBACK`、网络代理、检索评判模式配置
- 历史对话缺失：确认 `session_id` 一致，查看 `data/conversations/*.jsonl`
- 图片无效：确认文件存在、路径可读、图片格式可解析
- 工具调用异常：查看 trace 日志中的 `tool_call_start/end` 和错误输出

---

## 13. 安全与上线建议

- 生产环境建议增加 API 鉴权（API Key/JWT）
- 对上传文件做更严格的类型和大小限制
- 增加请求级 trace_id 与监控指标（QPS、延迟、错误率）
- 增加单元测试与端到端回归测试

---

## 14. 版本说明

本说明书基于当前仓库代码结构编写，若新增模块或接口，请同步更新本文档与 `README.md`、`RUN.md`。

