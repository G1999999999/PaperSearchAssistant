# SmartSearchAssistant 简历描述（工作内容突出版）

可直接复制到简历「项目经历」一栏，按需删减或微调表述。

---

## 项目名称

**SmartSearchAssistant（智能搜索助手）** | Python / FastAPI / LangChain / Chroma

---

## 项目简介（1 行）

基于个人知识库的智能搜索与问答后端：支持文档嵌入、多策略检索、RAG 问答、arXiv 论文检索、多轮对话及工具调用，提供 HTTP API 与 CLI，配套简易 Web 演示页。

---

## 工作内容 / 项目职责（建议保留 5～7 条）

- **RAG 全链路设计与实现**：负责文档切块（TokenTextSplitter）、向量嵌入（Qwen Embeddings）、Chroma 持久化与按 namespace 分 collection；设计并实现「检索 → 上下文拼接 → 提示词组装 → LLM 生成」的完整流程，返回带来源与得分的引用信息（citations）。
- **检索策略与优化**：实现多 query 检索、得分阈值过滤、BM25 与语义检索的 RRF 融合（hybrid）、以及 CrossEncoder 重排序（rerank）；对空 namespace / 空 query 做边界处理，保证接口稳定。
- **提示词工程**：在 `prompts.py` 中维护系统提示词与 RAG 消息构造逻辑；设计「仅依据上下文回答、未知则明确说明」的约束，并预留加强版 SYSTEM_PROMPT_DETAILED（输出格式、引用规范、语言与安全规则）便于扩展。
- **工具与中间件**：使用 LangChain `@tool` 封装知识库检索、天气查询、arXiv 论文检索等能力，支持规则路由与 LLM Tool Calling 双模式；实现 Agent 中间件（输入校验、调用限流、PII 脱敏、请求统计），对话与限流计数支持持久化（文件落盘）。
- **对话与多轮上下文**：实现按 session_id 的对话上下文管理，将历史消息拼入 RAG 提示；支持 CLI 与 HTTP 的 session_id 传递，便于多轮连贯问答。
- **API 与工程化**：基于 FastAPI 暴露 `/embed`、`/search`、`/chat_answer`、`/search_papers` 等接口，配置 CORS、统一异常处理与 lifespan（启动时加载向量库）；提供完整 CLI 子命令（embed / search / chat / papers / embed_paper / chat_paper 等）及运行说明文档（RUN.md），便于演示与对接前端。
- **前端演示**：编写单页 Web UI（`web/index.html`），通过 fetch 调用上述 API，实现文档入库、检索、问答、论文检索等操作的图形化演示。

---

## 技术栈（简历可简写）

- **后端**：Python 3.10+、FastAPI、LangChain 1.x（langchain-core / langchain-openai / langchain-chroma）
- **向量与检索**：Chroma、Qwen Embeddings、BM25（rank_bm25）、sentence-transformers（CrossEncoder 重排）
- **模型**：通义千问（Qwen）兼容 OpenAI 协议，用于对话与嵌入
- **其他**：pypdf / python-docx / openpyxl / BeautifulSoup（多格式文档解析）、arxiv（论文检索）、requests

---

## 可选：精简版（3～4 条，适合篇幅紧张时）

- 设计并实现基于 Chroma + Qwen 的 RAG 后端，完成文档切块、嵌入、多策略检索（多 query、BM25+语义 RRF、CrossEncoder 重排）及带引用的问答流程。
- 使用 LangChain Tools 与自研中间件实现知识库检索、天气、arXiv 论文等能力，支持规则路由与 LLM Tool Calling，并对输入校验、限流、PII 脱敏做持久化与统计。
- 基于 FastAPI 提供 REST API 与 CLI，实现多轮对话上下文管理、提示词模板与加强版 Prompt 设计，并配套简易 Web 演示页与运行文档。

---

使用时可将「工作内容」 bullets 按岗位 JD 适当突出「检索 / RAG」「提示词」「工程化」等关键词。
