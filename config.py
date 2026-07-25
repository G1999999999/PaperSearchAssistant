"""
全局配置（RAG 相关常量）

在这个 Demo 中，所有和 RAG 策略相关的参数都集中放在这里，
方便在面试时展示“如何通过调参来优化检索效果”。
"""

from __future__ import annotations

import os

# 基础参数
DEFAULT_NAMESPACE = "default"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4

# 跨 namespace：在非公共分区检索时，合并「公共知识库」检索结果（同一 Chroma persist 下另一 collection）
# 例：会话 namespace=conv_123 时，同时检索 RAG_PUBLIC_NAMESPACE（默认 default）并融合排序。
_pub_ns = (os.getenv("RAG_PUBLIC_NAMESPACE", DEFAULT_NAMESPACE) or DEFAULT_NAMESPACE).strip()
RAG_PUBLIC_NAMESPACE = _pub_ns or DEFAULT_NAMESPACE
_mpub = os.getenv("RAG_MERGE_PUBLIC_RETRIEVAL", "1").strip().lower()
RAG_MERGE_PUBLIC_RETRIEVAL = _mpub not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# arXiv 全文入库时是否同时写入 RAG_PUBLIC_NAMESPACE（便于会话 namespace=conv_xxx 时仍在公共库可检索）
_ingpub = os.getenv("RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC", "1").strip().lower()
RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC = _ingpub not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# arXiv 入库时是否将「标题+作者+摘要」单独写成 1 条向量文档（与 PDF 全文 chunk 同 namespace，便于 hybrid/BM25 命中摘要）
_abs = os.getenv("RAG_INGEST_ARXIV_ABSTRACT_VECTOR", "1").strip().lower()
RAG_INGEST_ARXIV_ABSTRACT_VECTOR = _abs not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# tool_download_arxiv_pdf 是否默认「下载后全文向量入库」（论文助手建议 1；只要 PDF+SQLite 设 0）
_dle = os.getenv("RAG_ARXIV_DOWNLOAD_DEFAULT_EMBED", "1").strip().lower()
RAG_ARXIV_DOWNLOAD_DEFAULT_EMBED = _dle not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# 仅给 ID 时用标题在 arXiv 上检索候选、让用户选序号入库时，最多展示几条
try:
    ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS = max(
        3, min(15, int(os.getenv("ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS", "8")))
    )
except ValueError:
    ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS = 8

# 入库前「按标题选论文」：向 arXiv 实际拉取条数（再截断为上一项展示条数）；优先 ti:"短语" 检索
_default_ingest_fetch = max(ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS * 3, 24)
try:
    ARXIV_INGEST_DISAMBIGUATION_FETCH_MAX = max(
        ARXIV_INGEST_DISAMBIGUATION_MAX_RESULTS,
        min(
            50,
            int(
                os.getenv(
                    "ARXIV_INGEST_DISAMBIGUATION_FETCH_MAX",
                    str(_default_ingest_fetch),
                )
            ),
        ),
    )
except ValueError:
    ARXIV_INGEST_DISAMBIGUATION_FETCH_MAX = _default_ingest_fetch

# arXiv PDF 直链下载（requests）：超时与 UA（部分网络/网关对默认 requests UA 或短超时极不友好）
try:
    ARXIV_PDF_DOWNLOAD_TIMEOUT = max(
        15.0, min(600.0, float(os.getenv("ARXIV_PDF_DOWNLOAD_TIMEOUT", "120")))
    )
except ValueError:
    ARXIV_PDF_DOWNLOAD_TIMEOUT = 120.0
ARXIV_PDF_USER_AGENT = (
    os.getenv(
        "ARXIV_PDF_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    or ""
).strip()

# 默认检索策略：稠密向量 + BM25（RRF 融合）+ CrossEncoder 重排（需 RAG_ENABLE_RERANK=1）
DEFAULT_RETRIEVAL_STRATEGY = "hybrid_rerank"

# 是否在 build_search_queries 中先调用 LLM 做 Query 改写（会多一次对话模型请求）
# 默认开启；关闭请设置环境变量：RAG_LLM_QUERY_REWRITE=0 / false / no / off
_rw = os.getenv("RAG_LLM_QUERY_REWRITE", "1").strip().lower()
RAG_LLM_QUERY_REWRITE = _rw not in ("0", "false", "no", "off", "disable", "disabled")

# 复合提问拆成多个子问题再分别做检索扩展（多一次 LLM；关闭：RAG_SUBQUESTION_SPLIT=0）
_sq = os.getenv("RAG_SUBQUESTION_SPLIT", "1").strip().lower()
RAG_SUBQUESTION_SPLIT = _sq not in ("0", "false", "no", "off", "disable", "disabled")
# 是否用 LangGraph 编排 chat_answer 主流程（路由 -> RAG/Tools）
_lg = os.getenv("RAG_USE_LANGGRAPH", "1").strip().lower()
RAG_USE_LANGGRAPH = _lg in ("1", "true", "yes", "on")
# 流程调试日志（RAG + Agent）：打印关键调用链并可落盘 jsonl。
_trace = os.getenv("RAG_TRACE_ENABLED", "0").strip().lower()
RAG_TRACE_ENABLED = _trace in ("1", "true", "yes", "on")
RAG_TRACE_LOG_FILE = (
    os.getenv("RAG_TRACE_LOG_FILE", "data/logs/rag_agent_trace.jsonl")
    or "data/logs/rag_agent_trace.jsonl"
).strip()
try:
    RAG_MAX_SUBQUESTIONS = max(1, min(20, int(os.getenv("RAG_MAX_SUBQUESTIONS", "8"))))
except ValueError:
    RAG_MAX_SUBQUESTIONS = 8

# 检索得分相关
# 对于 FAISS + 余弦距离，score 越小越相似，这里用一个经验阈值做过滤。
DEFAULT_SCORE_THRESHOLD = 0.5

# 多路检索 / MMR 相关
MMR_LAMBDA = 0.5
MMR_FETCH_K = 20

# 重排序：初检数量，重排后保留数量
RERANK_FETCH_K = 20
RERANK_TOP_K = 6

# Reranker 开关与模型配置
# - 代码内默认模型（改这里即可固定使用哪一款 CrossEncoder）
RERANK_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
# - RAG_ENABLE_RERANK=0 可完全关闭重排（hybrid_rerank / rerank 会退化为不重排）
# - RAG_RERANK_MODEL 可覆盖上面常量（便于临时 A/B 而不改代码）
_re = os.getenv("RAG_ENABLE_RERANK", "1").strip().lower()
RAG_ENABLE_RERANK = _re not in ("0", "false", "no", "off", "disable", "disabled")
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", RERANK_CROSS_ENCODER_MODEL).strip()
try:
    _rrbs = int(os.getenv("RAG_RERANK_BATCH_SIZE", "64"))
except ValueError:
    _rrbs = 64
RAG_RERANK_BATCH_SIZE = max(1, min(512, _rrbs))

# 混合检索（关键词 BM25 + 语义）：RRF 融合常数
RRF_K = 60
HYBRID_SEMANTIC_TOP_K = 15
HYBRID_BM25_TOP_K = 15

# 对话上下文：默认保留最近几轮
CONVERSATION_MAX_TURNS = 5

# 会话论文上下文快照（_paper_context.json）里 topic_summary 是否调用 LLM 润色（每轮一次短请求）
# 关闭：SESSION_PAPER_CONTEXT_LLM_SUMMARY=0
_spctx = os.getenv("SESSION_PAPER_CONTEXT_LLM_SUMMARY", "1").strip().lower()
SESSION_PAPER_CONTEXT_LLM_SUMMARY = _spctx not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# 拼进 LLM 的「检索上下文」长度上限（字符数近似控 token）。
# 说明：QWEN_MAX_Tokens 只限制**生成**长度；若 prompt（系统+历史+下面这段上下文）过长，
# 会挤占模型上下文窗口，表现为回答在「步骤 2」等处被截断或质量骤降。
try:
    RAG_CONTEXT_MAX_CHARS_PER_DOC = max(
        400, min(200_000, int(os.getenv("RAG_CONTEXT_MAX_CHARS_PER_DOC", "6000")))
    )
except ValueError:
    RAG_CONTEXT_MAX_CHARS_PER_DOC = 6000
try:
    _tmc = int(os.getenv("RAG_CONTEXT_TOTAL_MAX_CHARS", "72000"))
except ValueError:
    _tmc = 72000
# 0 表示不做「总字数」上限（仍受单条上限约束）
RAG_CONTEXT_TOTAL_MAX_CHARS = max(0, min(500_000, _tmc))

# 含「从 PG 合并的整段方法章节」上下文时，单条/总预算（避免 prompts 按默认 6000 字截断方法正文）
try:
    RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC = max(
        4000,
        min(500_000, int(os.getenv("RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC", "200000"))),
    )
except ValueError:
    RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC = 200_000
try:
    _tm_method = int(os.getenv("RAG_CONTEXT_PAPER_METHOD_TOTAL_MAX_CHARS", "260000"))
except ValueError:
    _tm_method = 260_000
RAG_CONTEXT_PAPER_METHOD_TOTAL_MAX_CHARS = max(0, min(500_000, _tm_method))

# paper_method 问答：是否把 PG 中 method 章节全部 chunk 合并后插入检索上下文（需 DATABASE_URL + 章节绑定）
_pm_inj = os.getenv("RAG_PAPER_METHOD_INJECT_PG_FULL_SECTION", "1").strip().lower()
RAG_PAPER_METHOD_INJECT_PG_FULL_SECTION = _pm_inj not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)
try:
    RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS = max(
        4000,
        min(500_000, int(os.getenv("RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS", "200000"))),
    )
except ValueError:
    RAG_PAPER_METHOD_PG_MERGE_MAX_CHARS = 200_000

# 为 True 时 RAG 用户消息用简短指令，不要求「步骤1–4」长结构，减少输出被 max_tokens 截断的概率
_rcp = os.getenv("RAG_USE_COMPACT_RAG_USER_PROMPT", "0").strip().lower()
RAG_USE_COMPACT_RAG_USER_PROMPT = _rcp in ("1", "true", "yes", "on")

# -------- 图片相关性判断（用户上传图片 vs 数据库检索上下文）--------
# 目标：判断用户图片是否和“本地检索到的数据库上下文”直接相关；
# - 相关：把用户图片合并进最终多模态 messages 让模型结合上下文+图回答
# - 不相关：不把用户图片注入最终 messages（避免模型被无关图干扰）
_img_judge = os.getenv("RAG_IMAGE_RELEVANCE_JUDGE", "1").strip().lower()
RAG_IMAGE_RELEVANCE_JUDGE_ENABLED = _img_judge in ("1", "true", "yes", "on")

# 给“图片相关性判断”提供的上下文字符预算（仅用于裁决图片是否用得上）
try:
    RAG_IMAGE_RELEVANCE_JUDGE_CONTEXT_MAX_CHARS = max(
        400, min(20000, int(os.getenv("RAG_IMAGE_RELEVANCE_JUDGE_CONTEXT_MAX_CHARS", "6000")))
    )
except ValueError:
    RAG_IMAGE_RELEVANCE_JUDGE_CONTEXT_MAX_CHARS = 6000

# 参与判断的用户图片数量上限（仍建议少于最终回答上限，降低成本）
try:
    RAG_IMAGE_RELEVANCE_JUDGE_MAX_IMAGES = max(
        1, min(16, int(os.getenv("RAG_IMAGE_RELEVANCE_JUDGE_MAX_IMAGES", "4")))
    )
except ValueError:
    RAG_IMAGE_RELEVANCE_JUDGE_MAX_IMAGES = 4

# 本地向量库（Chroma）无命中时，是否用联网摘要兜底（需安装 duckduckgo-search）
_wfb = os.getenv("RAG_WEB_FALLBACK", "1").strip().lower()
RAG_WEB_FALLBACK_ENABLED = _wfb not in ("0", "false", "no", "off", "disable", "disabled")

# use_tools / answer_with_tools：是否在首轮工具循环前先跑「子问题拆分 + 联网摘要」（与 RAG 兜底同源，优先 MCP）
_raw_tpf = os.getenv("RAG_TOOLS_PREFETCH_SUBQ_WEB", "0").strip().lower()
RAG_TOOLS_PREFETCH_SUBQ_WEB = _raw_tpf in ("1", "true", "yes", "on")
try:
    RAG_WEB_MAX_RESULTS = max(1, min(10, int(os.getenv("RAG_WEB_MAX_RESULTS", "5"))))
except ValueError:
    RAG_WEB_MAX_RESULTS = 5

# 按子问题分别联网检索后合并时，最多保留多少条去重后的摘要（更大 → 上下文更全，但更耗流量/限流风险）
try:
    RAG_WEB_MERGED_MAX_RESULTS = max(
        5, min(25, int(os.getenv("RAG_WEB_MERGED_MAX_RESULTS", "12")))
    )
except ValueError:
    RAG_WEB_MERGED_MAX_RESULTS = 12

# 子问题之间 sleep，缓解 DDG 限流（秒，0 表示不等待）
try:
    RAG_WEB_SUBQUERY_DELAY_SEC = max(
        0.0, float(os.getenv("RAG_WEB_SUBQUERY_DELAY_SEC", "0.45"))
    )
except ValueError:
    RAG_WEB_SUBQUERY_DELAY_SEC = 0.45

# 联网子查询并行度（1=保持串行+子查询间 sleep；>1 时并行 search_web_snippets，一般不 sleep）
try:
    RAG_WEB_SUBQUERY_MAX_CONCURRENT = max(
        1, min(8, int(os.getenv("RAG_WEB_SUBQUERY_MAX_CONCURRENT", "1")))
    )
except ValueError:
    RAG_WEB_SUBQUERY_MAX_CONCURRENT = 1

# 多 query 向量检索时线程池最大 worker 数（每个 query 一次 embedding + Chroma）
try:
    RAG_SIMILARITY_QUERY_MAX_WORKERS = max(
        1, min(16, int(os.getenv("RAG_SIMILARITY_QUERY_MAX_WORKERS", "4")))
    )
except ValueError:
    RAG_SIMILARITY_QUERY_MAX_WORKERS = 4

# 联网是否按子问题拆分搜索：空=与 RAG_SUBQUESTION_SPLIT 一致；0=整句只搜一次；1=强制按子问题搜
_wss_web = os.getenv("RAG_WEB_SUBQUESTION_SPLIT", "").strip().lower()
if _wss_web in ("0", "false", "no", "off", "disable"):
    RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE = False
elif _wss_web in ("1", "true", "yes", "on"):
    RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE = True
else:
    RAG_WEB_SUBQUESTION_SPLIT_OVERRIDE = None  # 跟随 RAG_SUBQUESTION_SPLIT

# 本地已有命中时：LLM 评判检索是否充分，不足则联网补充（需同时开启 RAG_WEB_FALLBACK）。
# - 1 / on：每轮（在 RAG 路径下）都跑评判
# - 0 / off：关闭
# - auto（默认）：由路由决定本轮是否跑评判（见 tools/agent/rag_judge_route.py：默认规则；
#   可再设 RAG_AUTO_JUDGE_USE_LLM=1 用一次轻量 LLM 裁决）
_rj = os.getenv("RAG_LLM_CONTEXT_SCORE", "auto").strip().lower()
if _rj in ("0", "false", "no", "off", "disable", "disabled"):
    RAG_LLM_CONTEXT_SCORE_MODE = "off"
elif _rj in ("1", "true", "yes", "on"):
    RAG_LLM_CONTEXT_SCORE_MODE = "on"
elif _rj in ("auto", "router", "路由"):
    RAG_LLM_CONTEXT_SCORE_MODE = "auto"
else:
    RAG_LLM_CONTEXT_SCORE_MODE = "auto"

RAG_LLM_CONTEXT_SCORE_ENABLED = RAG_LLM_CONTEXT_SCORE_MODE != "off"

_ajllm = os.getenv("RAG_AUTO_JUDGE_USE_LLM", "0").strip().lower()
RAG_AUTO_JUDGE_USE_LLM = _ajllm in ("1", "true", "yes", "on")
try:
    RAG_CONTEXT_SCORE_MIN = float(os.getenv("RAG_CONTEXT_SCORE_MIN", "6"))
except ValueError:
    RAG_CONTEXT_SCORE_MIN = 6.0
RAG_CONTEXT_SCORE_MIN = max(0.0, min(10.0, RAG_CONTEXT_SCORE_MIN))

# DuckDuckGo：默认 api 易触发 202 Ratelimit；依次尝试 lite/html（建议 pip install lxml）
# 例：export RAG_WEB_DDG_BACKENDS=lite,html,api
_raw_ddg_be = os.getenv("RAG_WEB_DDG_BACKENDS", "lite,html,api").strip()
RAG_WEB_DDG_BACKENDS = [
    b.strip().lower()
    for b in _raw_ddg_be.split(",")
    if b.strip().lower() in ("api", "html", "lite")
] or ["lite", "html", "api"]

# 可选：自建/可信 SearXNG 实例（JSON API），避开 DDG 限流
RAG_WEB_SEARXNG_URL = os.getenv("RAG_WEB_SEARXNG_URL", "").strip().rstrip("/")

# 可选：Brave Search API Key（https://brave.com/search/api/），有免费额度
RAG_WEB_BRAVE_API_KEY = os.getenv("RAG_WEB_BRAVE_API_KEY", "").strip()

# RAG 联网摘要「统一入口」渠道顺序（search_web_snippets）：
# - auto：若 MCP 已注册 `search` server，则**优先**走 MCP brave_web_search，失败再 DDG → SearXNG → REST Brave
# - ddg_first：保持旧行为（先 DDG，再 SearXNG / REST Brave），不在 RAG 路径里调 MCP
# - ddg_only：仅 DuckDuckGo（仍受 RAG_WEB_DDG_BACKENDS 影响）
_raw_rwb = os.getenv("RAG_WEB_BACKEND", "auto").strip().lower()
if _raw_rwb in ("ddg_first", "legacy"):
    RAG_WEB_BACKEND = "ddg_first"
elif _raw_rwb in ("ddg_only", "duckduckgo_only"):
    RAG_WEB_BACKEND = "ddg_only"
else:
    RAG_WEB_BACKEND = "auto"  # auto / mcp_first / optimal / 未知值

# ---------- RAG 联网：Streamable HTTP MCP（mcpmarket / Bright Data 等）----------
# 详见 tools/agent/mcp_runtime.py（streamable_http）
# RAG 联网：是否在 auto 模式下优先走 Streamable HTTP，再 Brave stdio
_raw_sf = os.getenv("RAG_WEB_STREAMABLE_FIRST", "1").strip().lower()
RAG_WEB_STREAMABLE_FIRST = _raw_sf not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)

# Streamable MCP 搜索工具名（Bright Data / mcpmarket 常见为 search_engine）
MCP_STREAMABLE_TOOL_NAME = (
    os.getenv("MCP_STREAMABLE_TOOL_NAME", "search_engine") or "search_engine"
).strip()

# 与 mcp_runtime 注册名一致（MCP_ENABLED 中需包含同名 token，默认 mcpmarket）
MCP_STREAMABLE_SERVER_NAME = (
    os.getenv("MCP_STREAMABLE_SERVER_NAME", "mcpmarket") or "mcpmarket"
).strip()

# ---------- 模型网络模式（统一 Chat + Embeddings）----------
# 只允许两种模式：
# - online：Chat + Embeddings 都走远程 OpenAI 兼容服务（默认）
# - offline：Chat 走本地 OpenAI 兼容服务，Embeddings 走本地 sentence-transformers
# 允许用 Python 文件直接覆盖（不依赖 shell source）：
# - 修改 `PaperSearchAssistant2/runtime_mode.py` 中的 `RAG_NETWORK_MODE_OVERRIDE`
# - 为空时仍按环境变量 `RAG_NETWORK_MODE` 生效
try:
    from runtime_mode import RAG_NETWORK_MODE_OVERRIDE as _RNMO
except Exception:
    _RNMO = ""

_rnm = (_RNMO or os.getenv("RAG_NETWORK_MODE", "online") or "online").strip().lower()
if _rnm in ("offline", "local", "离线", "本地"):
    RAG_NETWORK_MODE = "offline"
else:
    RAG_NETWORK_MODE = "online"

# offline 模式使用的本地 embedding 模型（sentence-transformers）
LOCAL_EMBED_MODEL = (
    os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3"
).strip()
LOCAL_EMBED_DEVICE = (os.getenv("LOCAL_EMBED_DEVICE", "cpu") or "cpu").strip()
# Reranker 设备与加载策略：
# - RAG_RERANK_DEVICE=auto：跟随 LOCAL_EMBED_DEVICE
# - 显式可设 cpu / cuda / cuda:0 / cuda:1 ...
_rrd_raw = (os.getenv("RAG_RERANK_DEVICE", "auto") or "auto").strip().lower()
if _rrd_raw in ("", "auto", "same", "embed"):
    RAG_RERANK_DEVICE = LOCAL_EMBED_DEVICE
else:
    RAG_RERANK_DEVICE = _rrd_raw
_rre = os.getenv("RAG_RERANK_EAGER_LOAD", "0").strip().lower()
RAG_RERANK_EAGER_LOAD = _rre in ("1", "true", "yes", "on")
_rrw = os.getenv("RAG_RERANK_WARMUP", "1").strip().lower()
RAG_RERANK_WARMUP = _rrw in ("1", "true", "yes", "on")

# ---------- 多模态（VL 对话 + 检索阶段附带图片）----------
# 对话模型名：支持视觉的 Qwen-VL 系列（DashScope OpenAI 兼容）。纯文本也可用同一接口。
QWEN_CHAT_MODEL = (os.getenv("QWEN_CHAT_MODEL", "qwen3.5-plus") or "qwen-vl-plus").strip()
# 单条最终 user 消息中最多附加多少张图（检索命中 + 用户上传合计后截断）
try:
    RAG_MAX_IMAGES_PER_MESSAGE = max(1, min(16, int(os.getenv("RAG_MAX_IMAGES_PER_MESSAGE", "6"))))
except ValueError:
    RAG_MAX_IMAGES_PER_MESSAGE = 6
try:
    RAG_USER_UPLOAD_MAX_IMAGES = max(1, min(12, int(os.getenv("RAG_USER_UPLOAD_MAX_IMAGES", "4"))))
except ValueError:
    RAG_USER_UPLOAD_MAX_IMAGES = 4
# 历史 JSONL 中带图 user 轮次：仅最近 N 轮在 VL 中重放像素，更早的只保留文字
try:
    RAG_CHAT_HISTORY_MAX_IMAGE_TURNS = max(0, min(5, int(os.getenv("RAG_CHAT_HISTORY_MAX_IMAGE_TURNS", "2"))))
except ValueError:
    RAG_CHAT_HISTORY_MAX_IMAGE_TURNS = 2
# PDF 入库时是否抽取位图并调用 VL 生成 caption 写入向量库（多图时 API 成本较高）
_fig = os.getenv("RAG_PDF_FIGURE_CAPTION_ENABLED", "0").strip().lower()
RAG_PDF_FIGURE_CAPTION_ENABLED = _fig in ("1", "true", "yes", "on")
try:
    PDF_EXTRACT_IMAGE_MAX_PER_PAPER = max(1, min(80, int(os.getenv("PDF_EXTRACT_IMAGE_MAX_PER_PAPER", "24"))))
except ValueError:
    PDF_EXTRACT_IMAGE_MAX_PER_PAPER = 24
# PDF 入库时是否用 pdfplumber 抽表格并写入 Chroma + paper_tables（需安装 pdfplumber）
_tbl = os.getenv("RAG_PDF_TABLE_EXTRACT_ENABLED", "1").strip().lower()
RAG_PDF_TABLE_EXTRACT_ENABLED = _tbl not in ("0", "false", "no", "off", "disable", "disabled")
try:
    PDF_EXTRACT_TABLE_MAX_PER_PAPER = max(1, min(200, int(os.getenv("PDF_EXTRACT_TABLE_MAX_PER_PAPER", "80"))))
except ValueError:
    PDF_EXTRACT_TABLE_MAX_PER_PAPER = 80
try:
    RAG_FIGURE_CAPTION_MAX_TOKENS = max(32, min(512, int(os.getenv("RAG_FIGURE_CAPTION_MAX_TOKENS", "256"))))
except ValueError:
    RAG_FIGURE_CAPTION_MAX_TOKENS = 256

# 展示「今天」等用的时区（IANA），空则使用系统本地时间。例：Asia/Shanghai
RAG_DISPLAY_TIMEZONE = os.getenv("RAG_DISPLAY_TIMEZONE", "").strip()

# 联网兜底后是否抓取首条结果 URL 的正文：auto=仅「相对时间+赛程类」问题；always=每次；off=关闭
_raw_fetch = os.getenv("RAG_WEB_FETCH_FIRST_PAGE", "auto").strip().lower()
if _raw_fetch in ("0", "false", "no", "off", "disable"):
    RAG_WEB_FETCH_FIRST_PAGE_MODE = "off"
elif _raw_fetch in ("1", "true", "yes", "on", "always", "all"):
    RAG_WEB_FETCH_FIRST_PAGE_MODE = "always"
else:
    RAG_WEB_FETCH_FIRST_PAGE_MODE = "auto"

# 不同“策略档位”的示例配置，你可以在面试时解释这些权衡（取舍）：
STRATEGY_PRESETS = {
    "aggressive_recall": {
        "top_k": 8,
        "score_threshold": 0.7,
        "mmr": False,
    },
    "high_precision": {
        "top_k": 4,
        "score_threshold": 0.3,
        "mmr": True,
    },
}

# 「你是谁 / 什么模型」等元问题：不依赖 RAG 检索的默认答复（可用环境变量 ASSISTANT_IDENTITY_REPLY 整段覆盖）
_default_identity = (
    "我是 **PaperSearchAssistant（论文检索助手）** 里的对话助手，用来帮你做本地论文库、向量知识库检索与问答。\n"
    "底层通过 `models_qwen.py` 接 **通义千问兼容 OpenAI 协议** 的接口（具体模型名以该文件为准，例如 qwen-plus 系列）。\n"
    "说明：文档/论文类事实我会尽量依据检索到的上下文；像「你是谁、什么模型」这类关于我自身的问题，**不需要**从知识库里找证据，可直接按上面回答。"
)
ASSISTANT_IDENTITY_REPLY = (os.getenv("ASSISTANT_IDENTITY_REPLY") or "").strip() or _default_identity

# 元问题（你是谁等）：默认走大模型直答、不检索；设 0/false 则仅用上面固定文案（省 token、更稳定）
_meta_llm = os.getenv("ASSISTANT_META_USE_LLM", "1").strip().lower()
ASSISTANT_META_USE_LLM = _meta_llm not in ("0", "false", "no", "off", "disable", "disabled")

# ---------- PostgreSQL（可选：DATABASE_REDESIGN_PLAN.md）----------
# DATABASE_URL 未配置时，不因「默认 RAG_PG_SYNC_ON_INGEST=1」而产生误导：同步默认关闭。
# 显式设置：1/on=强制开（仍要求 DATABASE_URL 非空才真正写 PG），0/off=强制关。
DATABASE_URL = (os.getenv("DATABASE_URL", "") or "").strip()
DATABASE_ECHO_SQL = (
    os.getenv("DATABASE_ECHO_SQL", "0").strip().lower() in ("1", "true", "yes", "on")
)
_pgsync_raw = (os.getenv("RAG_PG_SYNC_ON_INGEST", "") or "").strip().lower()
if _pgsync_raw in ("1", "true", "yes", "on"):
    RAG_PG_SYNC_ON_INGEST = True
elif _pgsync_raw in ("0", "false", "no", "off", "disable", "disabled"):
    RAG_PG_SYNC_ON_INGEST = False
else:
    RAG_PG_SYNC_ON_INGEST = bool(DATABASE_URL)

# ---------- 分层论文检索（PAPER_RETRIEVAL_OPTIMIZATION_PLAN.md）----------
def _ienv(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


RAG_PAPER_VECTOR_TOP_K = _ienv("RAG_PAPER_VECTOR_TOP_K", 30, 5, 80)
RAG_PAPER_BM25_TOP_K = _ienv("RAG_PAPER_BM25_TOP_K", 30, 5, 80)
RAG_PAPER_RERANK_TOP_K = _ienv("RAG_PAPER_RERANK_TOP_K", 8, 3, 20)
RAG_PAPER_RERANK_FETCH_K = _ienv("RAG_PAPER_RERANK_FETCH_K", 40, 10, 120)
_layered = os.getenv("RAG_LAYERED_PAPER_RETRIEVAL", "1").strip().lower()
RAG_LAYERED_PAPER_RETRIEVAL = _layered in ("1", "true", "yes", "on")
try:
    RAG_PAPER_METHOD_SECTION_RRF_WEIGHT = float(
        os.getenv("RAG_PAPER_METHOD_SECTION_RRF_WEIGHT", "1.45")
    )
except ValueError:
    RAG_PAPER_METHOD_SECTION_RRF_WEIGHT = 1.45
RAG_PAPER_METHOD_SECTION_RRF_WEIGHT = max(
    1.0, min(3.0, RAG_PAPER_METHOD_SECTION_RRF_WEIGHT)
)
# 「方法类」问题：对形似实验结果表 / 表头的 chunk 降权，避免 Table 3 因含 method 等词排到最前
_mtd = os.getenv("RAG_PAPER_METHOD_DOWNRANK_TABLE_CHUNKS", "1").strip().lower()
RAG_PAPER_METHOD_DOWNRANK_TABLE_CHUNKS = _mtd not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)
try:
    RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY = float(
        os.getenv("RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY", "0.35")
    )
except ValueError:
    RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY = 0.35
RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY = max(
    0.05, min(1.0, RAG_PAPER_METHOD_TABLE_CHUNK_PENALTY)
)
_ggw = os.getenv("RAG_PAPER_GLOSSARY_WEB_ENABLED", "1").strip().lower()
RAG_PAPER_GLOSSARY_WEB_ENABLED = _ggw not in (
    "0",
    "false",
    "no",
    "off",
    "disable",
    "disabled",
)
_mqg = os.getenv("RAG_PAPER_ENABLE_MULTI_QUERY", "1").strip().lower()
RAG_PAPER_ENABLE_MULTI_QUERY = _mqg in ("1", "true", "yes", "on")
_qd = os.getenv("RAG_PAPER_ENABLE_QUERY_DECOMPOSITION", "0").strip().lower()
RAG_PAPER_ENABLE_QUERY_DECOMPOSITION = _qd in ("1", "true", "yes", "on")
_tbl = os.getenv("RAG_PAPER_ENABLE_TABLE_RECALL", "1").strip().lower()
RAG_PAPER_ENABLE_TABLE_RECALL = _tbl in ("1", "true", "yes", "on")
_fig = os.getenv("RAG_PAPER_ENABLE_FIGURE_RECALL", "1").strip().lower()
RAG_PAPER_ENABLE_FIGURE_RECALL = _fig in ("1", "true", "yes", "on")
_pch = os.getenv("RAG_PAPER_ENABLE_PARENT_CHILD_RETRIEVAL", "1").strip().lower()
RAG_PAPER_ENABLE_PARENT_CHILD_RETRIEVAL = _pch in ("1", "true", "yes", "on")
_cexp = os.getenv("RAG_PAPER_ENABLE_CONTEXT_EXPANSION", "1").strip().lower()
RAG_PAPER_ENABLE_CONTEXT_EXPANSION = _cexp in ("1", "true", "yes", "on")
RAG_PAPER_CONTEXT_EXPANSION_DEPTH = _ienv("RAG_PAPER_CONTEXT_EXPANSION_DEPTH", 1, 0, 2)
_mmr = os.getenv("RAG_PAPER_ENABLE_MMR_PACKING", "0").strip().lower()
RAG_PAPER_ENABLE_MMR_PACKING = _mmr in ("1", "true", "yes", "on")
RAG_PAPER_SEARCH_PROFILE = (
    os.getenv("RAG_PAPER_SEARCH_PROFILE", "balanced") or "balanced"
).strip().lower()
_rcache = os.getenv("RAG_PAPER_RETRIEVAL_CACHE_TTL", "900").strip()
try:
    RAG_PAPER_RETRIEVAL_CACHE_TTL = max(0, min(86400, int(_rcache or "0")))
except ValueError:
    RAG_PAPER_RETRIEVAL_CACHE_TTL = 900

# 高级路由（ADVANCED_ROUTING_AND_HYBRID_RAG_PLAN_CN.md）
RAG_ROUTER_MODE = (os.getenv("RAG_ROUTER_MODE", "hybrid_router") or "hybrid_router").strip()
_rem = os.getenv("RAG_ROUTER_ENABLE_ENTITY_MATCH", "1").strip().lower()
RAG_ROUTER_ENABLE_ENTITY_MATCH = _rem in ("1", "true", "yes", "on")
# 无 PostgreSQL 时，SQL 论文候选匹配无意义；未设置 env 则随 DATABASE_URL 自动关/开。
_rsm_raw = (os.getenv("RAG_ROUTER_ENABLE_SQL_CANDIDATE_MATCH", "") or "").strip().lower()
if _rsm_raw in ("1", "true", "yes", "on"):
    RAG_ROUTER_ENABLE_SQL_CANDIDATE_MATCH = True
elif _rsm_raw in ("0", "false", "no", "off", "disable", "disabled"):
    RAG_ROUTER_ENABLE_SQL_CANDIDATE_MATCH = False
else:
    RAG_ROUTER_ENABLE_SQL_CANDIDATE_MATCH = bool(DATABASE_URL)
_rla = os.getenv("RAG_ROUTER_ENABLE_LLM_ASSIST", "0").strip().lower()
RAG_ROUTER_ENABLE_LLM_ASSIST = _rla in ("1", "true", "yes", "on")
try:
    RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD = float(
        os.getenv("RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD", "0.68")
    )
except ValueError:
    RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD = 0.68
RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD = max(
    0.0, min(1.0, RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD)
)
_pfl = os.getenv("RAG_PAPER_FORCE_LOCAL_FIRST", "1").strip().lower()
RAG_PAPER_FORCE_LOCAL_FIRST = _pfl in ("1", "true", "yes", "on")
# 章节预筛依赖 PG 中 paper_sections / chunk.section_id；无库时默认关闭以免空跑。
_spf_raw = (os.getenv("RAG_PAPER_ENABLE_SECTION_PREFILTER", "") or "").strip().lower()
if _spf_raw in ("1", "true", "yes", "on"):
    RAG_PAPER_ENABLE_SECTION_PREFILTER = True
elif _spf_raw in ("0", "false", "no", "off", "disable", "disabled"):
    RAG_PAPER_ENABLE_SECTION_PREFILTER = False
else:
    RAG_PAPER_ENABLE_SECTION_PREFILTER = bool(DATABASE_URL)

# 入库阶段：是否按章节进行更细粒度的重切分/重嵌入（用于修复 section 抽取不准导致的 method/result 命中偏差）
# 取值示例：off / section_aware
RAG_INGEST_SECTION_MODE = (os.getenv("RAG_INGEST_SECTION_MODE", "off") or "off").strip().lower()
_pre = os.getenv("RAG_PAPER_ENABLE_RERANK", "1").strip().lower()
RAG_PAPER_ENABLE_RERANK = _pre in ("1", "true", "yes", "on")

# 论文内容多通道问答（PAPER_CONTENT_QA_MULTIMODAL_PLAN_CN.md）
_ptq = os.getenv("RAG_PAPER_ENABLE_TABLE_QA", "1").strip().lower()
RAG_PAPER_ENABLE_TABLE_QA = _ptq in ("1", "true", "yes", "on")
_pfq = os.getenv("RAG_PAPER_ENABLE_FIGURE_QA", "1").strip().lower()
RAG_PAPER_ENABLE_FIGURE_QA = _pfq in ("1", "true", "yes", "on")
RAG_PAPER_TEXT_TOP_K = _ienv("RAG_PAPER_TEXT_TOP_K", 24, 4, 100)
RAG_PAPER_TABLE_TOP_K = _ienv("RAG_PAPER_TABLE_TOP_K", 12, 2, 60)
RAG_PAPER_FIGURE_TOP_K = _ienv("RAG_PAPER_FIGURE_TOP_K", 12, 2, 60)
RAG_PAPER_MULTIMODAL_RERANK_TOP_K = _ienv("RAG_PAPER_MULTIMODAL_RERANK_TOP_K", 8, 3, 20)
_mmp = os.getenv("RAG_PAPER_ENABLE_MULTIMODAL_PACKING", "1").strip().lower()
RAG_PAPER_ENABLE_MULTIMODAL_PACKING = _mmp in ("1", "true", "yes", "on")


