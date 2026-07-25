import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from datetime import timezone
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from runtime_settings import RUNTIME

# 引用 RUNTIME 即可触发统一环境加载（.env.runtime -> .env）
_RUNTIME = RUNTIME

# 代理的默认配置（仍以环境变量为准）。
_PROXY = "http://172.31.71.156:7890"
os.environ.setdefault("HTTP_PROXY", _PROXY)
os.environ.setdefault("HTTPS_PROXY", _PROXY)
os.environ.setdefault("http_proxy", os.environ.get("HTTP_PROXY", _PROXY))
os.environ.setdefault("https_proxy", os.environ.get("HTTPS_PROXY", _PROXY))
# 启动命令示例：uvicorn main:app --reload --host 0.0.0.0 --port 9000
from agent import RAGAgent
from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    RAG_USE_LANGGRAPH,
    RAG_USER_UPLOAD_MAX_IMAGES,
)
from tools.agent.arxiv_search import Paper, get_paper_by_id, search_arxiv
from tools.rag.knowledge import NamespaceVectorStore, vector_store
from models_qwen import qwen
from tools.rag.language import expand_retrieval_queries
from tools.rag.retrieval_merge import retrieve_with_public_merge
from tools.storage.paper_library import (
    LocalPaper,
    get_paper,
    list_papers as list_local_papers,
    reconcile_index_with_disk,
    upsert_paper,
)
from tools.rag.time_utils import filter_by_time
from tools.agent.router import extract_arxiv_id
from tools.agent.conversation import conversation_manager
from tools.agent.approvals import decide as approval_decide, get_approval, list_pending
from tools.agent.agent_tools import tool_delete_file, tool_read_file


# ---------- 配置与全局常量 ----------

# 会话知识上传：单文件上限（避免 OOM）
SESSION_EMBED_MAX_BYTES = 20 * 1024 * 1024


def _save_chat_images_base64(images: list[str] | None) -> list[str]:
    """解码 data URL 或裸 base64，写入 data/uploads/chat_images/，返回相对项目根路径。"""
    if not images:
        return []
    out_dir = Path("data/uploads/chat_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    saved: list[str] = []
    for raw in images[:RAG_USER_UPLOAD_MAX_IMAGES]:
        s = (raw or "").strip()
        if not s:
            continue
        mime = "image/png"
        b64data = s
        if s.startswith("data:"):
            m = re.match(r"data:([^;]+);base64,(.+)", s, re.DOTALL)
            if m:
                mime = (m.group(1) or "image/png").strip()
                b64data = (m.group(2) or "").strip()
        try:
            data = base64.standard_b64decode(b64data)
        except Exception:
            continue
        if not data:
            continue
        ext = ".png"
        low = mime.lower()
        if "jpeg" in low or "jpg" in low:
            ext = ".jpg"
        elif "webp" in low:
            ext = ".webp"
        elif "gif" in low:
            ext = ".gif"
        fn = out_dir / f"{uuid.uuid4().hex}{ext}"
        try:
            fn.write_bytes(data)
        except OSError:
            continue
        try:
            saved.append(str(fn.resolve().relative_to(cwd)))
        except ValueError:
            saved.append(str(fn.resolve()))
    return saved


def _ext_from_upload(content_type: str | None, filename: str | None) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    fn = (filename or "").lower()
    for ext, out in (
        (".jpeg", ".jpg"),
        (".jpg", ".jpg"),
        (".png", ".png"),
        (".webp", ".webp"),
        (".gif", ".gif"),
    ):
        if fn.endswith(ext):
            return out
    return ".png"


async def _save_chat_upload_files(files: list[UploadFile]) -> list[str]:
    """将 multipart 上传的图片写入 data/uploads/chat_images/，返回相对项目根路径。"""
    if not files:
        return []
    out_dir = Path("data/uploads/chat_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    saved: list[str] = []
    for uf in files[:RAG_USER_UPLOAD_MAX_IMAGES]:
        data = await uf.read()
        if not data:
            continue
        ext = _ext_from_upload(uf.content_type, uf.filename)
        fn = out_dir / f"{uuid.uuid4().hex}{ext}"
        try:
            fn.write_bytes(data)
        except OSError:
            continue
        try:
            saved.append(str(fn.resolve().relative_to(cwd)))
        except ValueError:
            saved.append(str(fn.resolve()))
    return saved


def _merge_user_image_paths(
    explicit: list[str] | None,
    from_base64: list[str],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in list(explicit or []) + list(from_base64):
        p = (raw or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= RAG_USER_UPLOAD_MAX_IMAGES:
            break
    return out


def _safe_session_upload_basename(name: str | None) -> str:
    """拒绝路径穿越/路径型文件名，仅保留安全 basename。"""
    raw = (name or "").strip()
    if not raw or raw in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid or empty filename")
    if "/" in raw or "\\" in raw or raw.startswith(".."):
        raise HTTPException(status_code=400, detail="Path-like filenames are not allowed")
    base = Path(raw).name
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base


class EmbedRequest(BaseModel):
    text: str
    namespace: str = DEFAULT_NAMESPACE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class SearchRequest(BaseModel):
    query: str
    namespace: str = DEFAULT_NAMESPACE
    k: int = DEFAULT_TOP_K
    strategy: str = "hybrid_rerank"  # 默认 | 多 query | 混合 | 混合重排 | 重排
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    since: datetime | None = None
    until: datetime | None = None


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ChatAnswerRequest(BaseModel):
    question: str
    namespace: str = DEFAULT_NAMESPACE
    strategy: str = "hybrid_rerank"  # 默认 | 多 query | 混合 | 混合重排 | 重排
    k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    use_multi_source: bool = False
    session_id: str | None = None
    use_tools: bool = False  # True：启用 LLM Tool Calling（天气/论文/知识库）
    use_business_middleware: bool = False  # 校验/限流/PII 脱敏/统计
    call_limit: int = 20  # use_business_middleware 模式下，每个会话最大调用次数
    stats_file: str | None = "data/logs/agent_stats.txt"  # 统计落盘路径，None 不落盘
    summarization: bool = False  # LangChain SummarizationMiddleware（如果可用）
    summarization_kwargs: dict | None = None  # SummarizationMiddleware 的可选参数
    auto_paper_qa_on_arxiv_id: bool = True  # 检测到 arXiv ID 时自动走 /papers/qa 逻辑
    images_base64: list[str] | None = None  # 可选：data:image/...;base64,... 或裸 base64
    user_image_paths: list[str] | None = None  # 已落盘相对/绝对路径（与 images_base64 合并，上限见配置）
    # use_tools 时：None=看 RAG_TOOLS_PREFETCH_SUBQ_WEB；True/False=强制开/关「子问题+联网(MCP优先)预取」
    prefetch_subq_web: bool | None = None


class ChatAnswerResponse(BaseModel):
    answer: str
    citations: list[dict]
    stats: dict | None = None  # 启用业务中间件后返回 _stats
    approval_required: dict | None = None
    web_fallback: bool | None = None  # True：本地 Chroma 无命中后已用联网摘要兜底
    web_supplement: bool | None = None  # True：本地有命中但 LLM 判不足，已补充联网摘要
    web_forced_by_user: bool | None = None  # True：用户明确要求联网后在本地基础上合并了网络摘要
    retrieval_context_score: float | None = None  # 检索充分性 0-10，仅评判开启时可能有值
    retrieval_judge_reason: str | None = None


class SessionEmbedFileResponse(BaseModel):
    namespace: str
    filename: str
    chunks_added: int
    ingest_id: str = ""
    arxiv_id: str | None = None
    paper_library_ingested: bool = False
    paper_ingest_note: str | None = None


class PaperInfo(BaseModel):
    title: str
    authors: list[str]
    summary: str
    url: str
    published: datetime


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    category: str | None = None
    sort_by: str = "relevance"


class PaperSearchResponse(BaseModel):
    results: list[PaperInfo]


class SmartPaperSearchRequest(BaseModel):
    """用户给出自然语言意图；LLM 将其转换为 arXiv 字段化检索式。"""

    intent: str
    max_results: int = 10
    sort_by: str = "lastUpdatedDate"  # 默认按最新排序
    category: str | None = None  # 可选的硬过滤条件，例如 cs.CV


class SmartPaperSearchResponse(BaseModel):
    query: str
    category: str | None = None
    sort_by: str
    since: str | None = None
    until: str | None = None
    results: list[PaperInfo]


class PaperDownloadRequest(BaseModel):
    arxiv_id: str
    pdf_dir: str | None = "data/papers"


class LocalPaperInfo(BaseModel):
    arxiv_id: str
    title: str | None = None
    authors: list[str] = []
    summary: str | None = None
    published: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    view_url: str | None = None
    namespaces: list[str] = []
    added_at: str | None = None


class PaperListResponse(BaseModel):
    results: list[LocalPaperInfo]


class PaperListQuery(BaseModel):
    title: str | None = None
    author: str | None = None
    keyword: str | None = None
    year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    limit: int = 50
    offset: int = 0


class PaperSearchLocalRequest(BaseModel):
    """本地论文检索：SQLite 过滤 + 可选的语义重排（通过 Chroma）。"""

    title: str | None = None
    author: str | None = None
    keyword: str | None = None
    year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    limit: int = 20
    offset: int = 0
    semantic_query: str | None = None
    semantic_topk_per_paper: int = 1
    semantic_candidate_multiplier: int = 5


class PaperSearchLocalResponse(BaseModel):
    results: list[LocalPaperInfo]
    total: int | None = None  # 可选：后续可用 COUNT 查询补充


class PaperSearchLocalRagRequest(BaseModel):
    """分层论文检索（查询理解 + 多路召回 + RRF + 证据重排）；见 PAPER_RETRIEVAL_OPTIMIZATION_PLAN.md。"""

    query: str
    namespace: str = DEFAULT_NAMESPACE
    k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    strategy: str = "hybrid_rerank"
    use_layered: bool = True


class PaperSearchLocalRagChunk(BaseModel):
    preview: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperSearchLocalRagResponse(BaseModel):
    chunks: list[PaperSearchLocalRagChunk]


class PaperEmbedRequest(BaseModel):
    arxiv_id: str
    namespace: str = DEFAULT_NAMESPACE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class PaperEmbedAllRequest(BaseModel):
    pdf_dir: str = "data/papers"
    namespace_prefix: str = "paper"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    recursive: bool = False
    incremental: bool = True
    force_rebuild: bool = False
    max_workers: int = 1  # >1 时并行 PDF 分块；向量写入仍加锁串行


class ConversationEmbedRequest(BaseModel):
    session_id: str | None = None  # None 表示全量
    persist_dir: str = "data/conversations"
    chunk_size: int = 800
    chunk_overlap: int = 80
    incremental: bool = True
    force_rebuild: bool = False


class ConversationSearchRequest(BaseModel):
    query: str
    session_id: str | None = None  # None 表示全部
    k: int = 6
    score_threshold: float = 0.8
    strategy: str = "hybrid_rerank"


class PaperQARequest(BaseModel):
    question: str
    arxiv_id: str | None = None  # 可选：可从问题中解析得到
    namespace: str | None = None  # 默认值：paper:<id>:full
    k: int = DEFAULT_TOP_K
    strategy: str = "hybrid_rerank"
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    auto_download: bool = True
    auto_embed: bool = True
    force_embed: bool = False
    session_id: str | None = None
    images_base64: list[str] | None = None
    user_image_paths: list[str] | None = None  # 已落盘相对路径（优先于 images_base64）


class PaperQAResponse(BaseModel):
    paper: LocalPaperInfo
    answer: str
    citations: list[dict]

class AgentRunRequest(BaseModel):
    goal: str
    namespace: str = DEFAULT_NAMESPACE
    session_id: str | None = None
    k: int = DEFAULT_TOP_K
    max_steps: int = 6
    write_memory: bool = True
    memory_k: int = 6
    use_business_middleware: bool = False
    call_limit: int = 30
    stats_file: str | None = "data/logs/agent_stats.txt"


class AgentRunResponse(BaseModel):
    answer: str
    citations: list[dict]
    plan: dict
    trace: list[dict]
    memory: dict
    approval_required: dict | None = None


class ApprovalDecideRequest(BaseModel):
    approval_id: str
    decision: str  # 批准 | 编辑 | 拒绝
    edited_args: dict | None = None
    note: str | None = None


class ApprovalDecideResponse(BaseModel):
    status: str
    approval: dict
    tool_result: str | None = None

# ---------- FastAPI 应用 ----------

EMBED_MANIFEST_PATH = Path("data/papers/embed_manifest.json")
AUDIT_LOG_PATH = Path("data/logs/audit.log")
_embed_all_chroma_lock = threading.Lock()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_embed_manifest() -> dict:
    if not EMBED_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(EMBED_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_embed_manifest(manifest: dict) -> None:
    EMBED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBED_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_audit_log(action: str, status: str, detail: str) -> None:
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {action} {status} {detail}\n")
    except Exception:
        pass


def _is_delete_intent_text(text: str) -> bool:
    q = (text or "").lower()
    keys = ["删除", "删掉", "移除", "清除", "delete", "remove"]
    return any(k in q for k in keys)


def _normalize_arxiv_id(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    s = s.replace("arXiv:", "").replace("ARXIV:", "").strip()
    # 支持传入完整 URL
    if "arxiv.org/abs/" in s:
        s = s.split("arxiv.org/abs/", 1)[-1]
    if "arxiv.org/pdf/" in s:
        s = s.split("arxiv.org/pdf/", 1)[-1]
    s = s.replace(".pdf", "").strip("/")
    s = __import__("re").sub(r"v\d+$", "", s, flags=__import__("re").IGNORECASE)
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载已持久化的向量库；关闭时可选落盘（当前未实现）。"""
    try:
        vector_store.load()
    except Exception:
        pass
    yield
    # 可选：如需可手动保存向量库（当前流程按需加载）


app = FastAPI(
    title="论文检索助手",
    description="论文检索助手：本地论文库（SQLite）+ 全文向量检索（Chroma）+ arXiv 检索与问答。",
    version="0.1.0",
    lifespan=lifespan,
)

# 提供简易 Web 页面
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# 允许网页跨域调用（前端在不同端口/域名时必需）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """统一捕获未处理异常，避免 500 直接暴露堆栈。"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sessions")
async def list_sessions() -> dict:
    """列出会话：合并本地 jsonl 与（若启用）PostgreSQL chat_sessions。"""
    from pathlib import Path

    from tools.storage.repos.chat_repo import list_session_ids as sql_session_ids

    ids: set[str] = set()
    d = Path("data/conversations")
    if d.exists():
        ids.update(p.stem for p in d.glob("*.jsonl"))
    try:
        ids.update(sql_session_ids())
    except Exception:
        pass
    return {"sessions": sorted(ids, reverse=True)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """获取某个会话的消息。"""
    msgs = conversation_manager.get_recent_messages(session_id)
    return {"session_id": session_id, "messages": msgs}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str) -> dict:
    """与 GET /sessions/{session_id} 相同，便于与 DATABASE_REDESIGN_PLAN 文档路径对齐。"""
    msgs = conversation_manager.get_recent_messages(session_id)
    return {"session_id": session_id, "messages": msgs}


@app.get("/sessions/{session_id}/attachments")
async def get_session_attachments(session_id: str) -> dict:
    """列出会话附件（需 PostgreSQL 与 chat_attachments 数据）。"""
    from tools.storage.repos.chat_repo import list_attachments_for_session

    try:
        items = list_attachments_for_session(session_id)
    except Exception:
        items = []
    return {"session_id": session_id, "attachments": items}


@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str) -> dict:
    """清空内存中的会话（文件仍保留）。"""
    conversation_manager.clear(session_id)
    return {"ok": True, "session_id": session_id}


@app.get("/", response_class=HTMLResponse)
async def root():
    """内置 Web UI 的首页。"""
    from pathlib import Path

    index_path = Path("web/index.html")
    if not index_path.exists():
        return HTMLResponse("<h3>UI not found</h3><p>Missing web/index.html</p>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    """将文本嵌入到内存向量库（类似 /remember）。"""
    return await asyncio.to_thread(_embed_sync, req)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """在指定 namespace 上进行向量检索，并支持策略与时间过滤；空 namespace 返回空列表。"""
    return await asyncio.to_thread(_search_sync, req)


async def _execute_chat_answer(req: ChatAnswerRequest) -> dict:
    """执行与 ``/chat_answer`` 相同的问答逻辑，返回可 JSON 序列化的字段字典（供流式接口复用）。"""
    return await asyncio.to_thread(_execute_chat_answer_sync, req)


@app.post("/chat_answer", response_model=ChatAnswerResponse)
async def chat_answer(req: ChatAnswerRequest) -> ChatAnswerResponse:
    """完整 RAG 问答：use_tools=True 时走 Tool Calling；use_business_middleware=True 时启用业务中间件。"""
    data = await _execute_chat_answer(req)
    return ChatAnswerResponse(
        answer=data["answer"],
        citations=data["citations"],
        stats=data.get("stats"),
        approval_required=data.get("approval_required"),
        web_fallback=data.get("web_fallback"),
        web_supplement=data.get("web_supplement"),
        web_forced_by_user=data.get("web_forced_by_user"),
        retrieval_context_score=data.get("retrieval_context_score"),
        retrieval_judge_reason=data.get("retrieval_judge_reason"),
    )


@app.post("/chat_answer/stream")
async def chat_answer_stream(req: ChatAnswerRequest) -> StreamingResponse:
    """SSE：先完成检索与生成，再按字符流式推送正文（meta 事件含 citations 等）。"""

    async def event_gen():
        data = await _execute_chat_answer(req)
        answer_text = data.get("answer") or ""
        meta_payload = {k: v for k, v in data.items() if k != "answer" and not str(k).startswith("_")}
        yield f"data: {json.dumps({'type': 'meta', 'payload': meta_payload}, ensure_ascii=False, default=str)}\n\n"
        # 按 Unicode 标量粗流式输出，减轻前端一次性渲染压力
        for i, ch in enumerate(answer_text):
            yield f"data: {json.dumps({'type': 'delta', 'text': ch}, ensure_ascii=False)}\n\n"
            if i % 24 == 0:
                await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat_answer/multipart", response_model=ChatAnswerResponse)
async def chat_answer_multipart(
    question: str = Form(...),
    namespace: str = Form(DEFAULT_NAMESPACE),
    strategy: str = Form("hybrid_rerank"),
    k: int = Form(DEFAULT_TOP_K),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    use_multi_source: bool = Form(False),
    session_id: str | None = Form(None),
    use_tools: bool = Form(False),
    prefetch_subq_web: bool = Form(False),
    use_business_middleware: bool = Form(False),
    call_limit: int = Form(20),
    stats_file: str | None = Form("data/logs/agent_stats.txt"),
    summarization: bool = Form(False),
    auto_paper_qa_on_arxiv_id: bool = Form(True),
    images: list[UploadFile] | None = File(None),
) -> ChatAnswerResponse:
    """与 ``/chat_answer`` 相同逻辑；``images`` 可传多文件（字段名均为 ``images``）。"""
    uploaded = await _save_chat_upload_files(list(images or []))
    sid = (session_id or "").strip() or None
    sf = (stats_file or "").strip() or None
    req = ChatAnswerRequest(
        question=question,
        namespace=namespace,
        strategy=strategy,
        k=k,
        score_threshold=score_threshold,
        use_multi_source=use_multi_source,
        session_id=sid,
        use_tools=use_tools,
        prefetch_subq_web=True if prefetch_subq_web else None,
        use_business_middleware=use_business_middleware,
        call_limit=call_limit,
        stats_file=sf,
        summarization=summarization,
        summarization_kwargs=None,
        auto_paper_qa_on_arxiv_id=auto_paper_qa_on_arxiv_id,
        images_base64=None,
        user_image_paths=uploaded or None,
    )
    data = await _execute_chat_answer(req)
    return ChatAnswerResponse(
        answer=data["answer"],
        citations=data["citations"],
        stats=data.get("stats"),
        approval_required=data.get("approval_required"),
        web_fallback=data.get("web_fallback"),
        web_supplement=data.get("web_supplement"),
        web_forced_by_user=data.get("web_forced_by_user"),
        retrieval_context_score=data.get("retrieval_context_score"),
        retrieval_judge_reason=data.get("retrieval_judge_reason"),
    )


@app.post("/session/embed_file", response_model=SessionEmbedFileResponse)
async def session_embed_file(
    file: UploadFile = File(...),
    namespace: str = Form(...),
    session_id: str | None = Form(None),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> SessionEmbedFileResponse:
    """将会话中的用户文件嵌入指定 RAG namespace（不写论文库）。"""
    ns = (namespace or "").strip()
    if not ns:
        raise HTTPException(status_code=400, detail="namespace is required")
    if chunk_size < 100 or chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="Invalid chunk_size or chunk_overlap")

    safe_name = _safe_session_upload_basename(file.filename)
    scratch_dir = Path("data/uploads/session_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_dir / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        total = 0
        with tmp_path.open("wb") as out:
            while True:
                buf = await file.read(1024 * 512)
                if not buf:
                    break
                total += len(buf)
                if total > SESSION_EMBED_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File larger than {SESSION_EMBED_MAX_BYTES // (1024 * 1024)}MB",
                    )
                out.write(buf)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        from tools.agent.session_file_embed import embed_session_file
        from tools.agent.conversation import conversation_manager

        ingest_id = uuid.uuid4().hex
        extra: dict = {
            "upload_via": "api_session_embed",
            "original_name": safe_name,
            "session_ingest_id": ingest_id,
        }
        if session_id and str(session_id).strip():
            extra["session_id"] = str(session_id).strip()

        result = await asyncio.to_thread(
            embed_session_file,
            tmp_path,
            ns,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_meta=extra,
        )
        if (
            session_id
            and str(session_id).strip()
            and int(result.chunks_added or 0) > 0
        ):
            conversation_manager.add_session_embed(
                str(session_id).strip(),
                ingest_id=ingest_id,
                namespace=ns,
                filename=safe_name,
                chunks_added=result.chunks_added,
            )
        return SessionEmbedFileResponse(
            namespace=ns,
            filename=safe_name,
            chunks_added=result.chunks_added,
            ingest_id=ingest_id,
            arxiv_id=result.arxiv_id,
            paper_library_ingested=result.paper_library_ingested,
            paper_ingest_note=result.paper_ingest_note,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/search_papers", response_model=PaperSearchResponse)
async def search_papers(req: PaperSearchRequest) -> PaperSearchResponse:
    """在 arXiv 上检索论文，并返回标准化后的列表。"""
    return await asyncio.to_thread(_search_papers_sync, req)


def _search_papers_smart_sync(req: SmartPaperSearchRequest) -> SmartPaperSearchResponse:
    import json
    import re
    from datetime import datetime, timezone

    from fastapi import HTTPException

    from models_qwen import qwen

    intent = (req.intent or "").strip()
    if not intent:
        raise HTTPException(status_code=400, detail="intent is required")

    system = (
        "你把用户意图转换成 arXiv 的字段化检索式。\n"
        "仅返回严格 JSON（不要 Markdown）。\n"
        '架构：\n'
        '{\n'
        '  "ti": [string],\n'
        '  "abs": [string],\n'
        '  "au": [string],\n'
        '  "keywords": [string],\n'
        '  "category": string|null,\n'
        '  "since": string|null,   // 格式：YYYY 或 YYYY-MM 或 YYYY-MM-DD 或 ISO8601\n'
        '  "until": string|null,   // 同上格式\n'
        '  "sort_by": "relevance"|"lastUpdatedDate"\n'
        '}\n'
        "指导原则：\n"
        "- 将标题相关的词放入 ti。\n"
        "- 将摘要/正文概念放入 abs。\n"
        "- 将作者姓名放入 au（例如 'Hinton', 'Yann LeCun'）。\n"
        "- 将通用术语放入 keywords。\n"
        "- 如果用户询问最新/近期，请优先选择 sort_by=lastUpdatedDate。\n"
        "- 如果用户提到时间范围，请设置 since/until。\n"
        "- 如果你能推断出明确的 arXiv 分类（例如 cs.CV、cs.CL），请设置 category。\n"
        "- 不要编造 ID 或 URL。\n"
    )
    user = (
        f"用户意图：{intent}\n"
        f"用户提供的分类（可选硬过滤）：{req.category}\n"
        f"用户期望的排序：{req.sort_by}\n"
        "现在返回 JSON。"
    )
    resp = qwen.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    raw = resp.content if hasattr(resp, "content") else str(resp)
    # 防御式提取第一个 JSON 对象
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise HTTPException(status_code=500, detail=f"LLM did not return JSON: {raw[:300]}")
    try:
        data = json.loads(m.group(0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON: {e}. raw={raw[:300]}")

    def _norm_list(v) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            s = str(x or "").strip()
            if s:
                out.append(s)
        return out

    def _esc(s: str) -> str:
        return s.replace('"', '\\"').strip()

    ti_terms = _norm_list(data.get("ti"))
    abs_terms = _norm_list(data.get("abs"))
    au_terms = _norm_list(data.get("au"))
    kw_terms = _norm_list(data.get("keywords"))

    def _uniq(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x.strip())
        return out

    ti_terms = _uniq(ti_terms)
    abs_terms = _uniq(abs_terms)
    au_terms = _uniq(au_terms)
    kw_terms = _uniq(kw_terms)

    # 构建“字段化但不过度严格”的检索式：
    # - 每个字段组内部使用“或”
    # - 不同字段组之间使用“且”
    def _or_group(items: list[str], field: str | None = None) -> str | None:
        if not items:
            return None
        # 最多保留 3 个，避免过度约束
        items = items[:3]
        terms = []
        for s in items:
            s2 = _esc(s)
            if field:
                terms.append(f'{field}:"{s2}"')
            else:
                terms.append(f'"{s2}"' if " " in s2 else s2)
        if len(terms) == 1:
            return terms[0]
        return "(" + " OR ".join(terms) + ")"

    groups: list[str] = []
    g_ti = _or_group(ti_terms, "ti")
    g_abs = _or_group(abs_terms, "abs")
    g_au = _or_group(au_terms, "au")
    g_kw = _or_group(kw_terms, None)
    for g in [g_ti, g_abs, g_au, g_kw]:
        if g:
            groups.append(g)

    query_strict = " AND ".join(groups).strip()
    if not query_strict:
        query_strict = intent

    # 放宽版检索：去掉字段约束，仅保留关键词/短标题词条。
    relaxed_terms = _uniq((kw_terms + ti_terms + abs_terms)[:6])
    query_relaxed = " OR ".join(
        [f'"{_esc(t)}"' if " " in t else _esc(t) for t in relaxed_terms]
    ).strip()
    if not query_relaxed:
        query_relaxed = intent

    sort_by = str(data.get("sort_by") or req.sort_by or "lastUpdatedDate").strip()
    if sort_by not in {"relevance", "lastUpdatedDate"}:
        sort_by = "lastUpdatedDate"

    category = data.get("category")
    if isinstance(category, str):
        category = category.strip() or None
    else:
        category = None

    # 请求中的硬过滤条件优先。
    if req.category and req.category.strip():
        category = req.category.strip()

    def _parse_dt(s: str | None) -> datetime | None:
        if not s:
            return None
        s = str(s).strip()
        if not s:
            return None
        # 年份 / 年-月份 / 年-月份-日期
        try:
            if re.fullmatch(r"\d{4}", s):
                return datetime(int(s), 1, 1, tzinfo=timezone.utc)
            if re.fullmatch(r"\d{4}-\d{2}", s):
                y, mth = s.split("-")
                return datetime(int(y), int(mth), 1, tzinfo=timezone.utc)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                y, mth, d = s.split("-")
                return datetime(int(y), int(mth), int(d), tzinfo=timezone.utc)
            # ISO8601 格式
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    since = _parse_dt(data.get("since"))
    until = _parse_dt(data.get("until"))
    since_str = since.isoformat() if since else None
    until_str = until.isoformat() if until else None

    # 按时间过滤时适当多取候选，避免漏检。
    fetch_n = int(req.max_results)
    if (since or until) and fetch_n < 50:
        fetch_n = min(50, max(fetch_n * 5, 20))

    def _run_search(q: str) -> list[Paper]:
        return search_arxiv(
            query=q,
            max_results=fetch_n,
            sort_by=sort_by,
            category=category,
        )

    query_used = query_strict
    papers: list[Paper] = _run_search(query_used)
    # 如果没有结果则自动放宽条件。
    if not papers and query_relaxed and query_relaxed != query_used:
        query_used = query_relaxed
        papers = _run_search(query_used)
    if since or until:
        filtered = []
        for p in papers:
            pub = p.published
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if since and pub < since:
                continue
            if until and pub > until:
                continue
            filtered.append(p)
        papers = filtered[: int(req.max_results)]
    else:
        papers = papers[: int(req.max_results)]
    infos = [
        PaperInfo(
            title=p.title,
            authors=p.authors,
            summary=p.summary,
            url=p.url,
            published=p.published,
        )
        for p in papers
    ]
    return SmartPaperSearchResponse(
        query=query_used,
        category=category,
        sort_by=sort_by,
        since=since_str,
        until=until_str,
        results=infos,
    )


@app.post("/search_papers_smart", response_model=SmartPaperSearchResponse)
async def search_papers_smart(req: SmartPaperSearchRequest) -> SmartPaperSearchResponse:
    """LLM 解析自然语言意图后检索 arXiv（字段化检索式 + 时间窗口）。"""
    return await asyncio.to_thread(_search_papers_smart_sync, req)


def _paper_view_url(filename: str) -> str:
    return f"/papers/view/{filename}"


def _embed_sync(req: EmbedRequest) -> dict:
    n_chunks = vector_store.add_text(
        text=req.text,
        namespace=req.namespace,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return {"namespace": req.namespace, "chunks_added": n_chunks}


def _search_sync(req: SearchRequest) -> SearchResponse:
    if not req.query or not req.query.strip():
        return SearchResponse(results=[])
    queries = expand_retrieval_queries(
        req.query,
        strategy=req.strategy,
        llm=qwen,
    )
    if not queries:
        fb = (req.query or "").strip()
        queries = [fb] if fb else []
    docs_and_scores = retrieve_with_public_merge(
        vector_store,
        queries=queries,
        namespace=req.namespace,
        k=req.k,
        score_threshold=req.score_threshold,
        strategy=req.strategy,
    )
    if req.since or req.until:
        docs_and_scores = filter_by_time(
            docs_and_scores,
            since=req.since,
            until=req.until,
        )
    results = [
        SearchResult(
            content=doc.page_content,
            metadata=doc.metadata,
            score=float(score),
        )
        for doc, score in docs_and_scores
    ]
    return SearchResponse(results=list(results))


def _execute_chat_answer_sync(req: ChatAnswerRequest) -> dict:
    from tools.agent.middleware import default_business_middleware

    user_image_paths = _merge_user_image_paths(
        req.user_image_paths,
        _save_chat_images_base64(req.images_base64),
    )

    if req.auto_paper_qa_on_arxiv_id:
        arxiv_id = extract_arxiv_id(req.question or "")
        if arxiv_id and not _is_delete_intent_text(req.question or ""):
            paper_namespace = None if req.namespace == DEFAULT_NAMESPACE else req.namespace
            paper_result = _papers_qa_sync(
                PaperQARequest(
                    question=req.question,
                    arxiv_id=arxiv_id,
                    namespace=paper_namespace,
                    k=req.k,
                    strategy=req.strategy,
                    score_threshold=req.score_threshold,
                    auto_download=True,
                    auto_embed=True,
                    force_embed=False,
                    session_id=req.session_id,
                    user_image_paths=user_image_paths or None,
                )
            )
            return {
                "answer": paper_result.answer,
                "citations": list(paper_result.citations or []),
                "stats": None,
                "approval_required": None,
                "web_fallback": None,
                "web_supplement": None,
                "web_forced_by_user": None,
                "retrieval_context_score": None,
                "retrieval_judge_reason": None,
                "_paper_qa_short_circuit": True,
            }

    middleware = []
    if req.use_business_middleware:
        middleware.extend(
            default_business_middleware(
                call_limit=req.call_limit,
                stats_file=req.stats_file,
                summarization=bool(req.summarization),
                summarization_kwargs=(req.summarization_kwargs or None),
            )
        )
    agent = RAGAgent(middleware=middleware)
    # 当本轮包含用户图片时，优先走主 RAG 流程（agent.answer），以便使用
    # “图片-数据库相关性裁决 -> 决定是否注入图片”的策略。
    use_tools = bool(req.use_tools) and not bool(user_image_paths)
    if RAG_USE_LANGGRAPH:
        from tools.agent.langgraph_orchestrator import execute_chat_with_langgraph

        result = execute_chat_with_langgraph(
            agent=agent,
            question=req.question,
            namespace=req.namespace,
            strategy=req.strategy,
            k=req.k,
            score_threshold=req.score_threshold,
            use_multi_source=req.use_multi_source,
            session_id=req.session_id,
            user_image_paths=user_image_paths or None,
            use_tools=use_tools,
            prefetch_subq_web=req.prefetch_subq_web,
        )
    else:
        if use_tools:
            result = agent.answer_with_tools(
                question=req.question,
                namespace=req.namespace,
                k=req.k,
                session_id=req.session_id,
                user_image_paths=user_image_paths or None,
                prefetch_subq_web=req.prefetch_subq_web,
            )
        else:
            result = agent.answer(
                question=req.question,
                namespace=req.namespace,
                strategy=req.strategy,
                k=req.k,
                score_threshold=req.score_threshold,
                use_multi_source=req.use_multi_source,
                session_id=req.session_id,
                user_image_paths=user_image_paths or None,
            )
    if not result.get("approval_required"):
        try:
            from tools.storage.chat_turn_embed import embed_chat_turn_into_rag_namespace

            q_embed = req.question or ""
            if user_image_paths:
                q_embed = f"{q_embed}\n[本轮用户上传 {len(user_image_paths)} 张图片]"
            embed_chat_turn_into_rag_namespace(
                namespace=req.namespace,
                question=q_embed,
                answer=str(result.get("answer") or ""),
                citations=list(result.get("citations") or []),
                source="api_chat_turn",
            )
        except Exception:
            pass
    return {
        "answer": result.get("answer") or "",
        "citations": list(result.get("citations") or []),
        "stats": result.get("_stats"),
        "approval_required": result.get("approval_required"),
        "web_fallback": result.get("web_fallback"),
        "web_supplement": result.get("web_supplement"),
        "web_forced_by_user": result.get("web_forced_by_user"),
        "retrieval_context_score": result.get("retrieval_context_score"),
        "retrieval_judge_reason": result.get("retrieval_judge_reason"),
        "_paper_qa_short_circuit": False,
    }


def _papers_qa_sync(req: PaperQARequest) -> PaperQAResponse:
    from pathlib import Path

    from fastapi import HTTPException

    from tools.agent.arxiv_search import download_pdf
    from tools.rag.document import load_pdf

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    arxiv_id = _normalize_arxiv_id((req.arxiv_id or "").strip() or (extract_arxiv_id(question) or ""))
    if not arxiv_id:
        raise HTTPException(status_code=400, detail="arxiv_id is required (or include it in question text)")

    reconcile_index_with_disk()
    existing = get_paper(arxiv_id) or {}
    pdf_path = Path(existing.get("pdf_path") or f"data/papers/{arxiv_id}.pdf")

    if (not pdf_path.exists()) and req.auto_download:
        pdf_path = download_pdf(arxiv_id, dest_dir="data/papers")
        paper_meta = get_paper_by_id(arxiv_id)
        upsert_paper(
            LocalPaper(
                arxiv_id=arxiv_id,
                pdf_path=str(pdf_path.as_posix()),
                title=paper_meta.title if paper_meta else None,
                authors=paper_meta.authors if paper_meta else [],
                published=(paper_meta.published.isoformat() if paper_meta else None),
                url=(paper_meta.url if paper_meta else f"https://arxiv.org/abs/{arxiv_id}"),
            )
        )
        existing = get_paper(arxiv_id) or existing

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Local PDF not found. Set auto_download=true or download first.")

    namespace = req.namespace or f"paper:{arxiv_id}:full"
    already = set(existing.get("namespaces") or [])

    if req.auto_embed and (req.force_embed or namespace not in already):
        text, meta = load_pdf(str(pdf_path), parent_id=arxiv_id)
        vector_store.embed_document(
            text=text,
            namespace=namespace,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            extra_metadata=meta,
        )
        from tools.rag.pdf_figures import embed_pdf_figures_to_namespace

        embed_pdf_figures_to_namespace(
            vector_store,
            pdf_path=str(pdf_path),
            parent_id=arxiv_id,
            namespace=namespace,
            arxiv_id=arxiv_id,
        )
        paper_meta = get_paper_by_id(arxiv_id)
        upsert_paper(
            LocalPaper(
                arxiv_id=arxiv_id,
                pdf_path=str(pdf_path.as_posix()),
                title=paper_meta.title if paper_meta else existing.get("title"),
                authors=paper_meta.authors if paper_meta else (existing.get("authors") or []),
                published=(paper_meta.published.isoformat() if paper_meta else existing.get("published")),
                url=(paper_meta.url if paper_meta else existing.get("url") or f"https://arxiv.org/abs/{arxiv_id}"),
                namespaces=[namespace],
            )
        )

    final_rec = get_paper(arxiv_id) or {}
    paper_info = LocalPaperInfo(
        arxiv_id=final_rec.get("arxiv_id", arxiv_id),
        title=final_rec.get("title"),
        authors=list(final_rec.get("authors") or []),
        summary=final_rec.get("summary"),
        published=final_rec.get("published"),
        url=final_rec.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
        pdf_path=final_rec.get("pdf_path") or str(pdf_path.as_posix()),
        view_url=_paper_view_url(Path(final_rec.get("pdf_path") or str(pdf_path.as_posix())).name),
        namespaces=list(final_rec.get("namespaces") or []),
        added_at=final_rec.get("added_at"),
    )

    user_paths: list[str] = list(req.user_image_paths or [])
    if not user_paths and req.images_base64:
        user_paths = _save_chat_images_base64(req.images_base64)

    agent = RAGAgent()
    result = agent.answer(
        question=question,
        namespace=namespace,
        strategy=req.strategy,
        k=req.k,
        score_threshold=req.score_threshold,
        session_id=req.session_id,
        user_image_paths=user_paths or None,
    )
    return PaperQAResponse(
        paper=paper_info,
        answer=result["answer"],
        citations=result["citations"],
    )


def _agent_run_sync(req: AgentRunRequest) -> AgentRunResponse:
    from tools.agent.middleware import default_business_middleware

    middleware = []
    if req.use_business_middleware:
        middleware.extend(
            default_business_middleware(
                call_limit=req.call_limit,
                stats_file=req.stats_file,
            )
        )
    agent = RAGAgent(middleware=middleware)
    result = agent.run_autonomous(
        goal=req.goal,
        namespace=req.namespace,
        session_id=req.session_id,
        k=req.k,
        max_steps=req.max_steps,
        write_memory=req.write_memory,
        memory_k=req.memory_k,
    )
    return AgentRunResponse(
        answer=result["answer"],
        citations=result["citations"],
        plan=result["plan"],
        trace=result["trace"],
        memory=result["memory"],
        approval_required=result.get("approval_required"),
    )


def _search_papers_sync(req: PaperSearchRequest) -> PaperSearchResponse:
    papers: list[Paper] = search_arxiv(
        query=req.query,
        max_results=req.max_results,
        sort_by=req.sort_by,
        category=req.category,
    )
    infos = [
        PaperInfo(
            title=p.title,
            authors=p.authors,
            summary=p.summary,
            url=p.url,
            published=p.published,
        )
        for p in papers
    ]
    return PaperSearchResponse(results=infos)


def _papers_search_local_sync(req: PaperSearchLocalRequest) -> PaperSearchLocalResponse:
    from pathlib import Path

    from tools.storage.papers_db import list_papers as db_list_papers

    reconcile_index_with_disk()

    limit = max(1, min(int(req.limit), 100))
    offset = max(0, int(req.offset))
    cand_limit = max(limit, limit * max(1, int(req.semantic_candidate_multiplier or 5)))

    candidates = db_list_papers(
        title=req.title,
        author=req.author,
        keyword=req.keyword,
        year=req.year,
        year_from=req.year_from,
        year_to=req.year_to,
        limit=cand_limit,
        offset=offset,
    )

    semantic_query = (req.semantic_query or "").strip()
    if semantic_query:
        scored: list[tuple[dict, float]] = []
        for rec in candidates[:cand_limit]:
            arxiv_id = rec.get("arxiv_id") or ""
            namespaces = list(rec.get("namespaces") or [])
            ns = namespaces[0] if namespaces else f"paper:{arxiv_id}:full"
            docs = vector_store.retrieve(
                queries=[semantic_query],
                namespace=ns,
                k=max(1, int(req.semantic_topk_per_paper or 1)),
                score_threshold=0.0,
                strategy="default",
            )
            best_score = float(docs[0][1]) if docs else 1e9
            scored.append((rec, best_score))
        scored.sort(key=lambda x: x[1])
        candidates = [r for r, _ in scored]

    results: list[LocalPaperInfo] = []
    for rec in candidates[:limit]:
        pdf_path = rec.get("pdf_path")
        view_url = _paper_view_url(Path(pdf_path).name) if pdf_path else None
        results.append(
            LocalPaperInfo(
                arxiv_id=rec.get("arxiv_id", ""),
                title=rec.get("title"),
                authors=list(rec.get("authors") or []),
                summary=rec.get("summary"),
                published=rec.get("published"),
                url=rec.get("url"),
                pdf_path=pdf_path,
                view_url=view_url,
                namespaces=list(rec.get("namespaces") or []),
                added_at=rec.get("added_at"),
            )
        )
    return PaperSearchLocalResponse(results=results, total=None)


def _papers_download_sync(req: PaperDownloadRequest) -> LocalPaperInfo:
    from pathlib import Path

    from tools.agent.arxiv_search import download_pdf

    arxiv_id = _normalize_arxiv_id(req.arxiv_id)
    if not arxiv_id:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="arxiv_id is required")
    reconcile_index_with_disk()
    existing = get_paper(arxiv_id) or {}
    pdf_path = Path(existing.get("pdf_path") or f"{(req.pdf_dir or 'data/papers').rstrip('/')}/{arxiv_id}.pdf")
    if not pdf_path.exists():
        pdf_path = download_pdf(arxiv_id, dest_dir=req.pdf_dir or "data/papers")
    paper = get_paper_by_id(arxiv_id)
    rec = upsert_paper(
        LocalPaper(
            arxiv_id=arxiv_id,
            pdf_path=str(pdf_path.as_posix()),
            title=paper.title if paper else None,
            authors=paper.authors if paper else [],
            published=(paper.published.isoformat() if paper else None),
            url=(paper.url if paper else f"https://arxiv.org/abs/{arxiv_id}"),
        )
    )
    return LocalPaperInfo(
        arxiv_id=rec.arxiv_id,
        title=rec.title,
        authors=rec.authors or [],
        published=rec.published,
        url=rec.url,
        pdf_path=rec.pdf_path,
        view_url=_paper_view_url(Path(rec.pdf_path).name),
        namespaces=rec.namespaces or [],
        added_at=rec.added_at,
    )


def _papers_embed_sync(req: PaperEmbedRequest) -> LocalPaperInfo:
    from pathlib import Path

    from fastapi import HTTPException

    from tools.rag.document import load_pdf_chunks_pymupdf

    reconcile_index_with_disk()
    existing = get_paper(req.arxiv_id) or {}
    pdf_path = Path(existing.get("pdf_path") or f"data/papers/{req.arxiv_id}.pdf")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Local PDF not found. Download first.")

    namespace = req.namespace
    if namespace == DEFAULT_NAMESPACE:
        namespace = f"paper:{req.arxiv_id}:full"

    chunks = load_pdf_chunks_pymupdf(
        str(pdf_path),
        parent_id=req.arxiv_id,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    vector_store.clear_namespace(namespace)
    vector_store.add_documents(
        documents=chunks,
        namespace=namespace,
    )
    from tools.rag.pdf_figures import embed_pdf_figures_to_namespace

    embed_pdf_figures_to_namespace(
        vector_store,
        pdf_path=str(pdf_path),
        parent_id=req.arxiv_id,
        namespace=namespace,
        arxiv_id=req.arxiv_id,
    )

    paper = get_paper_by_id(req.arxiv_id)
    rec = upsert_paper(
        LocalPaper(
            arxiv_id=req.arxiv_id,
            pdf_path=str(pdf_path.as_posix()),
            title=paper.title if paper else existing.get("title"),
            authors=paper.authors if paper else (existing.get("authors") or []),
            published=(paper.published.isoformat() if paper else existing.get("published")),
            url=(paper.url if paper else existing.get("url") or f"https://arxiv.org/abs/{req.arxiv_id}"),
            namespaces=[namespace],
        )
    )
    return LocalPaperInfo(
        arxiv_id=rec.arxiv_id,
        title=rec.title,
        authors=rec.authors or [],
        published=rec.published,
        url=rec.url,
        pdf_path=rec.pdf_path,
        view_url=_paper_view_url(Path(rec.pdf_path).name),
        namespaces=rec.namespaces or [],
        added_at=rec.added_at,
    )


def _memory_embed_conversations_sync(req: ConversationEmbedRequest) -> dict:
    from tools.storage.long_memory import embed_all_conversations, embed_conversation_history

    if req.session_id and req.session_id.strip():
        return embed_conversation_history(
            session_id=req.session_id.strip(),
            persist_dir=req.persist_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            incremental=req.incremental,
            force_rebuild=req.force_rebuild,
        )
    return embed_all_conversations(
        persist_dir=req.persist_dir,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        incremental=req.incremental,
        force_rebuild=req.force_rebuild,
    )


def _memory_search_conversations_sync(req: ConversationSearchRequest) -> dict:
    from tools.storage.long_memory import retrieve_conversation_memories

    items = retrieve_conversation_memories(
        query=req.query,
        session_id=req.session_id,
        k=req.k,
        score_threshold=req.score_threshold,
        strategy=req.strategy,
    )
    return {"results": items}


def _papers_upload_sync(
    pid: str,
    pdf_path_str: str,
    used_namespace: str,
    auto_embed: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> LocalPaperInfo:
    from pathlib import Path

    from tools.rag.document import load_pdf_chunks_pymupdf

    pdf_path = Path(pdf_path_str)
    if auto_embed:
        chunks = load_pdf_chunks_pymupdf(
            str(pdf_path),
            parent_id=pid,
            chunk_size=max(200, int(chunk_size)),
            chunk_overlap=max(0, int(chunk_overlap)),
        )
        vector_store.clear_namespace(used_namespace)
        vector_store.add_documents(chunks, namespace=used_namespace)

    rec = upsert_paper(
        LocalPaper(
            arxiv_id=pid,
            pdf_path=str(pdf_path.as_posix()),
            namespaces=[used_namespace] if auto_embed else [],
        )
    )
    return LocalPaperInfo(
        arxiv_id=rec.arxiv_id,
        title=rec.title,
        authors=rec.authors or [],
        summary=rec.summary,
        published=rec.published,
        url=rec.url,
        pdf_path=rec.pdf_path,
        view_url=_paper_view_url(Path(rec.pdf_path).name),
        namespaces=rec.namespaces or [],
        added_at=rec.added_at,
    )


def _papers_embed_all_local_sync(req: PaperEmbedAllRequest) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import HTTPException

    from tools.rag.document import load_pdf_chunks_pymupdf

    pdf_root = Path(req.pdf_dir or "data/papers")
    if not pdf_root.exists() or not pdf_root.is_dir():
        raise HTTPException(status_code=404, detail="pdf_dir not found")

    glob_pattern = "**/*.pdf" if req.recursive else "*.pdf"
    pdf_files = sorted(pdf_root.glob(glob_pattern))
    if not pdf_files:
        return {
            "pdf_dir": str(pdf_root.as_posix()),
            "embedded_files": 0,
            "embedded_chunks": 0,
            "failed_files": [],
        }

    manifest = _load_embed_manifest()
    mw = max(1, min(32, int(req.max_workers)))

    def _prepare_one(pdf_path: Path) -> dict:
        arxiv_id = pdf_path.stem
        namespace = f"{req.namespace_prefix}:{arxiv_id}:full"
        key = str(pdf_path.resolve())
        try:
            file_hash = _sha256_file(pdf_path)
            old = manifest.get(key) if isinstance(manifest.get(key), dict) else {}
            unchanged = (
                req.incremental
                and (not req.force_rebuild)
                and old.get("sha256") == file_hash
                and old.get("namespace") == namespace
                and int(old.get("chunk_size", -1)) == int(req.chunk_size)
                and int(old.get("chunk_overlap", -1)) == int(req.chunk_overlap)
            )
            if unchanged:
                return {"status": "skip", "key": key}
            chunks = load_pdf_chunks_pymupdf(
                str(pdf_path),
                parent_id=arxiv_id,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
            return {
                "status": "ok",
                "key": key,
                "arxiv_id": arxiv_id,
                "namespace": namespace,
                "file_hash": file_hash,
                "pdf_path": pdf_path,
                "chunks": chunks,
            }
        except Exception as e:
            return {
                "status": "err",
                "pdf_path": str(pdf_path.as_posix()),
                "error": str(e),
            }

    if mw <= 1:
        prepared = [_prepare_one(p) for p in pdf_files]
    else:
        workers = min(mw, len(pdf_files))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            prepared = list(ex.map(_prepare_one, pdf_files))

    embedded_files = 0
    embedded_chunks = 0
    skipped_files = 0
    failed_files: list[dict] = []

    for item in prepared:
        st = item.get("status")
        if st == "skip":
            skipped_files += 1
            continue
        if st == "err":
            failed_files.append(
                {"pdf_path": item["pdf_path"], "error": item.get("error", "")}
            )
            continue
        pdf_path = item["pdf_path"]
        try:
            with _embed_all_chroma_lock:
                vector_store.clear_namespace(item["namespace"])
                n = vector_store.add_documents(item["chunks"], namespace=item["namespace"])
                upsert_paper(
                    LocalPaper(
                        arxiv_id=item["arxiv_id"],
                        pdf_path=str(pdf_path.as_posix()),
                        namespaces=[item["namespace"]],
                    )
                )
            embedded_files += 1
            embedded_chunks += n
            manifest[item["key"]] = {
                "sha256": item["file_hash"],
                "namespace": item["namespace"],
                "chunk_size": int(req.chunk_size),
                "chunk_overlap": int(req.chunk_overlap),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            failed_files.append(
                {"pdf_path": str(pdf_path.as_posix()), "error": str(e)}
            )

    _save_embed_manifest(manifest)
    return {
        "pdf_dir": str(pdf_root.as_posix()),
        "incremental": bool(req.incremental),
        "force_rebuild": bool(req.force_rebuild),
        "total_pdf_files": len(pdf_files),
        "embedded_files": embedded_files,
        "skipped_files": skipped_files,
        "embedded_chunks": embedded_chunks,
        "failed_files": failed_files,
    }


def _safe_pdf_name(name: str) -> str:
    safe = Path(name or "").name
    if not safe.lower().endswith(".pdf"):
        safe = f"{safe}.pdf"
    return safe


def _papers_search_local_rag_sync(req: PaperSearchLocalRagRequest) -> PaperSearchLocalRagResponse:
    from tools.retrieval.paper_retriever import layered_paper_retrieve

    pairs = layered_paper_retrieve(
        vector_store,
        question=req.query.strip(),
        namespace=(req.namespace or DEFAULT_NAMESPACE).strip(),
        strategy=req.strategy,
        k=int(req.k),
        score_threshold=float(req.score_threshold),
        session_ingest_ids=None,
        llm=qwen,
        use_layered=bool(req.use_layered),
    )
    chunks: list[PaperSearchLocalRagChunk] = []
    for doc, sc in pairs or []:
        meta = getattr(doc, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        safe_meta: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe_meta[str(k)] = v
            elif isinstance(v, (list, dict)) and len(str(v)) < 500:
                safe_meta[str(k)] = v
        preview = (getattr(doc, "page_content", "") or "")[:4000]
        chunks.append(
            PaperSearchLocalRagChunk(preview=preview, score=float(sc), metadata=safe_meta)
        )
    return PaperSearchLocalRagResponse(chunks=chunks)


@app.get("/papers/list", response_model=PaperListResponse)
async def papers_list(
    title: str | None = None,
    author: str | None = None,
    keyword: str | None = None,
    year: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> PaperListResponse:
    """按条件列出本地论文（SQLite 后端）。"""
    from pathlib import Path

    reconcile_index_with_disk()
    results: list[LocalPaperInfo] = []
    from tools.storage.papers_db import list_papers as db_list_papers

    for rec in db_list_papers(
        title=title,
        author=author,
        keyword=keyword,
        year=year,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
        offset=offset,
    ):
        pdf_path = rec.get("pdf_path")
        view_url = _paper_view_url(Path(pdf_path).name) if pdf_path else None
        results.append(
            LocalPaperInfo(
                arxiv_id=rec.get("arxiv_id", ""),
                title=rec.get("title"),
                authors=list(rec.get("authors") or []),
                summary=rec.get("summary"),
                published=rec.get("published"),
                url=rec.get("url"),
                pdf_path=pdf_path,
                view_url=view_url,
                namespaces=list(rec.get("namespaces") or []),
                added_at=rec.get("added_at"),
            )
        )
    return PaperListResponse(results=results)


@app.get("/papers/storage/{paper_id}/sections")
async def papers_storage_sections(paper_id: int) -> dict:
    """PostgreSQL：论文章节树（无库或未迁移时返回空列表）。"""
    from tools.storage.repos.paper_repo import list_sections

    try:
        return {"paper_id": paper_id, "sections": list_sections(paper_id)}
    except Exception:
        return {"paper_id": paper_id, "sections": []}


@app.get("/papers/storage/{paper_id}/tables")
async def papers_storage_tables(paper_id: int) -> dict:
    from tools.storage.repos.paper_repo import list_tables

    try:
        return {"paper_id": paper_id, "tables": list_tables(paper_id)}
    except Exception:
        return {"paper_id": paper_id, "tables": []}


@app.get("/papers/storage/{paper_id}/figures")
async def papers_storage_figures(paper_id: int) -> dict:
    from tools.storage.repos.paper_repo import list_figures

    try:
        return {"paper_id": paper_id, "figures": list_figures(paper_id)}
    except Exception:
        return {"paper_id": paper_id, "figures": []}


@app.post("/papers/search_local", response_model=PaperSearchLocalRagResponse)
async def papers_search_local_rag(req: PaperSearchLocalRagRequest) -> PaperSearchLocalRagResponse:
    """向量分区上的分层检索（与原 /papers/search — SQLite 列表 — 互补）。"""
    return await asyncio.to_thread(_papers_search_local_rag_sync, req)


@app.get("/papers/{arxiv_id}", response_model=LocalPaperInfo)
async def papers_get(arxiv_id: str) -> LocalPaperInfo:
    """通过 arXiv ID 获取单条论文元数据记录（SQLite 后端）。"""
    from pathlib import Path

    from fastapi import HTTPException

    from tools.storage.paper_library import get_paper as get_paper_record

    reconcile_index_with_disk()
    rec = get_paper_record(arxiv_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Paper not found")
    pdf_path = rec.get("pdf_path")
    view_url = _paper_view_url(Path(pdf_path).name) if pdf_path else None
    return LocalPaperInfo(
        arxiv_id=rec.get("arxiv_id", arxiv_id),
        title=rec.get("title"),
        authors=list(rec.get("authors") or []),
        summary=rec.get("summary"),
        published=rec.get("published"),
        url=rec.get("url"),
        pdf_path=pdf_path,
        view_url=view_url,
        namespaces=list(rec.get("namespaces") or []),
        added_at=rec.get("added_at"),
    )


@app.post("/papers/search", response_model=PaperSearchLocalResponse)
async def papers_search(req: PaperSearchLocalRequest) -> PaperSearchLocalResponse:
    """通过 SQLite 检索本地论文，并可选用 Chroma 做语义相关性重排。"""
    return await asyncio.to_thread(_papers_search_local_sync, req)


@app.post("/papers/download", response_model=LocalPaperInfo)
async def papers_download(req: PaperDownloadRequest) -> LocalPaperInfo:
    """下载 arXiv 论文 PDF 到 data/papers，并更新/插入元数据索引。"""
    return await asyncio.to_thread(_papers_download_sync, req)


@app.post("/papers/upload", response_model=LocalPaperInfo)
async def papers_upload(
    file: UploadFile = File(...),
    arxiv_id: str | None = Form(None),
    namespace: str | None = Form(None),
    auto_embed: bool = Form(True),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> LocalPaperInfo:
    """上传本地 PDF，保存到 data/papers；可选地嵌入并建立索引。"""
    filename = _safe_pdf_name(file.filename or "uploaded.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pid = (arxiv_id or "").strip() or Path(filename).stem
    pdf_path = Path("data/papers") / f"{pid}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        pdf_path.write_bytes(content)
    finally:
        await file.close()

    used_namespace = (namespace or "").strip() or f"paper:{pid}:full"
    return await asyncio.to_thread(
        _papers_upload_sync,
        pid,
        str(pdf_path.as_posix()),
        used_namespace,
        bool(auto_embed),
        int(chunk_size),
        int(chunk_overlap),
    )


@app.get("/papers/view/{filename}")
async def papers_view(filename: str):
    """从本地提供数据目录中的 PDF（仅接受安全的文件名）。"""
    from pathlib import Path

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    name = Path(filename).name  # 防止目录穿越
    pdf_path = Path("data/papers") / name
    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=name)


@app.post("/papers/embed", response_model=LocalPaperInfo)
async def papers_embed(req: PaperEmbedRequest) -> LocalPaperInfo:
    """将已下载的本地论文 PDF 嵌入向量库，命名为 paper:<id>:full（推荐）。"""
    return await asyncio.to_thread(_papers_embed_sync, req)


@app.post("/papers/embed_all_local")
async def papers_embed_all_local(req: PaperEmbedAllRequest) -> dict:
    """批量把本地所有 PDF（用 PyMuPDFLoader 分块）嵌入到 Chroma。"""
    return await asyncio.to_thread(_papers_embed_all_local_sync, req)


@app.post("/memory/conversations/embed")
async def memory_embed_conversations(req: ConversationEmbedRequest) -> dict:
    """把持久化对话 JSONL 向量化入库（支持增量）。"""
    return await asyncio.to_thread(_memory_embed_conversations_sync, req)


@app.post("/memory/conversations/search")
async def memory_search_conversations(req: ConversationSearchRequest) -> dict:
    """在历史对话向量库中语义检索。"""
    return await asyncio.to_thread(_memory_search_conversations_sync, req)


@app.post("/papers/qa", response_model=PaperQAResponse)
async def papers_qa(req: PaperQARequest) -> PaperQAResponse:
    """一键论文问答：确保 PDF 已下载且已完成嵌入入库，然后回答。"""
    return await asyncio.to_thread(_papers_qa_sync, req)


@app.post("/agent_run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """自主执行流程：规划器 -> 步骤循环（工具调用）-> 长期记忆。"""
    return await asyncio.to_thread(_agent_run_sync, req)


@app.post("/approvals/decide", response_model=ApprovalDecideResponse)
async def approvals_decide(req: ApprovalDecideRequest) -> ApprovalDecideResponse:
    """审批/编辑/拒绝 HumanApprovalMiddleware 中断的待处理工具调用。

    说明：这里会直接执行工具并返回 tool_result。调用方（UI）随后可通过再发送一次 chat 请求继续对话。
    """
    item = get_approval(req.approval_id)
    if not item:
        raise HTTPException(status_code=404, detail="approval_id not found")

    try:
        it = approval_decide(
            approval_id=req.approval_id,
            decision_type=req.decision,
            edited_args=req.edited_args,
            note=req.note,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tool_result: str | None = None
    # 审计审批决策
    _append_audit_log(
        action="APPROVAL_DECIDE",
        status=it.status,
        detail=f"approval_id={it.approval_id} tool={it.tool_name} decision={it.decision}",
    )
    if it.status == "approved" and it.decision and it.decision.get("type") != "reject":
        # 仅执行受支持的工具
        try:
            if it.tool_name == "tool_read_file":
                tool_result = str(tool_read_file.invoke(it.tool_args))
            elif it.tool_name == "tool_delete_file":
                # 强校验：确保删除前文件存在，且删除后不存在。
                target = str((it.tool_args or {}).get("path") or "")
                p = Path(target).expanduser()
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                else:
                    p = p.resolve()
                existed_before = p.exists() and p.is_file()
                if not existed_before:
                    raise HTTPException(status_code=400, detail=f"delete target not found: {p}")
                tool_result = str(tool_delete_file.invoke(it.tool_args))
                exists_after = p.exists()
                if exists_after:
                    _append_audit_log(
                        action="DELETE_FILE",
                        status="FAILED",
                        detail=f"path={p} reason=exists_after_delete",
                    )
                    raise HTTPException(status_code=500, detail=f"delete verification failed: {p} still exists")
                _append_audit_log(
                    action="DELETE_FILE",
                    status="SUCCESS",
                    detail=f"path={p}",
                )
            else:
                raise HTTPException(status_code=400, detail=f"unsupported tool: {it.tool_name}")
            it.result = tool_result
        except HTTPException:
            raise
        except Exception as e:
            _append_audit_log(
                action="TOOL_EXECUTION",
                status="FAILED",
                detail=f"tool={it.tool_name} approval_id={it.approval_id} error={e}",
            )
            raise HTTPException(status_code=500, detail=f"tool execution error: {e}")

    return ApprovalDecideResponse(
        status=it.status,
        approval={
            "approval_id": it.approval_id,
            "session_id": it.session_id,
            "tool": it.tool_name,
            "args": it.tool_args,
            "allowed_decisions": it.allowed_decisions,
            "decision": it.decision,
        },
        tool_result=tool_result,
    )


@app.get("/approvals/pending")
async def approvals_pending(session_id: str | None = None) -> dict:
    """列出待审批项（可选按 session_id 过滤）。"""
    return {"items": list_pending(session_id=session_id)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

