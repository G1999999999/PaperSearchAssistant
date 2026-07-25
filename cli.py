"""
简单命令行入口，用于在终端里演示 RAG 功能。

示例：
    python cli.py embed --namespace work --file notes.txt
    python cli.py search --namespace work --query "问题" --k 4
    python cli.py chat --namespace work --question "问题"
    python cli.py chat --use-tools --prefetch-subq-web --question "RAG 与 Finetune 区别？如何选型？" \\
        --namespace default
    （检索评判：RAG_LLM_CONTEXT_SCORE=auto|on|off，默认 auto 由路由决定；轻量 LLM 路由：RAG_AUTO_JUDGE_USE_LLM=1）
    python cli.py session_finalize --session-id u1 --namespace conv_001  # 离开前刷新会话论文快照
    python cli.py preview_subquestions --question "RAG是什么？怎么评估检索？"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime_settings import RUNTIME

from agent import RAGAgent
from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_TOP_K,
    RAG_NETWORK_MODE,
)
from tools.agent.arxiv_search import download_pdf, get_arxiv_id, search_arxiv
from tools.rag.document import load_file, load_pdf
from tools.rag.knowledge import vector_store
from tools.rag.retrieval_merge import retrieve_with_public_merge
from tools.storage.chat_turn_embed import embed_chat_turn_into_rag_namespace
from tools.rag.language import expand_retrieval_queries
from tools.rag.time_utils import filter_by_time, now_utc
from prompts import build_rag_prompt
from models_qwen import qwen


def _print_runtime_mode_banner() -> None:
    if RAG_NETWORK_MODE == "offline":
        print(
            f"[mode] offline | chat=local_openai_compat | "
            f"embed=local:{RUNTIME.local_embed_model}"
        )
    else:
        print(
            f"[mode] online | chat=remote_openai_compat | "
            f"embed=local:{RUNTIME.local_embed_model}"
        )


def _cli_user_image_paths(paths: list[str] | None) -> list[str] | None:
    """将 CLI 传入的图片路径规范为相对 cwd 的路径（供 agent 与 multimodal 读取）。"""
    if not paths:
        return None
    cwd = Path.cwd()
    out: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            try:
                out.append(str(p.relative_to(cwd)))
            except ValueError:
                out.append(str(p))
    return out or None


def cmd_embed(args: argparse.Namespace) -> None:
    path = Path(args.file)
    text, meta = load_file(str(path))
    n_chunks = vector_store.embed_document(
        text=text,
        namespace=args.namespace,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extra_metadata=meta,
    )
    print(f"[embed] namespace={args.namespace}, chunks_added={n_chunks}")


def cmd_search(args: argparse.Namespace) -> None:
    queries = expand_retrieval_queries(
        args.query,
        strategy=args.strategy,
        llm=qwen,
    )
    if not queries:
        fb = (args.query or "").strip()
        queries = [fb] if fb else []
    docs_and_scores = retrieve_with_public_merge(
        vector_store,
        queries=queries,
        namespace=args.namespace,
        k=args.k,
        score_threshold=args.score_threshold,
        strategy=args.strategy,
    )
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = getattr(doc, "metadata", {})
        print(f"\n[{i}] score={score:.3f}, source={meta.get('source', 'unknown')}")
        print(getattr(doc, "page_content", "")[:300])


def cmd_chat(args: argparse.Namespace) -> None:
    from tools.agent.middleware import (
        LoggingMiddleware,
        default_business_middleware,
    )

    middleware = []
    if getattr(args, "log_middleware", False):
        middleware.append(LoggingMiddleware())
    if getattr(args, "business_middleware", False):
        middleware.extend(
            default_business_middleware(
                call_limit=getattr(args, "call_limit", 20),
                stats_file=getattr(args, "stats_file", "data/logs/agent_stats.txt"),
                summarization=bool(getattr(args, "summarization", False)),
                session_paper_context=bool(getattr(args, "session_paper_context", False)),
            )
        )
    elif getattr(args, "session_paper_context", False):
        from tools.agent.session_paper_context import SessionPaperContextMiddleware

        middleware.append(SessionPaperContextMiddleware())
    agent = RAGAgent(middleware=middleware)
    attach = getattr(args, "attach_paper", None)
    if attach:
        from tools.agent.paper_session_mirror import mirror_local_paper_to_namespace

        mirror_note = mirror_local_paper_to_namespace(
            vector_store,
            args.namespace,
            attach,
            replace=bool(getattr(args, "attach_paper_replace", False)),
        )
        print(f"[attach-paper] {mirror_note}")
    embed_files = getattr(args, "embed_file", None) or []
    if embed_files:
        import uuid
        from pathlib import Path

        from tools.agent.session_file_embed import embed_session_file
        from tools.agent.conversation import conversation_manager

        sid = getattr(args, "session_id", None)
        for fp in embed_files:
            ingest_id = uuid.uuid4().hex
            extra: dict = {"upload_via": "cli_chat", "session_ingest_id": ingest_id}
            if sid:
                extra["session_id"] = str(sid).strip()
            try:
                res = embed_session_file(
                    fp,
                    args.namespace,
                    chunk_size=int(getattr(args, "embed_chunk_size", DEFAULT_CHUNK_SIZE)),
                    chunk_overlap=int(
                        getattr(args, "embed_chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                    ),
                    extra_meta=extra,
                )
                if (
                    sid
                    and str(sid).strip()
                    and int(res.chunks_added or 0) > 0
                ):
                    safe_name = Path(str(fp)).name
                    conversation_manager.add_session_embed(
                        str(sid).strip(),
                        ingest_id=ingest_id,
                        namespace=str(args.namespace or "").strip(),
                        filename=safe_name,
                        chunks_added=res.chunks_added,
                    )
                print(
                    f"[embed-file] {fp} -> namespace={args.namespace}, chunks={res.chunks_added}, ingest_id={ingest_id}"
                )
                if res.arxiv_id:
                    print(
                        f"[paper] arxiv_id={res.arxiv_id} "
                        f"library_ingested={res.paper_library_ingested}"
                    )
                    if res.paper_ingest_note:
                        print(f"[paper] {res.paper_ingest_note}")
            except Exception as exc:
                print(f"[embed-file] FAILED {fp}: {exc}")
    user_img = _cli_user_image_paths(getattr(args, "image", None) or [])
    if getattr(args, "use_tools", False):
        _pf = getattr(args, "prefetch_subq_web", False)
        result = agent.answer_with_tools(
            question=args.question,
            namespace=args.namespace,
            k=args.k,
            session_id=getattr(args, "session_id", None),
            user_image_paths=user_img,
            prefetch_subq_web=True if _pf else None,
        )
    else:
        result = agent.answer(
            question=args.question,
            namespace=args.namespace,
            strategy=args.strategy,
            k=args.k,
            score_threshold=args.score_threshold,
            session_id=getattr(args, "session_id", None),
            user_image_paths=user_img,
        )
    print("\n[Answer]")
    print(result["answer"])
    if result.get("web_fallback"):
        print("\n[Note] 本地向量库未命中，已使用联网搜索摘要作为上下文。")
    elif result.get("web_forced_by_user"):
        print("\n[Note] 检测到您要求联网检索，已在本地上下文基础上合并网络摘要。")
    elif result.get("web_supplement"):
        rs = result.get("retrieval_context_score")
        rr = result.get("retrieval_judge_reason") or ""
        print("\n[Note] 本地已命中但检索充分性评分偏低，已补充联网摘要。")
        if rs is not None:
            print(f"  检索充分性评分: {rs}/10")
        if rr:
            print(f"  评判说明: {rr[:300]}{'…' if len(rr) > 300 else ''}")
    elif result.get("web_search_error"):
        print("\n[Note] 已尝试联网兜底，但未获得可用摘要：")
        print(result["web_search_error"])
    print("\n[Citations]")
    for i, c in enumerate(result["citations"], start=1):
        print(f"{i}. source={c['source']}, score={c['score']:.3f}")
        if c.get("subquestion"):
            print(f"   子问题: {c['subquestion']}")
        print(c["preview"])
        print("---")
    _append_chat_log(
        question=args.question,
        namespace=args.namespace,
        answer=result["answer"],
        citations=result["citations"],
    )


def cmd_session_finalize(args: argparse.Namespace) -> None:
    """离开前显式刷新 ``<session>_paper_context.json``（不跑 RAG、不追加用户句）。"""
    import json

    from tools.agent.session_paper_context import finalize_session_paper_context

    sid = (getattr(args, "session_id", None) or "").strip()
    ns = (getattr(args, "namespace", None) or "").strip()
    if not sid:
        print("请提供 --session-id。")
        return
    if not ns:
        print("请提供 --namespace（须与当时 chat 的 --namespace 一致）。")
        return
    path = finalize_session_paper_context(sid, ns)
    if not path:
        print("写入失败：请确认 data/conversations 可写，且 session 历史已持久化。")
        return
    print(f"[session_finalize] 已刷新：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = data.get("arxiv_ids") or []
        print("arxiv_ids:", ", ".join(str(x) for x in ids) if ids else "（无）")
        ts = (data.get("topic_summary") or "").strip()
        print("topic_summary:\n" + (ts[:1200] + ("…" if len(ts) > 1200 else "")))
    except Exception as exc:
        print(f"读取快照失败：{exc}")


def cmd_papers(args: argparse.Namespace) -> None:
    papers = search_arxiv(
        query=args.query,
        max_results=args.max_results,
        category=args.category,
        sort_by=args.sort_by,
    )
    if not papers:
        print("No papers found.")
        return
    for i, p in enumerate(papers, start=1):
        authors = ", ".join(p.authors)
        print(f"\n[{i}] {p.title}")
        print(f"  authors : {authors}")
        print(f"  published: {p.published}")
        print(f"  url     : {p.url}")


def cmd_embed_paper(args: argparse.Namespace) -> None:
    """根据查询在 arXiv 上选中一篇论文，并把摘要嵌入向量库。"""

    papers = search_arxiv(
        query=args.query,
        max_results=args.max_results,
        category=args.category,
        sort_by=args.sort_by,
    )
    if not papers:
        print("No papers found.")
        return
    index = args.index - 1
    if index < 0 or index >= len(papers):
        print(f"Invalid index {args.index}, got {len(papers)} results.")
        return
    paper = papers[index]
    arxiv_id = get_arxiv_id(paper)
    namespace = args.namespace or f"paper:{arxiv_id}"
    text = paper.summary
    meta = {
        "source": paper.url,
        "title": paper.title,
        "authors": ", ".join(paper.authors),
        "arxiv_id": arxiv_id,
        "type": "paper_abstract",
    }
    n_chunks = vector_store.embed_document(
        text=text,
        namespace=namespace,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        extra_metadata=meta,
    )
    print(
        f"[embed_paper] namespace={namespace}, arxiv_id={arxiv_id}, chunks_added={n_chunks}"
    )


def cmd_embed_paper_full(args: argparse.Namespace) -> None:
    """根据 arXiv ID 下载论文 PDF 并将全文嵌入向量库。"""

    arxiv_id = args.arxiv_id
    try:
        pdf_path = download_pdf(arxiv_id, dest_dir=args.pdf_dir)
    except RuntimeError as e:
        print(str(e))
        return

    text, meta = load_pdf(str(pdf_path), parent_id=arxiv_id)
    namespace = args.namespace or f"paper:{arxiv_id}:full"
    meta.update(
        {
            "arxiv_id": arxiv_id,
            "type": "paper_full",
        }
    )
    n_chunks = vector_store.embed_document(
        text=text,
        namespace=namespace,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        extra_metadata=meta,
    )
    print(
        f"[embed_paper_full] namespace={namespace}, arxiv_id={arxiv_id}, chunks_added={n_chunks}"
    )


def cmd_explain_file(args: argparse.Namespace) -> None:
    """嵌入任意支持格式的文件并让 LLM 进行解读/总结。"""

    path = Path(args.file)
    text, meta = load_file(str(path))
    namespace = args.namespace or f"file:{path.stem}"
    n_chunks = vector_store.embed_document(
        text=text,
        namespace=namespace,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        extra_metadata=meta,
    )
    print(
        f"[embed_file] namespace={namespace}, file={path.name}, chunks_added={n_chunks}"
    )
    agent = RAGAgent()
    result = agent.answer(
        question=args.question,
        namespace=namespace,
        strategy=DEFAULT_RETRIEVAL_STRATEGY,
        k=DEFAULT_TOP_K,
        score_threshold=0.5,
    )
    print("\n[Explanation]")
    print(result["answer"])
    _append_chat_log(
        question=args.question,
        namespace=namespace,
        answer=result["answer"],
        citations=result["citations"],
    )


def cmd_preview_subquestions(args: argparse.Namespace) -> None:
    """打印 LLM 子问题拆分结果与最终检索 query，便于质检（不访问向量库）。"""

    from config import RAG_LLM_QUERY_REWRITE, RAG_SUBQUESTION_SPLIT
    from tools.rag.language import expand_retrieval_queries, split_compound_question

    q = (args.question or "").strip()
    if not q:
        print("请提供 --question")
        return

    print("\n=== 1) LLM 子问题拆分（force=True，不受 RAG_SUBQUESTION_SPLIT 影响）===")
    forced = split_compound_question(q, qwen, force=True)
    for i, s in enumerate(forced, start=1):
        print(f"  {i}. {s}")
    print(f"  （共 {len(forced)} 条）")

    print("\n=== 2) 当前配置下的子问题（与线上一致，受 RAG_SUBQUESTION_SPLIT 影响）===")
    actual_subs = split_compound_question(q, qwen, force=False)
    for i, s in enumerate(actual_subs, start=1):
        print(f"  {i}. {s}")
    print(
        f"  （共 {len(actual_subs)} 条；RAG_SUBQUESTION_SPLIT={RAG_SUBQUESTION_SPLIT}）"
    )

    use_split = False if getattr(args, "no_split", False) else None
    use_rw = False if getattr(args, "no_rewrite", False) else None
    queries = expand_retrieval_queries(
        q,
        strategy=args.strategy,
        llm=qwen,
        use_subquestion_split=use_split,
        use_llm_rewrite=use_rw,
    )
    print(
        f"\n=== 3) 展开后用于检索的 query（strategy={args.strategy}, "
        f"RAG_LLM_QUERY_REWRITE={RAG_LLM_QUERY_REWRITE}）==="
    )
    if getattr(args, "no_split", False):
        print("  （本段已传 --no-split，不进行子问题拆分）")
    if getattr(args, "no_rewrite", False):
        print("  （本段已传 --no-rewrite，不进行 Query 改写）")
    for i, rq in enumerate(queries, start=1):
        print(f"  {i}. {rq}")
    print(f"  （共 {len(queries)} 条）\n")


def cmd_daily_summary(args: argparse.Namespace) -> None:
    """对指定 namespace 做最近 N 天的知识快照/每日总结。"""

    from datetime import timedelta

    namespace = args.namespace
    days = args.days
    k = args.k

    # 使用一个通用查询来召回尽可能多的片段，再按时间过滤（不做复合问题拆分）
    queries = expand_retrieval_queries(
        "recent updates and key points",
        strategy=DEFAULT_RETRIEVAL_STRATEGY,
        llm=qwen,
        use_subquestion_split=False,
    )
    docs_and_scores = retrieve_with_public_merge(
        vector_store,
        queries=queries,
        namespace=namespace,
        k=k * 5,
        score_threshold=args.score_threshold,
        strategy=DEFAULT_RETRIEVAL_STRATEGY,
    )

    since = now_utc() - timedelta(days=days)
    filtered = filter_by_time(docs_and_scores, since=since)
    if not filtered:
        print("No documents found for the given time window.")
        return

    messages = build_rag_prompt(args.question, filtered[:k])
    resp = qwen.invoke(messages)
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    print("\n[Daily Summary]")
    print(answer_text)
    _append_chat_log(
        question=args.question,
        namespace=namespace,
        answer=answer_text,
        citations=[],
    )

def cmd_agent_run(args: argparse.Namespace) -> None:
    """规划器 -> 步骤循环 -> 长期记忆。"""

    from tools.agent.middleware import default_business_middleware

    middleware = []
    if getattr(args, "business_middleware", False):
        middleware.extend(
            default_business_middleware(
                call_limit=getattr(args, "call_limit", 30),
                stats_file=getattr(args, "stats_file", "data/logs/agent_stats.txt"),
                summarization=bool(getattr(args, "summarization", False)),
            )
        )

    agent = RAGAgent(middleware=middleware)
    result = agent.run_autonomous(
        goal=args.goal,
        namespace=args.namespace,
        session_id=getattr(args, "session_id", None),
        k=args.k,
        max_steps=args.max_steps,
        write_memory=not getattr(args, "no_memory", False),
        memory_k=args.memory_k,
    )

    print("\n[Plan]")
    for s in result.get("plan", {}).get("steps", []):
        print(f"- {s.get('id')}: {s.get('title')} :: {s.get('instruction')}")

    print("\n[Final Answer]")
    print(result["answer"])

    print("\n[Citations]")
    for i, c in enumerate(result.get("citations", []), start=1):
        print(f"{i}. source={c.get('source')}, score={float(c.get('score', 0.0)):.3f}")
        print((c.get("preview") or "")[:300])
        print("---")

    mem = result.get("memory", {}) or {}
    if mem.get("written"):
        print("\n[Memory] written to vector store.")
    else:
        print("\n[Memory] not written.")


def cmd_list_papers(args: argparse.Namespace) -> None:
    """列出当前本地已保存的论文向量库（Chroma collection 名以 paper_ 开头）。"""

    names = vector_store.list_collection_names()
    items = []
    for name in sorted(names):
        if not name.startswith("paper_"):
            continue
        # Chroma collection 名为 sanitized（清洗后的标识符）：paper_2401_12345_full -> arxiv_id 约 2401.12345
        rest = name[6:]  # 去掉前缀 "paper_"
        is_full = rest.endswith("_full")
        if is_full:
            rest = rest[:-5]
        arxiv_id = rest.replace("_", ".") if rest else "unknown"
        items.append((name, arxiv_id, is_full))

    if not items:
        print("No paper collections found in Chroma (data/chroma).")
        return

    print("Local paper collections (Chroma):")
    for i, (coll, arxiv_id, is_full) in enumerate(items, start=1):
        kind = "full" if is_full else "abstract"
        print(f"[{i}] collection={coll}, arxiv_id≈{arxiv_id}, type={kind}")


def cmd_chat_paper(args: argparse.Namespace) -> None:
    """根据 arXiv ID 直接对某篇已嵌入的论文进行聊天。"""

    # 根据是否指定 --full 选择 namespace
    if args.full:
        namespace = f"paper:{args.arxiv_id}:full"
    else:
        namespace = f"paper:{args.arxiv_id}"

    # 确保从磁盘加载已保存的索引（如果存在）
    try:
        vector_store.load()
    except Exception:
        # 加载失败时也尝试继续，可能当前进程中已经有内存索引
        pass

    user_img = _cli_user_image_paths(getattr(args, "image", None) or [])
    agent = RAGAgent()
    result = agent.answer(
        question=args.question,
        namespace=namespace,
        strategy=args.strategy,
        k=args.k,
        score_threshold=args.score_threshold,
        session_id=getattr(args, "session_id", None),
        user_image_paths=user_img,
    )
    print(f"\n[Answer] namespace={namespace}")
    print(result["answer"])
    print("\n[Citations]")
    for i, c in enumerate(result["citations"], start=1):
        print(f"{i}. source={c['source']}, score={c['score']:.3f}")
        print(c["preview"])
        print("---")
    _append_chat_log(
        question=args.question,
        namespace=namespace,
        answer=result["answer"],
        citations=result["citations"],
    )


def _append_chat_log(
    question: str,
    namespace: str,
    answer: str,
    citations: list[dict],
    log_path: str = "logs/chat_history.jsonl",
) -> None:
    """聊天记录写入 logs/*.jsonl，并把本轮问答嵌入与 RAG 相同的 namespace。

    这样下一轮 `chat`/`search` 使用同一 `--namespace` 时，本地向量检索可命中上一轮
    的问答内容（metadata.type=chat_memory），而不必依赖单独的 `{ns}:chat` collection。
    """

    import json
    from datetime import datetime, timezone

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "question": question,
        "answer": answer,
        "citations": citations,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入与 CLI `--namespace` 一致的向量库，供后续检索（含联网兜底前的本地 RAG）
    try:
        embed_chat_turn_into_rag_namespace(
            namespace=namespace,
            question=question,
            answer=answer,
            citations=citations,
            source="cli_chat_turn",
        )
    except Exception:
        # 不影响主流程，安静失败
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartSearchAssistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="Embed a text file into the vector store")
    p_embed.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_embed.add_argument("--file", required=True)
    p_embed.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_embed.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="Search in the vector store")
    p_search.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p_search.add_argument(
        "--strategy",
        default=DEFAULT_RETRIEVAL_STRATEGY,
        choices=["default", "multi_query", "hybrid", "hybrid_rerank", "rerank"],
        help="default|multi_query|hybrid(语义+BM25 RRF)|hybrid_rerank|rerank",
    )
    p_search.add_argument("--score-threshold", type=float, default=0.5)
    p_search.set_defaults(func=cmd_search)

    p_chat = sub.add_parser("chat", help="Full RAG QA with LLM")
    p_chat.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_chat.add_argument("--question", required=True)
    p_chat.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p_chat.add_argument(
        "--strategy",
        default=DEFAULT_RETRIEVAL_STRATEGY,
        choices=["default", "multi_query", "hybrid", "hybrid_rerank", "rerank"],
        help="default|multi_query|hybrid|hybrid_rerank|rerank",
    )
    p_chat.add_argument("--score-threshold", type=float, default=0.5)
    p_chat.add_argument(
        "--session-id",
        default=None,
        help="session id for conversation context (multi-turn)",
    )
    p_chat.add_argument(
        "--attach-paper",
        default=None,
        metavar="ARXIV_ID",
        help=(
            "在本轮问答前，将论文库 paper:<id>:full 只读镜像到当前 --namespace（"
            "不修改论文库/SQLite；可用 --attach-paper-replace 先删会话内旧镜像）"
        ),
    )
    p_chat.add_argument(
        "--attach-paper-replace",
        action="store_true",
        help="镜像前删除当前 namespace 内该篇 session_paper_mirror 文档，避免重复堆积",
    )
    p_chat.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="随问题附带的本地图片路径（多模态对话；可重复）",
    )
    p_chat.add_argument(
        "--embed-file",
        action="append",
        default=[],
        metavar="PATH",
        help="本轮问答前将本地文件嵌入当前 --namespace（与 load_file 格式一致；可重复）",
    )
    p_chat.add_argument(
        "--embed-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="--embed-file 分块大小（默认与全局 DEFAULT_CHUNK_SIZE 一致）",
    )
    p_chat.add_argument(
        "--embed-chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="--embed-file 分块重叠",
    )
    p_chat.add_argument(
        "--session-paper-context",
        action="store_true",
        help=(
            "启用会话论文上下文中间件：每轮后把涉及的 arXiv ID 与话题摘要写入 "
            "data/conversations/<session>_paper_context.json（topic_summary 默认由 LLM 润色，"
            "SESSION_PAPER_CONTEXT_LLM_SUMMARY=0 可改回仅滚动拼接；另存 topic_summary_rolling）；"
            "下次同 --session-id + --namespace 启动时自动 replace 镜像这些论文到会话向量库。"
            "离开前可再执行：python cli.py session_finalize --session-id ... --namespace ..."
        ),
    )
    p_chat.add_argument(
        "--use-tools",
        action="store_true",
        help="use LangChain Tool Calling (LLM chooses weather/arxiv/knowledge tools)",
    )
    p_chat.add_argument(
        "--prefetch-subq-web",
        action="store_true",
        help=(
            "需与 --use-tools 合用：首轮前先按子问题拆分并联网检索（search_web_snippets，"
            "优先 MCP），摘要注入 system；也可用环境变量 RAG_TOOLS_PREFETCH_SUBQ_WEB=1 默认开启"
        ),
    )
    p_chat.add_argument(
        "--log-middleware",
        action="store_true",
        help="enable LoggingMiddleware (print before_agent/after_model/etc.)",
    )
    p_chat.add_argument(
        "--business-middleware",
        action="store_true",
        help="enable input validation, call limit, PII masking, usage stats",
    )
    p_chat.add_argument(
        "--summarization",
        action="store_true",
        help="enable LangChain SummarizationMiddleware (summarize history when long)",
    )
    p_chat.add_argument(
        "--call-limit",
        type=int,
        default=20,
        help="max LLM/tool calls per session when using business middleware (default: 20)",
    )
    p_chat.add_argument(
        "--stats-file",
        default="data/logs/agent_stats.txt",
        help="file to append usage stats when using business middleware",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_sess_fin = sub.add_parser(
        "session_finalize",
        help=(
            "结束当前会话分支：仅根据已保存的对话 jsonl 刷新 "
            "<session>_paper_context.json（arXiv 列表 + 摘要），不调用 RAG。"
            "平时若已加 --session-paper-context，每轮答完也会自动保存；本命令用于离开前再整段压一次摘要。"
        ),
    )
    p_sess_fin.add_argument("--session-id", required=True)
    p_sess_fin.add_argument(
        "--namespace",
        required=True,
        help="必须与 chat 时 --namespace 一致，否则下次 resume 镜像会对错分区",
    )
    p_sess_fin.set_defaults(func=cmd_session_finalize)

    p_papers = sub.add_parser("papers", help="Search papers on arXiv")
    p_papers.add_argument("--query", required=True)
    p_papers.add_argument("--max-results", type=int, default=5)
    p_papers.add_argument("--category", default=None)
    p_papers.add_argument(
        "--sort-by",
        default="relevance",
        choices=["relevance", "lastUpdatedDate"],
    )
    p_papers.set_defaults(func=cmd_papers)

    p_embed_paper = sub.add_parser(
        "embed_paper", help="Search arXiv and embed one paper abstract into vector store"
    )
    p_embed_paper.add_argument("--query", required=True)
    p_embed_paper.add_argument("--index", type=int, default=1, help="1-based index")
    p_embed_paper.add_argument("--max-results", type=int, default=5)
    p_embed_paper.add_argument("--category", default=None)
    p_embed_paper.add_argument(
        "--sort-by",
        default="relevance",
        choices=["relevance", "lastUpdatedDate"],
    )
    p_embed_paper.add_argument(
        "--namespace",
        default=None,
        help="optional namespace; default is paper:<arxiv_id>",
    )
    p_embed_paper.set_defaults(func=cmd_embed_paper)

    p_embed_paper_full = sub.add_parser(
        "embed_paper_full",
        help="Download arXiv PDF by id and embed full text into vector store",
    )
    p_embed_paper_full.add_argument("--arxiv-id", required=True)
    p_embed_paper_full.add_argument(
        "--namespace",
        default=None,
        help="optional namespace; default is paper:<arxiv_id>:full",
    )
    p_embed_paper_full.add_argument(
        "--pdf-dir",
        default="data/papers",
        help="directory to store downloaded PDFs",
    )
    p_embed_paper_full.set_defaults(func=cmd_embed_paper_full)

    p_explain_file = sub.add_parser(
        "explain_file",
        help="Embed a file (txt/pdf/docx/xlsx/md/html, etc.) then ask LLM to explain it",
    )
    p_explain_file.add_argument("--file", required=True)
    p_explain_file.add_argument(
        "--namespace",
        default=None,
        help="optional namespace; default is file:<stem>",
    )
    p_explain_file.add_argument(
        "--question",
        default="请用中文总结一下这份文件的重点内容。",
    )
    p_explain_file.set_defaults(func=cmd_explain_file)

    p_daily = sub.add_parser(
        "daily_summary",
        help="Generate a recent N-days knowledge snapshot for a namespace",
    )
    p_daily.add_argument(
        "--namespace",
        required=True,
        help="target namespace, e.g. project:rag_book",
    )
    p_daily.add_argument(
        "--days",
        type=int,
        default=1,
        help="look back N days (default: 1)",
    )
    p_daily.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of chunks to summarize",
    )
    p_daily.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )
    p_daily.add_argument(
        "--question",
        default="请用中文总结一下最近这段时间在该知识库中的关键信息和变化。",
    )
    p_daily.set_defaults(func=cmd_daily_summary)

    p_subq = sub.add_parser(
        "preview_subquestions",
        help="质检：打印 LLM 拆分的子问题与最终检索 query（不写向量库）",
    )
    p_subq.add_argument("--question", required=True, help="用户原始提问（可含多个子问题）")
    p_subq.add_argument(
        "--strategy",
        default=DEFAULT_RETRIEVAL_STRATEGY,
        choices=["default", "multi_query", "hybrid", "hybrid_rerank", "rerank"],
        help="与 RAG retrieve 策略一致，影响第 3 段展开方式",
    )
    p_subq.add_argument(
        "--no-split",
        action="store_true",
        help="第 3 段 expand 时不做子问题拆分（对比用）",
    )
    p_subq.add_argument(
        "--no-rewrite",
        action="store_true",
        help="第 3 段不做 LLM Query 改写（对比用）",
    )
    p_subq.set_defaults(func=cmd_preview_subquestions)

    p_agent = sub.add_parser(
        "agent_run",
        help="Autonomous agent run (planner -> step loop -> long-term memory)",
    )
    p_agent.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_agent.add_argument("--goal", required=True, help="goal for the autonomous run")
    p_agent.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.add_argument("--memory-k", type=int, default=6)
    p_agent.add_argument(
        "--session-id",
        default=None,
        help="session id for long-term memory + conversation context",
    )
    p_agent.add_argument(
        "--no-memory",
        action="store_true",
        help="do not write long-term memory at the end",
    )
    p_agent.add_argument(
        "--business-middleware",
        action="store_true",
        help="enable input validation, call limit, PII masking, usage stats",
    )
    p_agent.add_argument(
        "--summarization",
        action="store_true",
        help="enable LangChain SummarizationMiddleware (summarize executor messages when long)",
    )
    p_agent.add_argument(
        "--call-limit",
        type=int,
        default=30,
        help="max calls per session when using business middleware (default: 30)",
    )
    p_agent.add_argument(
        "--stats-file",
        default="data/logs/agent_stats.txt",
        help="file to append usage stats when using business middleware",
    )
    p_agent.set_defaults(func=cmd_agent_run)

    p_list_papers = sub.add_parser(
        "list_papers",
        help="List local paper namespaces stored in vector store",
    )
    p_list_papers.set_defaults(func=cmd_list_papers)

    p_chat_paper = sub.add_parser(
        "chat_paper",
        help="Chat with a specific paper by arXiv id using existing vector store",
    )
    p_chat_paper.add_argument("--arxiv-id", required=True)
    p_chat_paper.add_argument(
        "--full",
        action="store_true",
        help="use full-text namespace paper:<id>:full instead of abstract-only",
    )
    p_chat_paper.add_argument(
        "--question",
        required=True,
        help="question to ask about this paper",
    )
    p_chat_paper.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of chunks to retrieve",
    )
    p_chat_paper.add_argument(
        "--strategy",
        default=DEFAULT_RETRIEVAL_STRATEGY,
        choices=["default", "multi_query", "hybrid", "hybrid_rerank", "rerank"],
        help="retrieval strategy",
    )
    p_chat_paper.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="score threshold for retrieved chunks",
    )
    p_chat_paper.add_argument(
        "--session-id",
        default=None,
        help="session id for conversation context",
    )
    p_chat_paper.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="随问题附带的本地图片路径（多模态；可重复）",
    )
    p_chat_paper.set_defaults(func=cmd_chat_paper)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _print_runtime_mode_banner()
    args.func(args)


if __name__ == "__main__":
    main()

