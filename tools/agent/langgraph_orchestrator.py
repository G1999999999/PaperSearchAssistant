from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from tools.rag.math_utils import merge_ranked_lists

from config import DEFAULT_RETRIEVAL_STRATEGY
from prompts import build_rag_prompt
from tools.agent.conversation import (
    collect_session_embed_ids_for_namespace,
    conversation_manager,
)
from tools.agent.middleware import (
    run_after_agent,
    run_after_model,
    run_before_model,
    trace_event,
)
from tools.agent.router import (
    Route,
    allow_web_search_when_local_misses,
    extract_arxiv_id,
    paper_namespace_arxiv_id,
)
from tools.retrieval.query_router import build_query_route
from tools.rag.multimodal_content import dict_messages_to_lc


class ChatGraphState(TypedDict, total=False):
    question: str
    namespace: str
    strategy: str
    k: int
    score_threshold: float
    use_multi_source: bool
    session_id: str | None
    user_image_paths: list[str] | None
    use_tools: bool
    prefetch_subq_web: bool | None
    route: Any
    paper_intent: bool
    selected_path: str
    grouped: list[Any]
    grouped_for_prompt: list[Any]
    grouped_for_citations: list[Any]
    image_only_mode: bool
    include_user_images_in_history: bool
    retrieval_context_score: float | None
    retrieval_judge_reason: str | None
    web_fallback_used: bool
    web_supplement_used: bool
    web_forced_by_user: bool
    need_web_supplement: bool
    paper_glossary_web_supplement: bool
    result: dict[str, Any]


def _route_node(state: ChatGraphState) -> ChatGraphState:
    question = str(state.get("question") or "")
    user_images = state.get("user_image_paths") or []
    use_tools = bool(state.get("use_tools"))
    decision = build_query_route(question)
    route = decision.route
    paper = decision.intent in (
        "paper_search",
        "paper_read",
        "paper_qa",
        "local_paper_search",
    )
    # 兜底：内容问答 + arXiv id 应视为 paper_qa 本地阅读意图。
    try:
        from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

        if looks_like_paper_content_qa(question) and extract_arxiv_id(question):
            paper = True
        # router 常把「仅方法/实验/结论」判成 local_rag（句子里无「论文」），
        # 但会话已绑定论文、分区为 paper: 或历史里出现过 arXiv —— 必须仍走 RAG，勿进 tools（否则 LLM 会去搜聊天历史）。
        if (not paper) and looks_like_paper_content_qa(question):
            sid = (state.get("session_id") or "").strip()
            ns = str(state.get("namespace") or "").strip()
            if ns.startswith("paper:"):
                paper = True
            elif sid:
                try:
                    from tools.retrieval.session_paper_state import paper_state_store

                    cur = paper_state_store.get_current_paper(sid)
                    if cur and str(cur.get("arxiv_id") or "").strip():
                        paper = True
                except Exception:
                    pass
                if not paper:
                    agent = state.get("_agent")
                    if agent is not None:
                        try:
                            recent = conversation_manager.get_recent_messages(sid)
                            if agent._extract_last_arxiv_id_from_history(recent):
                                paper = True
                        except Exception:
                            pass
    except Exception:
        pass

    # 本地论文列表须优先于 tools loop（否则「列出本地库论文」会误进工具链）
    if decision.intent == "local_paper_search" and (not user_images):
        selected = "local_library_list"
    # 论文问答/阅读/搜索统一优先走本地 RAG，不进入 tools 分支（避免误触 online arXiv）。
    elif paper:
        selected = "rag"
    # 优先保留原行为：带图时不走 tools loop。
    elif use_tools and (not user_images):
        selected = "tools"
    elif route is Route.WEATHER:
        selected = "weather"
    elif bool(state.get("session_id")) and bool(getattr(state.get("_agent"), "_is_history_query", lambda _q: False)(question)):
        selected = "history"
    elif (not paper) and (not getattr(state.get("_agent"), "_is_history_query", lambda _q: False)(question)):
        selected = "nonpaper"
    else:
        selected = "rag"

    out: ChatGraphState = {
        "route": route,
        "paper_intent": bool(paper),
        "selected_path": selected,
    }
    # Trace 断言：论文意图不应进入 tools 分支；若发生，记录高优先级告警用于回归排查。
    if paper and selected == "tools":
        trace_event(
            "assert_paper_intent_routed_to_tools",
            {
                "intent": decision.intent,
                "sub_intent": decision.sub_intent,
                "question": question[:220],
            },
        )
    trace_event(
        "langgraph_route",
        {
            "route": str(route),
            "paper_intent": bool(paper),
            "intent": decision.intent,
            "sub_intent": decision.sub_intent,
            "selected_path": selected,
            "has_user_images": bool(user_images),
            "use_tools": use_tools,
        },
    )
    return out


def _local_library_list_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "local_library_list"})
    question = str(state.get("question") or "")
    session_id = state.get("session_id")
    user_image_paths = state.get("user_image_paths")
    out = agent.local_paper_search_list_response(question, session_id)
    run_after_agent(agent.middleware, out)
    if session_id:
        try:
            conversation_manager.add_turn(
                session_id, "user", question, image_paths=user_image_paths
            )
            conversation_manager.add_turn(
                session_id, "assistant", str(out.get("answer") or "")
            )
        except Exception:
            pass
    return {"result": out}


def _rag_local_retrieve_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event(
        "langgraph_node_enter",
        {"node": "rag_local_retrieve", "strategy": state.get("strategy"), "k": state.get("k")},
    )
    question_raw = str(state.get("question") or "")
    session_rag_namespace = str(state.get("namespace") or "")
    strategy = str(state.get("strategy") or DEFAULT_RETRIEVAL_STRATEGY)
    k = int(state.get("k") or 4)
    score_threshold = float(state.get("score_threshold") or 0.5)
    use_multi_source = bool(state.get("use_multi_source"))
    session_id = state.get("session_id")
    paper_intent = bool(state.get("paper_intent"))
    user_image_paths = state.get("user_image_paths")

    # 与 agent.answer() 一致：下载/入库/候选确认须先走论文入库对话，不能走纯 RAG 生成（否则会误答「不能下载」）。
    early_ingest = agent._maybe_handle_paper_ingest_dialogue(
        question_raw,
        session_id,
        user_image_paths=user_image_paths,
    )
    if early_ingest is not None:
        run_after_agent(agent.middleware, early_ingest)
        trace_event(
            "langgraph_paper_ingest_early_return",
            {"question": question_raw[:220]},
        )
        # 会话写入已由 _maybe_handle_paper_ingest_dialogue 内 _finish() 完成，这里不再重复 add_turn。
        return {
            "question": question_raw,
            "result": early_ingest,
            "grouped": [],
            "grouped_for_prompt": [],
            "grouped_for_citations": [],
            "web_fallback_used": False,
            "web_supplement_used": False,
            "web_forced_by_user": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": None,
        }

    question, read_paper_merge_ns, _cur_paper, early_bind = (
        agent.binding_and_paper_merge_namespace(
            question_raw, session_id, session_rag_namespace
        )
    )
    if early_bind is not None:
        run_after_agent(agent.middleware, early_bind)
        if session_id:
            try:
                conversation_manager.add_turn(
                    session_id,
                    "user",
                    question_raw,
                    image_paths=user_image_paths,
                )
                conversation_manager.add_turn(
                    session_id,
                    "assistant",
                    str(early_bind.get("answer") or ""),
                )
            except Exception:
                pass
        return {
            "question": question,
            "result": early_bind,
            "grouped": [],
            "grouped_for_prompt": [],
            "grouped_for_citations": [],
            "web_fallback_used": False,
            "web_supplement_used": False,
            "web_forced_by_user": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": None,
        }

    session_ingest_ids: list[str] = []
    if session_id:
        session_ingest_ids = collect_session_embed_ids_for_namespace(
            conversation_manager.get_recent_messages(session_id),
            session_rag_namespace,
        )

    if (not read_paper_merge_ns) and str(session_rag_namespace).startswith("paper:"):
        read_paper_merge_ns = session_rag_namespace
    # 兜底：论文意图 + 问句显式 arXiv id 时，强制绑定到 paper:<id>:full，
    # 避免偶发路由/会话态导致仍在默认分区检索。
    aid_in_question = extract_arxiv_id(question or "")
    if (not read_paper_merge_ns) and aid_in_question and paper_intent:
        try:
            forced_ns = agent._namespace_for_arxiv_id(aid_in_question)
            if forced_ns:
                read_paper_merge_ns = forced_ns
                trace_event(
                    "paper_ns_forced_by_arxiv_id",
                    {"arxiv_id": str(aid_in_question), "namespace": str(forced_ns)},
                )
        except Exception:
            pass

    # 路由偶发未标 paper 时，只要已绑定到 paper:… 分区，仍按论文检索处理。
    paper_intent_effective = paper_intent or (
        bool(read_paper_merge_ns) and str(read_paper_merge_ns).startswith("paper:")
    )

    method_focus_query = agent._question_wants_method(question)
    retrieved: list[Any] = []

    if (
        read_paper_merge_ns
        and str(read_paper_merge_ns).startswith("paper:")
        and paper_intent_effective
        and method_focus_query
    ):
        paper_hits = agent._retrieve_local(
            question=question,
            namespace=read_paper_merge_ns,
            strategy=strategy,
            k=max(int(k or 4), 20),
            score_threshold=min(float(score_threshold or 0.5), 0.15),
            session_ingest_ids=None,
            paper_intent_hint=True,
        )
        paper_hits = agent.store.expand_neighbor_chunks(
            retrieved=paper_hits,
            namespace=read_paper_merge_ns,
            window=2,
        )
        retrieved = list(paper_hits)[: max(k * 3, 24)]
        trace_event(
            "paper_method_primary_retrieve",
            {
                "namespace": read_paper_merge_ns,
                "k": max(int(k or 4), 20),
                "score_threshold": min(float(score_threshold or 0.5), 0.15),
                "hits": len(retrieved),
                "session_merged": False,
                "langgraph": True,
            },
        )
    elif read_paper_merge_ns and read_paper_merge_ns != session_rag_namespace:

        def _session_retrieval_bundle() -> list[Any]:
            if use_multi_source:
                return agent._retrieve_multi_source(
                    question=question,
                    namespace=session_rag_namespace,
                    k=k,
                    score_threshold=score_threshold,
                    session_ingest_ids=session_ingest_ids or None,
                )
            loc = agent._retrieve_local(
                question=question,
                namespace=session_rag_namespace,
                strategy=strategy,
                k=k,
                score_threshold=score_threshold,
                session_ingest_ids=session_ingest_ids or None,
                paper_intent_hint=paper_intent_effective,
            )
            return agent.store.expand_neighbor_chunks(
                retrieved=loc,
                namespace=session_rag_namespace,
                window=1,
            )

        def _paper_retrieval_bundle() -> list[Any]:
            ph = agent._retrieve_local(
                question=question,
                namespace=read_paper_merge_ns,
                strategy=strategy,
                k=max(k, 8),
                score_threshold=score_threshold,
                session_ingest_ids=None,
                paper_intent_hint=True,
            )
            return agent.store.expand_neighbor_chunks(
                retrieved=ph,
                namespace=read_paper_merge_ns,
                window=1,
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_s = ex.submit(_session_retrieval_bundle)
            fut_p = ex.submit(_paper_retrieval_bundle)
            sess_hits = fut_s.result()
            paper_hits_par = fut_p.result()
        retrieved = merge_ranked_lists([list(sess_hits), list(paper_hits_par)])
        retrieved = retrieved[: max(k * 2, 12)]
    elif use_multi_source:
        retrieved = agent._retrieve_multi_source(
            question=question,
            namespace=session_rag_namespace,
            k=k,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids or None,
        )
    else:
        retrieved = agent._retrieve_local(
            question=question,
            namespace=session_rag_namespace,
            strategy=strategy,
            k=k,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids or None,
            paper_intent_hint=paper_intent_effective,
        )
        retrieved = agent.store.expand_neighbor_chunks(
            retrieved=retrieved,
            namespace=session_rag_namespace,
            window=1,
        )

    if (
        (not retrieved)
        and session_id
        and (
            agent._is_paper_coref_query(question)
            or agent._looks_like_paper_content_qa_not_ingest(question)
        )
        and not agent._is_download_or_ingest_intent(question)
        and not agent._is_delete_intent(question)
    ):
        recent = conversation_manager.get_recent_messages(session_id)
        fallback_ns = agent._extract_last_namespace_from_history(recent)
        if (
            fallback_ns
            and str(fallback_ns).startswith("paper:")
            and fallback_ns != session_rag_namespace
        ):
            read_paper_merge_ns = read_paper_merge_ns or str(fallback_ns)
            paper_hits_fb = agent._retrieve_local(
                question=question,
                namespace=fallback_ns,
                strategy=strategy,
                k=max(k, 8),
                score_threshold=score_threshold,
                session_ingest_ids=None,
                paper_intent_hint=True,
            )
            paper_hits_fb = agent.store.expand_neighbor_chunks(
                retrieved=paper_hits_fb,
                namespace=fallback_ns,
                window=1,
            )
            retrieved = merge_ranked_lists([list(retrieved), list(paper_hits_fb)])
            retrieved = retrieved[: max(k * 2, 12)]

    grouped = agent._group_by_parent(retrieved)

    # 二次兜底：若论文问答在首轮仍无命中，且问题里带 arXiv id，则仅在目标 paper namespace 再检一次。
    if (
        (not grouped)
        and paper_intent_effective
        and aid_in_question
        and read_paper_merge_ns
        and str(read_paper_merge_ns).startswith("paper:")
    ):
        try:
            retry_hits = agent._retrieve_local(
                question=question,
                namespace=read_paper_merge_ns,
                strategy=strategy,
                k=max(int(k or 4), 16),
                score_threshold=min(float(score_threshold or 0.5), 0.12),
                session_ingest_ids=None,
                paper_intent_hint=True,
            )
            retry_hits = agent.store.expand_neighbor_chunks(
                retrieved=retry_hits,
                namespace=read_paper_merge_ns,
                window=2,
            )
            if retry_hits:
                retrieved = list(retry_hits)[: max(int(k or 4) * 3, 24)]
                grouped = agent._group_by_parent(retrieved)
            trace_event(
                "paper_arxiv_id_retry_retrieve",
                {
                    "arxiv_id": str(aid_in_question),
                    "namespace": str(read_paper_merge_ns),
                    "retry_hits": len(retry_hits or []),
                    "grouped_after": len(grouped or []),
                },
            )
        except Exception as e:
            trace_event("paper_arxiv_id_retry_retrieve_error", {"error": str(e)[:300]})

    if (
        paper_intent_effective
        and read_paper_merge_ns
        and str(read_paper_merge_ns).startswith("paper:")
        and agent._paper_context_is_thin(grouped)
    ):
        try:
            from tools.rag.language import expand_retrieval_queries

            body_queries = expand_retrieval_queries(
                question,
                strategy=strategy,
                llm=agent.llm,
            )
            if not body_queries:
                qq = (question or "").strip()
                body_queries = [qq] if qq else []
            body_hits = agent.store.retrieve(
                body_queries,
                namespace=read_paper_merge_ns,
                k=max(int(k or 4), 16),
                score_threshold=min(float(score_threshold or 0.5), 0.25),
                strategy="hybrid_rerank",
                session_ingest_ids=None,
                extra_chroma_filter={"chunk_role": "generic"},
            )
            body_hits = agent.store.expand_neighbor_chunks(
                retrieved=body_hits,
                namespace=read_paper_merge_ns,
                window=2,
            )
            if body_hits:
                retrieved = merge_ranked_lists([list(retrieved), list(body_hits)])
                retrieved = retrieved[: max(int(k or 4) * 3, 24)]
                grouped = agent._group_by_parent(retrieved)
            trace_event(
                "paper_body_boost",
                {
                    "namespace": read_paper_merge_ns,
                    "thin_before": True,
                    "body_hits": len(body_hits or []),
                    "grouped_after": len(grouped or []),
                    "langgraph": True,
                },
            )
        except Exception as e:
            trace_event("paper_body_boost_error", {"error": str(e)[:300]})

    if (
        paper_intent_effective
        and read_paper_merge_ns
        and str(read_paper_merge_ns).startswith("paper:")
        and agent._question_wants_method(question)
        and (not agent._has_method_evidence(grouped))
    ):
        try:
            method_queries = [
                question,
                "method approach architecture model design",
                "section 3 method approach",
                "方法 模型结构 技术路线",
            ]
            method_hits = agent.store.retrieve(
                method_queries,
                namespace=read_paper_merge_ns,
                k=max(int(k or 4), 26),
                score_threshold=min(float(score_threshold or 0.5), 0.05),
                strategy="hybrid_rerank",
                session_ingest_ids=None,
                extra_chroma_filter={"chunk_role": "generic"},
            )
            method_hits = agent.store.expand_neighbor_chunks(
                retrieved=method_hits,
                namespace=read_paper_merge_ns,
                window=2,
            )
            if method_hits:
                retrieved = merge_ranked_lists([list(retrieved), list(method_hits)])
                retrieved = retrieved[: max(int(k or 4) * 3, 24)]
                grouped = agent._group_by_parent(retrieved)
            trace_event(
                "paper_method_fallback",
                {
                    "question": question[:220],
                    "namespace": read_paper_merge_ns,
                    "method_hits": len(method_hits or []),
                    "grouped_after": len(grouped or []),
                    "langgraph": True,
                },
            )
        except Exception as e:
            trace_event("paper_method_fallback_error", {"error": str(e)[:300]})

    paper_ns = ""
    if read_paper_merge_ns and str(read_paper_merge_ns).startswith("paper:"):
        paper_ns = str(read_paper_merge_ns)
    elif str(session_rag_namespace).startswith("paper:"):
        paper_ns = str(session_rag_namespace)
    try:
        from tools.retrieval.method_section_context import prepend_method_section_full_context

        grouped = prepend_method_section_full_context(
            list(grouped),
            namespace=paper_ns,
            question=question,
        )
    except Exception:
        pass

    trace_event(
        "langgraph_rag_local_retrieve_done",
        {
            "retrieved_count": len(retrieved or []),
            "grouped_count": len(grouped or []),
            "read_paper_merge_ns": str(read_paper_merge_ns or ""),
        },
    )
    return {
        "question": question,
        "namespace": (str(read_paper_merge_ns) if str(read_paper_merge_ns or "").startswith("paper:") else state.get("namespace")),
        "grouped": grouped,
        "grouped_for_prompt": grouped,
        "grouped_for_citations": grouped,
        "web_fallback_used": False,
        "web_supplement_used": False,
        "web_forced_by_user": False,
        "retrieval_context_score": None,
        "retrieval_judge_reason": None,
    }


def _rag_image_relevance_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "rag_image_relevance"})
    if state.get("result"):
        return {}
    grouped = list(state.get("grouped") or [])
    user_image_paths = state.get("user_image_paths")
    image_relevant_to_db = True
    if user_image_paths:
        try:
            image_relevant_to_db = agent._judge_use_user_images(
                question=str(state.get("question") or ""),
                grouped=grouped,
                user_image_paths=user_image_paths,
            )
        except Exception:
            image_relevant_to_db = True
    image_only_mode = bool(user_image_paths) and (not image_relevant_to_db)
    include_hist = bool(user_image_paths) and image_relevant_to_db
    return {
        "image_only_mode": image_only_mode,
        "include_user_images_in_history": include_hist,
        "grouped_for_prompt": ([] if image_only_mode else grouped),
        "grouped_for_citations": ([] if image_only_mode else grouped),
    }


def _rag_paper_fallback_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "rag_paper_fallback"})
    if state.get("result"):
        return {}
    paper_intent = bool(state.get("paper_intent"))
    grouped = list(state.get("grouped") or [])
    image_only_mode = bool(state.get("image_only_mode"))
    question = str(state.get("question") or "")
    ns = str(state.get("namespace") or "").strip()
    has_arxiv_in_q = bool(extract_arxiv_id(question))
    in_paper_ns = ns.startswith("paper:")
    content_qa = False
    try:
        from tools.retrieval.local_paper_qa_resolver import looks_like_paper_content_qa

        content_qa = bool(looks_like_paper_content_qa(question))
    except Exception:
        content_qa = False

    # 内容问答（尤其带 arXiv id / paper: namespace）在无本地命中时不应自动降级到在线 arXiv 搜索。
    if paper_intent and (not grouped) and content_qa and (has_arxiv_in_q or in_paper_ns):
        trace_event(
            "rag_paper_fallback_blocked_for_content_qa",
            {
                "has_arxiv_in_question": bool(has_arxiv_in_q),
                "in_paper_namespace": bool(in_paper_ns),
                "question": question[:220],
            },
        )
        return {}

    if paper_intent and (not grouped) and (not image_only_mode):
        from tools.agent.agent_tools import tool_search_arxiv

        answer_text = str(
            tool_search_arxiv.invoke(
                {
                    "query": str(state.get("question") or ""),
                    "max_results": max(2, int(state.get("k") or 2)),
                }
            )
        )
        return {"result": {"answer": answer_text, "citations": []}}
    return {}


def _rag_judge_context_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "rag_judge_context"})
    if state.get("result"):
        return {}
    question = str(state.get("question") or "")
    route = state.get("route")
    grouped = list(state.get("grouped_for_prompt") or [])
    image_only_mode = bool(state.get("image_only_mode"))
    paper_intent = bool(state.get("paper_intent"))
    if image_only_mode:
        return {
            "need_web_supplement": False,
            "paper_glossary_web_supplement": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": None,
        }

    from config import (
        RAG_CONTEXT_SCORE_MIN,
        RAG_LLM_CONTEXT_SCORE_MODE,
        RAG_PAPER_GLOSSARY_WEB_ENABLED,
    )
    from tools.retrieval.glossary_escalation import paper_glossary_should_supplement_web
    from tools.retrieval.query_router import build_query_route

    decision = build_query_route(question)
    in_paper_ns = str(state.get("namespace") or "").strip().startswith("paper:")
    if in_paper_ns and bool(getattr(decision, "needs_table", False)):
        trace_event(
            "paper_table_qa_no_web",
            {
                "namespace": str(state.get("namespace") or "")[:120],
                "question": question[:220],
            },
        )
        return {
            "need_web_supplement": False,
            "paper_glossary_web_supplement": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": "paper_namespace_table_local_only",
        }
    glossary_scope = decision.intent == "paper_qa" or (
        in_paper_ns
        and decision.intent
        not in ("paper_search", "local_paper_search", "tool_task")
    )
    if (
        grouped
        and RAG_PAPER_GLOSSARY_WEB_ENABLED
        and glossary_scope
        and (paper_intent or in_paper_ns)
    ):
        gs = paper_glossary_should_supplement_web(question, grouped)
        if gs.need_supplement:
            trace_event(
                "paper_glossary_web_escalation",
                {
                    "reason": gs.reason[:500],
                    "intent": decision.intent,
                    "in_paper_ns": in_paper_ns,
                },
            )
            return {
                "need_web_supplement": True,
                "paper_glossary_web_supplement": True,
                "retrieval_context_score": None,
                "retrieval_judge_reason": gs.reason,
            }
        return {
            "need_web_supplement": False,
            "paper_glossary_web_supplement": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": None,
        }

    if paper_intent or (not allow_web_search_when_local_misses(route)):
        return {
            "need_web_supplement": False,
            "paper_glossary_web_supplement": False,
            "retrieval_context_score": None,
            "retrieval_judge_reason": None,
        }

    need_web = not bool(grouped)
    retrieval_context_score = None
    retrieval_judge_reason = None
    if grouped and RAG_LLM_CONTEXT_SCORE_MODE != "off":
        from tools.rag.retrieval_judge import judge_retrieval_context

        verdict = judge_retrieval_context(
            agent.llm,
            question,
            grouped,
            score_min=RAG_CONTEXT_SCORE_MIN,
        )
        retrieval_context_score = float(verdict.get("score", 0.0))
        retrieval_judge_reason = str(verdict.get("reason") or "")
        need_web = bool(verdict.get("should_supplement_web")) or (
            retrieval_context_score < float(RAG_CONTEXT_SCORE_MIN)
        )
    return {
        "need_web_supplement": bool(need_web),
        "paper_glossary_web_supplement": False,
        "retrieval_context_score": retrieval_context_score,
        "retrieval_judge_reason": retrieval_judge_reason,
    }


def _rag_web_retrieve_merge_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "rag_web_retrieve_merge"})
    if state.get("result"):
        return {}
    if not bool(state.get("need_web_supplement")):
        return {}
    from config import RAG_WEB_FALLBACK_ENABLED, RAG_WEB_MERGED_MAX_RESULTS
    from tools.agent.web_search import search_web_with_subquestions, web_items_to_document_pairs

    if not RAG_WEB_FALLBACK_ENABLED:
        return {}
    question = str(state.get("question") or "")
    grouped = list(state.get("grouped_for_prompt") or [])
    web_items, _web_note = search_web_with_subquestions(
        question,
        agent.llm,
        max_merged_results=RAG_WEB_MERGED_MAX_RESULTS,
    )
    if not web_items:
        return {}
    web_pairs = web_items_to_document_pairs(
        web_items,
        score_base=0.1,
        meta_type="web_search_nonpaper",
    )
    web_grouped = agent._group_by_parent(web_pairs)
    if grouped:
        merged = list(grouped) + list(web_grouped)
        return {
            "grouped_for_prompt": merged,
            "grouped_for_citations": merged,
            "web_supplement_used": True,
        }
    return {
        "grouped_for_prompt": web_grouped,
        "grouped_for_citations": web_grouped,
        "web_fallback_used": True,
    }


def _rag_final_generation_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "rag_final_generation"})
    if state.get("result"):
        return {}
    question = str(state.get("question") or "")
    session_id = state.get("session_id")
    grouped_for_prompt = list(state.get("grouped_for_prompt") or [])
    grouped_for_citations = list(state.get("grouped_for_citations") or [])
    user_image_paths = state.get("user_image_paths")
    include_hist = bool(state.get("include_user_images_in_history"))

    chat_history = (
        conversation_manager.get_recent_messages(session_id) if session_id else None
    )
    messages = build_rag_prompt(
        question,
        grouped_for_prompt,
        chat_history=chat_history,
        user_image_paths=user_image_paths,
        include_user_images_in_history=include_hist,
        paper_glossary_web_supplement=bool(
            state.get("paper_glossary_web_supplement")
        ),
        paper_scope_arxiv_id=paper_namespace_arxiv_id(
            str(state.get("namespace") or "")
        ),
    )
    run_before_model(agent.middleware, messages)
    resp = agent.llm.invoke(dict_messages_to_lc(messages))
    run_after_model(agent.middleware, resp)
    answer_text = resp.content if hasattr(resp, "content") else str(resp)
    answer_text = agent._with_truncation_reason(resp, str(answer_text))

    if session_id:
        conversation_manager.add_turn(
            session_id, "user", question, image_paths=user_image_paths
        )
        conversation_manager.add_turn(session_id, "assistant", answer_text)

    citations: list[dict[str, Any]] = []
    for doc, score in grouped_for_citations:
        meta = getattr(doc, "metadata", {}) or {}
        citations.append(
            {
                "source": meta.get("source", "unknown"),
                "score": float(score),
                "preview": str(getattr(doc, "page_content", "") or "")[:200],
            }
        )
    return {
        "result": {
            "answer": answer_text,
            "citations": citations,
            "web_fallback": bool(state.get("web_fallback_used")),
            "web_supplement": bool(state.get("web_supplement_used")),
            "web_forced_by_user": bool(state.get("web_forced_by_user")),
            "retrieval_context_score": state.get("retrieval_context_score"),
            "retrieval_judge_reason": state.get("retrieval_judge_reason"),
        }
    }


def _weather_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "weather_answer"})
    result = agent.answer(
        question=str(state.get("question") or ""),
        namespace=str(state.get("namespace") or ""),
        strategy=str(state.get("strategy") or DEFAULT_RETRIEVAL_STRATEGY),
        k=int(state.get("k") or 4),
        score_threshold=float(state.get("score_threshold") or 0.5),
        use_multi_source=bool(state.get("use_multi_source")),
        session_id=state.get("session_id"),
        user_image_paths=state.get("user_image_paths"),
        preclassified_route=Route.WEATHER,
        preclassified_paper_intent=bool(state.get("paper_intent")),
        skip_special_branches=False,
    )
    return {"result": result}


def _history_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "history_answer"})
    result = agent.answer(
        question=str(state.get("question") or ""),
        namespace=str(state.get("namespace") or ""),
        strategy=str(state.get("strategy") or DEFAULT_RETRIEVAL_STRATEGY),
        k=int(state.get("k") or 4),
        score_threshold=float(state.get("score_threshold") or 0.5),
        use_multi_source=bool(state.get("use_multi_source")),
        session_id=state.get("session_id"),
        user_image_paths=state.get("user_image_paths"),
        preclassified_route=Route.RAG,
        preclassified_paper_intent=bool(state.get("paper_intent")),
        skip_special_branches=False,
    )
    return {"result": result}


def _nonpaper_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event("langgraph_node_enter", {"node": "nonpaper_answer"})
    result = agent.answer(
        question=str(state.get("question") or ""),
        namespace=str(state.get("namespace") or ""),
        strategy=str(state.get("strategy") or DEFAULT_RETRIEVAL_STRATEGY),
        k=int(state.get("k") or 4),
        score_threshold=float(state.get("score_threshold") or 0.5),
        use_multi_source=bool(state.get("use_multi_source")),
        session_id=state.get("session_id"),
        user_image_paths=state.get("user_image_paths"),
        preclassified_route=Route.RAG,
        preclassified_paper_intent=False,
        skip_special_branches=False,
    )
    return {"result": result}


def _tools_node(state: ChatGraphState, agent: Any) -> ChatGraphState:
    trace_event(
        "langgraph_node_enter",
        {
            "node": "tools_answer",
            "k": state.get("k"),
            "prefetch_subq_web": state.get("prefetch_subq_web"),
        },
    )
    result = agent.answer_with_tools(
        question=str(state.get("question") or ""),
        namespace=str(state.get("namespace") or ""),
        k=int(state.get("k") or 4),
        session_id=state.get("session_id"),
        user_image_paths=state.get("user_image_paths"),
        prefetch_subq_web=state.get("prefetch_subq_web"),
    )
    return {"result": result}


def execute_chat_with_langgraph(
    *,
    agent: Any,
    question: str,
    namespace: str,
    strategy: str,
    k: int,
    score_threshold: float,
    use_multi_source: bool,
    session_id: str | None,
    user_image_paths: list[str] | None,
    use_tools: bool,
    prefetch_subq_web: bool | None,
) -> dict[str, Any]:
    """使用 LangGraph 编排 chat 主流程：route -> (rag_answer | tools_answer)。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:
        trace_event("langgraph_unavailable_fallback", {"error": str(exc)})
        # 兜底：保持旧行为
        if use_tools and not user_image_paths:
            return agent.answer_with_tools(
                question=question,
                namespace=namespace,
                k=k,
                session_id=session_id,
                user_image_paths=user_image_paths,
                prefetch_subq_web=prefetch_subq_web,
            )
        return agent.answer(
            question=question,
            namespace=namespace,
            strategy=strategy,
            k=k,
            score_threshold=score_threshold,
            use_multi_source=use_multi_source,
            session_id=session_id,
            user_image_paths=user_image_paths,
        )

    graph = StateGraph(ChatGraphState)
    graph.add_node("route", _route_node)
    graph.add_node(
        "local_library_list", lambda s: _local_library_list_node(s, agent)
    )
    graph.add_node("rag_local_retrieve", lambda s: _rag_local_retrieve_node(s, agent))
    graph.add_node("rag_image_relevance", lambda s: _rag_image_relevance_node(s, agent))
    graph.add_node("rag_paper_fallback", lambda s: _rag_paper_fallback_node(s, agent))
    graph.add_node("rag_judge_context", lambda s: _rag_judge_context_node(s, agent))
    graph.add_node("rag_web_retrieve_merge", lambda s: _rag_web_retrieve_merge_node(s, agent))
    graph.add_node("rag_final_generation", lambda s: _rag_final_generation_node(s, agent))
    graph.add_node("tools_answer", lambda s: _tools_node(s, agent))
    graph.add_node("weather_answer", lambda s: _weather_node(s, agent))
    graph.add_node("history_answer", lambda s: _history_node(s, agent))
    graph.add_node("nonpaper_answer", lambda s: _nonpaper_node(s, agent))
    graph.add_edge(START, "route")

    def _pick_path(s: ChatGraphState) -> str:
        path = str(s.get("selected_path") or "rag")
        if path == "tools":
            return "tools_answer"
        if path == "local_library_list":
            return "local_library_list"
        if path == "weather":
            return "weather_answer"
        if path == "history":
            return "history_answer"
        if path == "nonpaper":
            return "nonpaper_answer"
        return "rag_local_retrieve"

    graph.add_conditional_edges(
        "route",
        _pick_path,
        {
            "tools_answer": "tools_answer",
            "local_library_list": "local_library_list",
            "weather_answer": "weather_answer",
            "history_answer": "history_answer",
            "nonpaper_answer": "nonpaper_answer",
            "rag_local_retrieve": "rag_local_retrieve",
        },
    )
    graph.add_edge("rag_local_retrieve", "rag_image_relevance")
    graph.add_edge("rag_image_relevance", "rag_paper_fallback")

    def _paper_fallback_or_continue(s: ChatGraphState) -> str:
        return "end" if bool(s.get("result")) else "continue"

    graph.add_conditional_edges(
        "rag_paper_fallback",
        _paper_fallback_or_continue,
        {"end": END, "continue": "rag_judge_context"},
    )
    graph.add_edge("rag_judge_context", "rag_web_retrieve_merge")
    graph.add_edge("rag_web_retrieve_merge", "rag_final_generation")
    graph.add_edge("rag_final_generation", END)
    graph.add_edge("tools_answer", END)
    graph.add_edge("local_library_list", END)
    graph.add_edge("weather_answer", END)
    graph.add_edge("history_answer", END)
    graph.add_edge("nonpaper_answer", END)
    app = graph.compile()

    initial: ChatGraphState = {
        "question": question,
        "namespace": namespace,
        "strategy": strategy,
        "k": k,
        "score_threshold": score_threshold,
        "use_multi_source": use_multi_source,
        "session_id": session_id,
        "user_image_paths": user_image_paths,
        "use_tools": use_tools,
        "prefetch_subq_web": prefetch_subq_web,
        "_agent": agent,
    }
    trace_event("langgraph_start", {"question": question[:200], "namespace": namespace})
    final_state = app.invoke(initial)
    trace_event("langgraph_end", {"has_result": bool(final_state.get("result"))})
    return dict(final_state.get("result") or {})

