"""
向量存储与检索：按 namespace 使用 Chroma 持久化，支持多策略检索与 BM25 混合。
"""
from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from models_qwen import qwen_embeddings

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    HYBRID_BM25_TOP_K,
    HYBRID_SEMANTIC_TOP_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RERANK_TOP_K,
    RAG_SIMILARITY_QUERY_MAX_WORKERS,
)
from tools.rag.bm25 import bm25_top_k, build_bm25_index, rrf_fusion
from tools.rag.math_utils import apply_score_threshold, merge_ranked_lists
from tools.rag.rerank import rerank as rerank_docs
from tools.rag.time_utils import add_timestamp_metadata

# Chroma 持久化目录；每个 namespace 对应一个 collection
CHROMA_PERSIST_DIR = "data/chroma"


def _sanitize_collection_name(namespace: str) -> str:
    """Chroma collection 名只允许字母数字、下划线、横线。"""
    return re.sub(r"[^\w\-]", "_", namespace).strip("_") or "default"


try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None  # type: ignore[misc, assignment]


@dataclass
class NamespaceVectorStore:
    """按 namespace 使用 Chroma 的向量存储，持久化到本地目录。

    - 每个 namespace 对应 Chroma 的一个 collection，同一 persist_directory 下多 collection
    - 使用 Qwen 兼容 Embeddings（text-embedding-v3）生成向量
    - 支持多策略检索、BM25 混合、重排序
    """

    embeddings: OpenAIEmbeddings
    persist_directory: str = CHROMA_PERSIST_DIR
    stores: Dict[str, object] = field(default_factory=dict)  # namespace 对应的 Chroma
    _bm25_cache: Dict[str, tuple] = field(default_factory=dict, repr=False)

    def _invalidate_bm25(self, namespace: str) -> None:
        self._bm25_cache.pop(namespace, None)

    def export_documents(self, namespace: str) -> List[Document]:
        """导出某 namespace 在 Chroma 中的全部文档（只读），用于会话语义镜像等到其他 namespace。

        若当前进程尚未加载该 collection，会按需打开持久化目录中的对应 collection。
        """
        if Chroma is None:
            return []
        try:
            self._get_or_create_store(namespace)
        except Exception:
            return []
        raw = self._get_docs_for_namespace(namespace)
        out: List[Document] = []
        for d in raw:
            if isinstance(d, Document):
                out.append(
                    Document(
                        page_content=d.page_content or "",
                        metadata=dict(d.metadata or {}),
                    )
                )
        return out

    def _get_docs_for_namespace(self, namespace: str) -> List[object]:
        """从 Chroma collection 中取出该 namespace 下全部文档（用于 BM25）。"""
        store = self.stores.get(namespace)
        if store is None:
            return []
        coll = getattr(store, "_collection", None)
        if coll is None:
            return []
        try:
            data = coll.get(include=["documents", "metadatas"])
            docs = []
            for i, doc_text in enumerate(data.get("documents") or []):
                meta = (data.get("metadatas") or [{}])[i] if data.get("metadatas") else {}
                docs.append(Document(page_content=doc_text, metadata=meta))
            return docs
        except Exception:
            return []

    def _get_bm25_for_namespace(self, namespace: str):
        if namespace in self._bm25_cache:
            return self._bm25_cache[namespace]
        docs = self._get_docs_for_namespace(namespace)
        bm25, doc_list = build_bm25_index(docs)
        if bm25 is not None:
            self._bm25_cache[namespace] = (bm25, doc_list)
        return (bm25, doc_list)

    def _get_or_create_store(self, namespace: str) -> object:
        if Chroma is None:
            raise RuntimeError("请安装 chromadb 与 langchain-chroma: pip install chromadb langchain-chroma")
        if namespace not in self.stores:
            name = _sanitize_collection_name(namespace)
            try:
                self.stores[namespace] = Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            except BaseException as e:
                # 某些环境下 chromadb Rust 后端会因历史损坏索引触发 PanicException（非普通 Exception）。
                # 遇到该类错误时隔离旧目录并重建，避免服务直接起不来。
                msg = str(e)
                if "pyo3_runtime.PanicException" not in msg and "range start index" not in msg:
                    raise
                old_dir = Path(self.persist_directory)
                if old_dir.exists():
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    bak = old_dir.parent / f"{old_dir.name}.corrupt-{ts}"
                    try:
                        shutil.move(str(old_dir), str(bak))
                    except OSError:
                        # 移动失败时尝试删除，确保后续可重建
                        shutil.rmtree(old_dir, ignore_errors=True)
                old_dir.mkdir(parents=True, exist_ok=True)
                self.stores[namespace] = Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
        return self.stores[namespace]

    def delete_by_where(self, namespace: str, where: dict) -> None:
        """按 metadata 条件删除条目（用于会话内论文镜像去重等）。collection 不存在时忽略。"""
        if Chroma is None or not where:
            return
        try:
            self._get_or_create_store(namespace)
        except Exception:
            return
        store = self.stores.get(namespace)
        coll = getattr(store, "_collection", None) if store is not None else None
        if coll is None:
            return
        try:
            coll.delete(where=where)
        except Exception:
            pass
        self._invalidate_bm25(namespace)

    def clear_namespace(self, namespace: str) -> None:
        """清空指定 namespace 对应 collection（用于重建索引避免重复 chunks）。"""
        if Chroma is None:
            raise RuntimeError("请安装 chromadb 与 langchain-chroma: pip install chromadb langchain-chroma")
        name = _sanitize_collection_name(namespace)
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=name)
        except Exception:
            # collection 不存在等场景，按空操作处理
            pass
        self.stores.pop(namespace, None)
        self._invalidate_bm25(namespace)

    def add_text(
        self,
        text: str,
        namespace: str = DEFAULT_NAMESPACE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="gpt-3.5-turbo",
        )
        chunks = splitter.split_text(text)
        metadata = extra_metadata or {}
        store = self._get_or_create_store(namespace)
        store.add_texts(
            texts=chunks,
            metadatas=[{"namespace": namespace, **metadata} for _ in chunks],
        )
        self._invalidate_bm25(namespace)
        return len(chunks)

    def add_documents(
        self,
        documents: Sequence[Document],
        namespace: str = DEFAULT_NAMESPACE,
        extra_metadata: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """直接写入已分块文档（如 PyMuPDFLoader + splitter 的输出）。

        ``ids`` 与 ``documents`` 等长时传入 Chroma，便于与 PostgreSQL ``chroma_doc_id`` 对齐。
        """
        if not documents:
            return 0
        metadata = extra_metadata or {}
        store = self._get_or_create_store(namespace)
        docs: list[Document] = []
        for d in documents:
            md = {"namespace": namespace, **metadata, **dict(d.metadata or {})}
            docs.append(Document(page_content=d.page_content, metadata=md))
        if ids is not None:
            if len(ids) != len(docs):
                raise ValueError("ids length must match documents length")
            store.add_documents(docs, ids=ids)
        else:
            store.add_documents(docs)
        self._invalidate_bm25(namespace)
        return len(docs)

    def add_chunked_text_with_prefixed_ids(
        self,
        text: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        id_prefix: str,
        extra_metadata: Optional[dict] = None,
    ) -> List[Tuple[str, str]]:
        """分块写入 Chroma，id 为 ``{id_prefix}_{i:06d}``；返回 ``[(chroma_doc_id, chunk_text), ...]`` 供 PostgreSQL 对齐。"""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="gpt-3.5-turbo",
        )
        chunks = splitter.split_text(text)
        if not chunks:
            return []
        prefix = (id_prefix or "chunk").strip() or "chunk"
        ids = [f"{prefix}_{i:06d}" for i in range(len(chunks))]
        base = add_timestamp_metadata(dict(extra_metadata or {}))
        store = self._get_or_create_store(namespace)
        metadatas = [
            {"namespace": namespace, **base, "chunk_index": i} for i in range(len(chunks))
        ]
        store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        self._invalidate_bm25(namespace)
        return list(zip(ids, chunks))

    def _similarity_search_with_score_single(
        self,
        query: str,
        namespace: str,
        k: int,
        chroma_filter: Optional[dict[str, Any]] = None,
    ) -> List[Tuple[object, float]]:
        # Chroma 的 collection 是持久化在磁盘的，但 `self.stores` 只在首次访问时创建。
        # 之前的实现如果 `namespace` 尚未被加载到 `self.stores`，会直接返回空列表。
        # 这里改为按需创建 store，从而读取持久化数据。
        store = self.stores.get(namespace)
        if store is None:
            try:
                store = self._get_or_create_store(namespace)
            except Exception:
                return []
        kwargs: dict[str, Any] = {"k": k}
        if chroma_filter:
            kwargs["filter"] = chroma_filter
        return store.similarity_search_with_score(query, **kwargs)

    def similarity_search_with_score_multi(
        self,
        queries: Iterable[str],
        namespace: str = DEFAULT_NAMESPACE,
        k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        chroma_filter: Optional[dict[str, Any]] = None,
    ) -> List[Tuple[object, float]]:
        query_list = list(queries)
        if len(query_list) <= 1:
            per_query_results: list[Sequence[Tuple[object, float]]] = [
                self._similarity_search_with_score_single(q, namespace, k, chroma_filter)
                for q in query_list
            ]
        else:
            max_w = max(1, min(RAG_SIMILARITY_QUERY_MAX_WORKERS, len(query_list)))
            with ThreadPoolExecutor(max_workers=max_w) as ex:
                futs = [
                    ex.submit(
                        self._similarity_search_with_score_single,
                        q,
                        namespace,
                        k,
                        chroma_filter,
                    )
                    for q in query_list
                ]
                per_query_results = [f.result() for f in futs]
        merged = merge_ranked_lists(per_query_results)
        filtered = apply_score_threshold(merged, score_threshold)
        return filtered[:k]

    def embed_document(
        self,
        text: str,
        namespace: str = DEFAULT_NAMESPACE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        extra_metadata = add_timestamp_metadata(extra_metadata)
        n_chunks = self.add_text(
            text=text,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=extra_metadata,
        )
        # Chroma 自动持久化，无需显式保存
        return n_chunks

    def _merge_chroma_filters(
        self,
        base: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not base and not extra:
            return None
        if not base:
            return dict(extra) if extra else None
        if not extra:
            return dict(base)
        # Chroma：多条件用 $and 更稳
        return {"$and": [dict(base), dict(extra)]}

    def retrieve(
        self,
        queries: Iterable[str],
        namespace: str = DEFAULT_NAMESPACE,
        k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
        session_ingest_ids: Optional[List[str]] = None,
        extra_chroma_filter: Optional[dict[str, Any]] = None,
    ) -> List[Tuple[object, float]]:
        """检索文档；若提供 session_ingest_ids，则先仅在这些会话上传块内检索，无命中再全分区检索。

        ``extra_chroma_filter`` 与 session 过滤合并，用于按 arXiv id、chunk_role 等窄化候选（PAPER_RETRIEVAL_OPTIMIZATION_PLAN）。
        """
        raw_ids = [str(x).strip() for x in (session_ingest_ids or []) if str(x).strip()]
        chroma_where: dict[str, Any] | None = None
        if raw_ids:
            chroma_where = (
                {"session_ingest_id": raw_ids[0]}
                if len(raw_ids) == 1
                else {"session_ingest_id": {"$in": raw_ids}}
            )
            chroma_where = self._merge_chroma_filters(chroma_where, extra_chroma_filter)
            filtered = self._retrieve_impl(
                queries=queries,
                namespace=namespace,
                k=k,
                score_threshold=score_threshold,
                strategy=strategy,
                chroma_filter=chroma_where,
                restrict_to_filter=True,
            )
            if filtered:
                return filtered

        return self._retrieve_impl(
            queries=queries,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            chroma_filter=extra_chroma_filter,
            restrict_to_filter=False,
        )

    def _retrieve_impl(
        self,
        *,
        queries: Iterable[str],
        namespace: str,
        k: int,
        score_threshold: float,
        strategy: str,
        chroma_filter: dict[str, Any] | None,
        restrict_to_filter: bool,
    ) -> List[Tuple[object, float]]:
        # 兼容：前端/CLI 可能传入 strategy="default"（语义上期望“向量+BM25+可选重排”）
        if restrict_to_filter and chroma_filter is None:
            return []
        if strategy == "default":
            strategy = "hybrid_rerank"

        query_list = list(queries)
        first_query = query_list[0] if query_list else ""

        if strategy == "hybrid" or strategy == "hybrid_rerank":
            semantic = self.similarity_search_with_score_multi(
                queries=query_list,
                namespace=namespace,
                k=HYBRID_SEMANTIC_TOP_K,
                score_threshold=score_threshold,
                chroma_filter=chroma_filter,
            )
            if chroma_filter is not None:
                merged = semantic[:k]
            else:
                bm25, doc_list = self._get_bm25_for_namespace(namespace)
                if bm25 is not None and doc_list:
                    bm25_results = bm25_top_k(bm25, doc_list, first_query, k=HYBRID_BM25_TOP_K)
                    merged = rrf_fusion([semantic, bm25_results])[:k]
                else:
                    merged = semantic[:k]
            if strategy == "hybrid_rerank":
                merged = rerank_docs(first_query, merged, top_k=min(k, RERANK_TOP_K))
            return merged

        if strategy == "rerank":
            base = self.similarity_search_with_score_multi(
                queries=query_list,
                namespace=namespace,
                k=max(k, RERANK_TOP_K * 2),
                score_threshold=score_threshold,
                chroma_filter=chroma_filter,
            )
            return rerank_docs(first_query, base, top_k=k)

        return self.similarity_search_with_score_multi(
            queries=query_list,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            chroma_filter=chroma_filter,
        )

    def expand_neighbor_chunks(
        self,
        retrieved: Sequence[Tuple[object, float]],
        namespace: str = DEFAULT_NAMESPACE,
        window: int = 1,
    ) -> List[Tuple[object, float]]:
        """按 parent_id/chunk_index 补充邻接 chunk，提升上下文连续性。"""
        if window <= 0 or not retrieved:
            return list(retrieved)
        store = self.stores.get(namespace)
        if store is None:
            return list(retrieved)
        coll = getattr(store, "_collection", None)
        if coll is None:
            return list(retrieved)
        try:
            data = coll.get(include=["documents", "metadatas"])
        except Exception:
            return list(retrieved)

        chunk_map: dict[str, dict[int, Document]] = defaultdict(dict)
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for i, doc_text in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            parent_id = (meta or {}).get("parent_id")
            chunk_index = (meta or {}).get("chunk_index")
            if parent_id is None or not isinstance(chunk_index, int):
                continue
            chunk_map[str(parent_id)][chunk_index] = Document(
                page_content=doc_text,
                metadata=dict(meta or {}),
            )

        picked: dict[tuple, Tuple[object, float]] = {}

        def _key_for(doc: object) -> tuple:
            md = getattr(doc, "metadata", {}) or {}
            pid = md.get("parent_id")
            idx = md.get("chunk_index")
            if pid is not None and isinstance(idx, int):
                return ("parent_chunk", str(pid), idx)
            return ("doc_obj", id(doc))

        for doc, score in retrieved:
            key = _key_for(doc)
            prev = picked.get(key)
            if prev is None or score < prev[1]:
                picked[key] = (doc, float(score))

            md = getattr(doc, "metadata", {}) or {}
            pid = md.get("parent_id")
            idx = md.get("chunk_index")
            if pid is None or not isinstance(idx, int):
                continue
            p = str(pid)
            for n_idx in range(idx - window, idx + window + 1):
                if n_idx == idx:
                    continue
                n_doc = chunk_map.get(p, {}).get(n_idx)
                if n_doc is None:
                    continue
                n_key = ("parent_chunk", p, n_idx)
                # 对邻居块给略差一点的分数，保留主命中优先级
                n_score = float(score) + (abs(n_idx - idx) * 1e-3)
                prev_n = picked.get(n_key)
                if prev_n is None or n_score < prev_n[1]:
                    picked[n_key] = (n_doc, n_score)

        return sorted(picked.values(), key=lambda x: x[1])

    # ---------- 持久化（Chroma 自动落盘，保存/加载仅兼容旧接口） ----------

    def save(self, base_dir: str = "data/vectorstores") -> None:
        """Chroma 已按 persist_directory 自动持久化，此方法保留为空实现以兼容调用方。"""
        pass

    def load(self, base_dir: str = "data/vectorstores") -> None:
        """Chroma 在首次访问某 collection 时自动从 persist_directory 加载；此处仅清 BM25 缓存。"""
        self._bm25_cache.clear()

    def list_collection_names(self) -> List[str]:
        """列出已存在的 Chroma collection 名（即已使用过的 namespace 的 sanitized 名）。"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            return [c.name for c in client.list_collections()]
        except Exception:
            return []


vector_store = NamespaceVectorStore(embeddings=qwen_embeddings, persist_directory=CHROMA_PERSIST_DIR)
