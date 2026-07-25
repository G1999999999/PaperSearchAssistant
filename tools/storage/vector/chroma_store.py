"""Chroma 向量入口：保持 Chroma 仅作向量召回，业务事实以 PostgreSQL 为准。"""

from tools.rag.knowledge import CHROMA_PERSIST_DIR, NamespaceVectorStore, vector_store

__all__ = ["CHROMA_PERSIST_DIR", "NamespaceVectorStore", "vector_store"]
