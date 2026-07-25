from __future__ import annotations

import os


class RedisKeys:
    """与 DATABASE_REDESIGN_PLAN.md / PAPER_RETRIEVAL_OPTIMIZATION_PLAN.md 一致的 key 前缀。"""

    def __init__(self, prefix: str | None = None) -> None:
        self.prefix = (prefix or os.getenv("REDIS_PREFIX", "psa2") or "psa2").strip()

    def chat_recent(self, session_id: str) -> str:
        return f"{self.prefix}:chat:recent:{self._slug(session_id)}"

    def paper_detail(self, paper_id: int | str) -> str:
        return f"{self.prefix}:paper:detail:{paper_id}"

    def paper_sections_tree(self, paper_id: int | str) -> str:
        return f"{self.prefix}:paper:sections:{paper_id}"

    def paper_section_summary(self, section_id: int | str) -> str:
        return f"{self.prefix}:paper:section_summary:{section_id}"

    def paper_section_roles(self, paper_id: int | str) -> str:
        return f"{self.prefix}:paper:section_roles:{paper_id}"

    def paper_summary_bundle(self, paper_id: int | str) -> str:
        return f"{self.prefix}:paper:summary_bundle:{paper_id}"

    def chunk_neighbors(self, chunk_id: int | str) -> str:
        return f"{self.prefix}:chunk:neighbors:{chunk_id}"

    def search_paper(self, query_hash: str) -> str:
        return f"{self.prefix}:search:paper:{query_hash}"

    def search_chat(self, query_hash: str) -> str:
        return f"{self.prefix}:search:chat:{query_hash}"

    @staticmethod
    def _slug(session_id: str) -> str:
        return (session_id or "default").replace("/", "_").replace("\\", "_")
