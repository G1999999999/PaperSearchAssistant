"""首次部署：根据 ORM 元数据创建表（未包含复杂迁移）。生产环境建议使用正式 migration 工具。"""

from __future__ import annotations


def create_tables_if_needed() -> bool:
    from tools.storage.sql.db import get_engine
    from tools.storage.sql.models import Base

    eng = get_engine()
    if eng is None:
        return False
    Base.metadata.create_all(eng)
    return True
