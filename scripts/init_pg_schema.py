#!/usr/bin/env python3
"""创建 PostgreSQL 表（需配置 DATABASE_URL）。也可用 psql 执行 tools/storage/sql/migrations/001_initial.sql。"""

from __future__ import annotations

import os
import sys

# 项目根目录作为 cwd
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env.runtime"))
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass


def main() -> int:
    from sqlalchemy.exc import OperationalError

    from tools.storage.sql.db import is_database_configured
    from tools.storage.sql.schema_init import create_tables_if_needed

    if not is_database_configured():
        print("schema: skip — DATABASE_URL 未设置（请检查 .env.runtime）")
        return 1
    try:
        if create_tables_if_needed():
            print("schema: OK — 已根据 ORM 元数据创建或确认表结构")
            return 0
    except OperationalError as e:
        print("schema: 无法连接 PostgreSQL，请先启动数据库。例如：")
        print("  docker compose -f docker-compose.postgres.yml up -d")
        orig = getattr(e, "orig", None)
        detail = repr(orig) if orig is not None else str(e)
        print(f"  详情: {detail}")
        return 2
    print("schema: 失败 — get_engine 返回 None")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
