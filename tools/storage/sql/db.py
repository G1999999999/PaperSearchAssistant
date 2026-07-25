from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def is_database_configured() -> bool:
    from config import DATABASE_URL

    return bool((DATABASE_URL or "").strip())


def get_engine() -> Optional[Engine]:
    global _engine
    if not is_database_configured():
        return None
    if _engine is None:
        from config import DATABASE_ECHO_SQL, DATABASE_URL

        _engine = create_engine(
            DATABASE_URL,
            echo=bool(DATABASE_ECHO_SQL),
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> Optional[sessionmaker]:
    global _session_factory
    eng = get_engine()
    if eng is None:
        return None
    if _session_factory is None:
        _session_factory = sessionmaker(bind=eng, expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """事务性 Session 上下文。"""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("DATABASE_URL 未配置")
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
