"""SQLAlchemy 2.0 模型：与 DATABASE_REDESIGN_PLAN.md /schema 对齐。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_norm: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors_json: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    venue: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pdf_path: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[str] = mapped_column(String(16), nullable=False, server_default="en")
    parse_status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending")
    ingest_status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    sections: Mapped[list["PaperSection"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    chunks: Mapped[list["PaperChunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class PaperSection(Base):
    __tablename__ = "paper_sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    parent_section_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="CASCADE"), nullable=True
    )
    section_level: Mapped[int] = mapped_column(Integer, nullable=False)
    section_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_norm: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper: Mapped["Paper"] = relationship(back_populates="sections")
    __table_args__ = (Index("ix_paper_sections_paper_order", "paper_id", "order_index"),)


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="SET NULL"), nullable=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_norm: Mapped[str] = mapped_column(Text, nullable=False)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_from: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_to: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    block_ids_json: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    prev_chunk_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    has_table: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    has_figure: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    importance_score: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, server_default="0")
    chroma_doc_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper: Mapped["Paper"] = relationship("Paper", back_populates="chunks")
    __table_args__ = (
        Index("ix_paper_chunks_paper_idx", "paper_id", "chunk_index"),
        Index("ix_paper_chunks_paper_role", "paper_id", "chunk_role"),
        Index("ix_paper_chunks_section", "section_id"),
    )


class PaperTable(Base):
    __tablename__ = "paper_tables"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="SET NULL"), nullable=True
    )
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    table_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    caption_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    markdown_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PaperFigure(Base):
    __tablename__ = "paper_figures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="SET NULL"), nullable=True
    )
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    figure_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    caption_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vision_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PaperSectionSummary(Base):
    __tablename__ = "paper_section_summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_role: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keywords_json: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    __table_args__ = (
        Index("ix_paper_section_summaries_paper_role", "paper_id", "section_role"),
    )


class PaperSummaryView(Base):
    __tablename__ = "paper_summary_views"

    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    abstract_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intro_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    method_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    conclusion_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, server_default="default")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="text")
    has_images: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    has_files: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    extra_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    attachments: Mapped[list["ChatAttachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )
    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at"),)


class ChatAttachment(Base):
    __tablename__ = "chat_attachments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    asset_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    session_ingest_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    message: Mapped["ChatMessage"] = relationship(back_populates="attachments")


class PaperIngestJob(Base):
    __tablename__ = "paper_ingest_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    paper_id: Mapped[Optional[int]] = mapped_column(ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
