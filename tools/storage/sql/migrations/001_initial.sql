-- PaperSearchAssistant2 — PostgreSQL 初始结构（与 tools/storage/sql/models.py 对齐）
-- 用法: psql "$DATABASE_URL" -f tools/storage/sql/migrations/001_initial.sql
-- 亦可改用 Python: python -c "from tools.storage.sql.schema_init import create_tables_if_needed; create_tables_if_needed()"

CREATE TABLE IF NOT EXISTS papers (
    id BIGSERIAL PRIMARY KEY,
    arxiv_id VARCHAR(64) UNIQUE,
    doi VARCHAR(128),
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    abstract TEXT,
    authors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    venue VARCHAR(255),
    published_at TIMESTAMPTZ,
    year INT,
    source_url TEXT,
    pdf_path TEXT NOT NULL,
    pdf_sha256 VARCHAR(64) NOT NULL,
    page_count INT,
    language VARCHAR(16) NOT NULL DEFAULT 'en',
    parse_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    ingest_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS ix_papers_title_norm ON papers(title_norm);

CREATE TABLE IF NOT EXISTS paper_sections (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    parent_section_id BIGINT REFERENCES paper_sections(id) ON DELETE CASCADE,
    section_level INT NOT NULL,
    section_number VARCHAR(64),
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    page_start INT,
    page_end INT,
    order_index INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_paper_sections_paper_order ON paper_sections(paper_id, order_index);

CREATE TABLE IF NOT EXISTS paper_chunks (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id BIGINT REFERENCES paper_sections(id) ON DELETE SET NULL,
    chunk_index INT NOT NULL,
    chunk_role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    content_norm TEXT NOT NULL,
    summary_text TEXT,
    token_count INT NOT NULL,
    page_from INT,
    page_to INT,
    block_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    prev_chunk_id BIGINT,
    next_chunk_id BIGINT,
    has_table BOOLEAN NOT NULL DEFAULT FALSE,
    has_figure BOOLEAN NOT NULL DEFAULT FALSE,
    importance_score NUMERIC(6,3) NOT NULL DEFAULT 0,
    chroma_doc_id VARCHAR(128) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_paper_chunks_paper_idx ON paper_chunks(paper_id, chunk_index);
CREATE INDEX IF NOT EXISTS ix_paper_chunks_paper_role ON paper_chunks(paper_id, chunk_role);
CREATE INDEX IF NOT EXISTS ix_paper_chunks_section ON paper_chunks(section_id);
CREATE INDEX IF NOT EXISTS ix_paper_chunks_fts ON paper_chunks USING gin (to_tsvector('simple', content_norm));

CREATE TABLE IF NOT EXISTS paper_tables (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id BIGINT REFERENCES paper_sections(id) ON DELETE SET NULL,
    page_no INT NOT NULL,
    table_number VARCHAR(64),
    title TEXT,
    caption_text TEXT,
    summary_text TEXT,
    markdown_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_figures (
    id BIGSERIAL PRIMARY KEY,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id BIGINT REFERENCES paper_sections(id) ON DELETE SET NULL,
    page_no INT NOT NULL,
    figure_number VARCHAR(64),
    caption_text TEXT,
    ocr_text TEXT,
    vision_summary TEXT,
    image_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_section_summaries (
    id BIGSERIAL PRIMARY KEY,
    section_id BIGINT NOT NULL UNIQUE REFERENCES paper_sections(id) ON DELETE CASCADE,
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_role VARCHAR(32),
    summary_text TEXT,
    keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_paper_section_summaries_paper_role ON paper_section_summaries(paper_id, section_role);

CREATE TABLE IF NOT EXISTS paper_summary_views (
    paper_id BIGINT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    abstract_summary TEXT,
    intro_summary TEXT,
    method_summary TEXT,
    result_summary TEXT,
    conclusion_summary TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_ingest_jobs (
    id VARCHAR(36) PRIMARY KEY,
    paper_id BIGINT REFERENCES papers(id) ON DELETE SET NULL,
    job_type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    stage VARCHAR(64) NOT NULL,
    error_message TEXT,
    metrics_json JSONB,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(128),
    title TEXT,
    namespace VARCHAR(128) NOT NULL DEFAULT 'default',
    summary TEXT,
    last_message_at TIMESTAMPTZ,
    message_count INT NOT NULL DEFAULT 0,
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_chat_sessions_updated ON chat_sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_chat_sessions_namespace ON chat_sessions(namespace);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(16) NOT NULL DEFAULT 'text',
    has_images BOOLEAN NOT NULL DEFAULT FALSE,
    has_files BOOLEAN NOT NULL DEFAULT FALSE,
    extra_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS chat_attachments (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    session_id VARCHAR(128) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(64),
    file_size BIGINT,
    asset_kind VARCHAR(16) NOT NULL,
    session_ingest_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
