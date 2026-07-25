# PaperSearchAssistant2 Database Redesign Plan

## 1. Goal

This document defines a new storage and retrieval architecture for `PaperSearchAssistant2`.

The redesign must solve these problems:

- Local paper retrieval is too slow.
- Conversation history depends too much on JSONL scanning.
- Chroma currently stores too much mixed business data.
- Paper metadata, chunks, figures, tables, and session uploads are not modeled as first-class entities.
- Retrieval lacks SQL pre-filtering and Redis hot-path caching.

The target architecture uses:

- PostgreSQL: source of truth for structured data
- Redis: hot cache, queue, rate limit, session state
- Chroma: vector retrieval only
- JSONL: append-only archive and audit logs

This plan is intended to be executed directly in Cursor.

---

## 2. Core Design Principles

### 2.1 Storage responsibilities

- PostgreSQL stores business truth.
- Chroma stores embedding lookup units only.
- Redis stores short-lived hot data and async workflow state.
- JSONL stores audit trails and cold archives only.

### 2.2 Retrieval responsibilities

- SQL performs candidate narrowing.
- Chroma performs semantic recall.
- SQL FTS/BM25 performs exact and term-sensitive recall.
- Redis accelerates repeated lookups.
- Final prompt hydration always reads authoritative content from PostgreSQL.

### 2.3 Modeling principles

- A paper is not just one blob of text.
- A paper must be decomposed into sections, blocks, chunks, tables, figures, and assets.
- A conversation is not just a JSONL file.
- Chat sessions, messages, attachments, and vectorized memory must be modeled explicitly.

---

## 3. New Storage Architecture

## 3.1 PostgreSQL

Use PostgreSQL as the primary database.

It stores:

- paper metadata
- paper structure
- paper blocks
- paper retrieval chunks
- tables
- figures
- assets
- ingest jobs
- chat sessions
- chat messages
- chat attachments
- vector reference mappings

## 3.2 Redis

Use Redis for:

- recent session cache
- paper detail cache
- section tree cache
- retrieval result cache
- chunk neighbor cache
- ingest queues
- OCR queues
- embedding queues
- throttling and call limits

## 3.3 Chroma

Use Chroma only for semantic vector retrieval.

It stores:

- paper retrieval chunks
- session upload chunks
- summarized chat memory

Do not use Chroma as the business source of truth.

## 3.4 JSONL

Use JSONL only for append-only archive and diagnostics:

- raw chat event archive
- paper ingest logs
- OCR failure logs
- retrieval traces
- agent traces

JSONL must not be used for online primary reads.

---

## 4. New Directory Structure

Create or refactor to this structure:

```text
tools/
  storage/
    sql/
      db.py
      models.py
      migrations/
    repos/
      paper_repo.py
      chat_repo.py
      asset_repo.py
      job_repo.py
      retrieval_repo.py
    vector/
      chroma_store.py
    redis/
      cache.py
      keys.py
      queue.py
    archive/
      jsonl_writer.py
      event_logger.py
```

Legacy files to phase out or convert into facades:

- `tools/storage/papers_db.py`
- `tools/storage/paper_library.py`
- `tools/storage/long_memory.py`
- `tools/storage/chat_turn_embed.py`

---

## 5. PostgreSQL Schema

## 5.1 papers

Purpose:

- primary paper metadata
- list page
- detail page
- deduplication

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `arxiv_id VARCHAR(64) UNIQUE NULL`
- `doi VARCHAR(128) NULL`
- `title TEXT NOT NULL`
- `title_norm TEXT NOT NULL`
- `abstract TEXT NULL`
- `authors_json JSONB NOT NULL DEFAULT '[]'::jsonb`
- `venue VARCHAR(255) NULL`
- `published_at TIMESTAMPTZ NULL`
- `year INT NULL`
- `source_url TEXT NULL`
- `pdf_path TEXT NOT NULL`
- `pdf_sha256 VARCHAR(64) NOT NULL`
- `page_count INT NULL`
- `language VARCHAR(16) NOT NULL DEFAULT 'en'`
- `parse_status VARCHAR(32) NOT NULL DEFAULT 'pending'`
- `ingest_status VARCHAR(32) NOT NULL DEFAULT 'pending'`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- unique index on `arxiv_id`
- index on `year`
- index on `published_at`
- GIN on `authors_json`
- B-tree on `title_norm`
- trigram index on `title`

## 5.2 paper_sections

Purpose:

- section tree
- section-level filtering
- support queries like "what does section 3 say"

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `parent_section_id BIGINT NULL REFERENCES paper_sections(id) ON DELETE CASCADE`
- `section_level INT NOT NULL`
- `section_number VARCHAR(64) NULL`
- `title TEXT NOT NULL`
- `title_norm TEXT NOT NULL`
- `page_start INT NULL`
- `page_end INT NULL`
- `order_index INT NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- index on `(paper_id, order_index)`
- index on `(paper_id, title_norm)`

## 5.3 paper_blocks

Purpose:

- structured parse output
- atomic content units before chunk generation

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `section_id BIGINT NULL REFERENCES paper_sections(id) ON DELETE SET NULL`
- `block_type VARCHAR(32) NOT NULL`
- `page_no INT NOT NULL`
- `order_index INT NOT NULL`
- `text_content TEXT NULL`
- `text_norm TEXT NULL`
- `token_count INT NOT NULL DEFAULT 0`
- `char_count INT NOT NULL DEFAULT 0`
- `bbox_json JSONB NULL`
- `asset_id BIGINT NULL`
- `table_id BIGINT NULL`
- `figure_id BIGINT NULL`
- `source_parser VARCHAR(32) NOT NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Recommended `block_type` values:

- `title`
- `heading`
- `paragraph`
- `list`
- `equation`
- `caption`
- `table_caption`
- `figure_caption`
- `table_body`
- `footnote`

Indexes:

- index on `(paper_id, page_no, order_index)`
- index on `(paper_id, block_type)`
- GIN on `bbox_json`

## 5.4 paper_chunks

Purpose:

- semantic retrieval unit
- exact retrieval hydration unit
- chunk adjacency traversal

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `section_id BIGINT NULL REFERENCES paper_sections(id) ON DELETE SET NULL`
- `chunk_index INT NOT NULL`
- `chunk_role VARCHAR(32) NOT NULL`
- `content TEXT NOT NULL`
- `content_norm TEXT NOT NULL`
- `summary_text TEXT NULL`
- `token_count INT NOT NULL`
- `page_from INT NULL`
- `page_to INT NULL`
- `block_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb`
- `prev_chunk_id BIGINT NULL`
- `next_chunk_id BIGINT NULL`
- `has_table BOOLEAN NOT NULL DEFAULT FALSE`
- `has_figure BOOLEAN NOT NULL DEFAULT FALSE`
- `importance_score NUMERIC(6,3) NOT NULL DEFAULT 0`
- `chroma_doc_id VARCHAR(128) NOT NULL UNIQUE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Recommended `chunk_role` values:

- `abstract`
- `introduction`
- `method`
- `experiment`
- `result`
- `conclusion`
- `table`
- `figure`
- `appendix`
- `generic`
- `paper_summary`
- `method_summary`
- `experiment_summary`

Indexes:

- index on `(paper_id, chunk_index)`
- index on `(paper_id, chunk_role)`
- index on `(section_id)`
- index on `(paper_id, has_table)`
- index on `(paper_id, has_figure)`
- GIN full text index on `to_tsvector('simple', content_norm)`

## 5.5 paper_tables

Purpose:

- structured table storage
- table summary retrieval
- exact table lookup

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `section_id BIGINT NULL REFERENCES paper_sections(id) ON DELETE SET NULL`
- `page_no INT NOT NULL`
- `table_number VARCHAR(64) NULL`
- `title TEXT NULL`
- `caption_text TEXT NULL`
- `summary_text TEXT NULL`
- `markdown_text TEXT NULL`
- `json_path TEXT NULL`
- `csv_path TEXT NULL`
- `html_path TEXT NULL`
- `bbox_json JSONB NULL`
- `parse_quality NUMERIC(4,3) NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- index on `(paper_id, page_no)`
- index on `(paper_id, table_number)`

## 5.6 paper_figures

Purpose:

- image and figure metadata
- OCR indexing
- vision summary indexing

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `section_id BIGINT NULL REFERENCES paper_sections(id) ON DELETE SET NULL`
- `page_no INT NOT NULL`
- `figure_number VARCHAR(64) NULL`
- `title TEXT NULL`
- `caption_text TEXT NULL`
- `ocr_text TEXT NULL`
- `vision_summary TEXT NULL`
- `vision_keywords_json JSONB NULL`
- `image_path TEXT NOT NULL`
- `thumbnail_path TEXT NULL`
- `bbox_json JSONB NULL`
- `parse_quality NUMERIC(4,3) NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- index on `(paper_id, page_no)`
- index on `(paper_id, figure_number)`
- GIN on `vision_keywords_json`

## 5.7 paper_assets

Purpose:

- file registry for extracted assets

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE`
- `asset_type VARCHAR(32) NOT NULL`
- `file_path TEXT NOT NULL`
- `mime_type VARCHAR(128) NULL`
- `size_bytes BIGINT NULL`
- `sha256 VARCHAR(64) NOT NULL`
- `page_no INT NULL`
- `bbox_json JSONB NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

## 5.8 paper_ingest_jobs

Purpose:

- background ingest orchestration
- status tracking
- retries

Columns:

- `id UUID PRIMARY KEY`
- `paper_id BIGINT NULL REFERENCES papers(id) ON DELETE SET NULL`
- `job_type VARCHAR(32) NOT NULL`
- `status VARCHAR(32) NOT NULL`
- `stage VARCHAR(64) NOT NULL`
- `error_message TEXT NULL`
- `metrics_json JSONB NULL`
- `started_at TIMESTAMPTZ NULL`
- `finished_at TIMESTAMPTZ NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

## 5.9 chat_sessions

Purpose:

- session list
- session metadata
- namespace binding

Columns:

- `id VARCHAR(128) PRIMARY KEY`
- `user_id VARCHAR(128) NULL`
- `title TEXT NULL`
- `namespace VARCHAR(128) NOT NULL`
- `summary TEXT NULL`
- `last_message_at TIMESTAMPTZ NULL`
- `message_count INT NOT NULL DEFAULT 0`
- `status VARCHAR(16) NOT NULL DEFAULT 'active'`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- index on `(updated_at DESC)`
- index on `user_id`
- index on `namespace`

## 5.10 chat_messages

Purpose:

- primary conversation storage
- pagination
- session hydration

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `session_id VARCHAR(128) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `role VARCHAR(16) NOT NULL`
- `content TEXT NOT NULL`
- `content_type VARCHAR(16) NOT NULL DEFAULT 'text'`
- `token_count INT NULL`
- `reply_to_id BIGINT NULL`
- `has_images BOOLEAN NOT NULL DEFAULT FALSE`
- `has_files BOOLEAN NOT NULL DEFAULT FALSE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

Indexes:

- index on `(session_id, created_at)`
- index on `(session_id, id)`

## 5.11 chat_attachments

Purpose:

- uploaded files and images
- session document mapping

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `message_id BIGINT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE`
- `session_id VARCHAR(128) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `file_name TEXT NOT NULL`
- `file_path TEXT NOT NULL`
- `file_type VARCHAR(64) NULL`
- `file_size BIGINT NULL`
- `asset_kind VARCHAR(16) NOT NULL`
- `session_ingest_id VARCHAR(128) NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

## 5.12 chat_memory_embeddings

Purpose:

- mapping between session memory summaries and Chroma documents

Columns:

- `id BIGSERIAL PRIMARY KEY`
- `session_id VARCHAR(128) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE`
- `message_id BIGINT NULL REFERENCES chat_messages(id) ON DELETE SET NULL`
- `memory_type VARCHAR(32) NOT NULL`
- `summary_text TEXT NOT NULL`
- `chroma_doc_id VARCHAR(128) NOT NULL UNIQUE`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

---

## 6. Chroma Design

## 6.1 Collections

Create these collections:

- `paper_chunks_public`
- `paper_chunks_private`
- `chat_memory`
- `session_upload_chunks`

## 6.2 Chroma rules

Chroma must only store:

- retrievable text
- minimal filter metadata
- stable id references to PostgreSQL

Chroma must not store:

- authors arrays
- long JSON blobs
- full table markdown for large tables
- raw OCR dumps if very large
- business status fields

## 6.3 paper_chunks_public metadata

Store:

- `chunk_id`
- `paper_id`
- `section_id`
- `chunk_role`
- `year`
- `has_table`
- `has_figure`

## 6.4 chat_memory metadata

Store:

- `session_id`
- `message_id`
- `memory_type`
- `created_at`

## 6.5 session_upload_chunks metadata

Store:

- `session_id`
- `attachment_id`
- `chunk_id`
- `namespace`
- `created_at`

---

## 7. Paper Parsing and Chunking Strategy

## 7.1 Parsing stages

Each paper ingest job must follow this flow:

1. save PDF
2. register paper row
3. parse pages into sections and blocks
4. extract tables
5. extract figures
6. generate retrieval chunks
7. write structured data into PostgreSQL
8. embed retrieval chunks into Chroma
9. warm Redis caches
10. append JSONL archive logs

## 7.2 block layer

Blocks are atomic parse outputs.

Examples:

- section title
- paragraph
- list
- formula explanation
- figure caption
- table caption
- footnote

Blocks are not directly the main retrieval unit.

## 7.3 chunk layer

Chunks are retrieval units and must be generated from blocks.

Rules:

- target size: `350-650 tokens`
- overlap by blocks, not by raw token window
- do not cross major sections
- attach the nearest heading to each chunk
- abstract is always a separate chunk
- conclusion is always a separate chunk
- figure caption text is usually a separate chunk
- table summary is usually a separate chunk

## 7.4 special derived chunks

Generate additional high-value chunks:

- `paper_summary`
- `method_summary`
- `experiment_summary`
- `table_summary`
- `figure_summary`

These chunks are extremely useful for fast and accurate retrieval.

---

## 8. Table Storage Strategy

Each table should have four representations when possible:

1. raw extracted image
2. structured JSON
3. markdown or HTML form
4. table summary text

Retrieval should primarily use the summary text, not the entire raw table.

Example summary text:

```text
Table 2. Results on HotpotQA.
Columns: model, EM, F1, latency.
Best result is Retriever-X with F1 78.3.
Compared with baseline DPR, F1 improves by 4.8 points.
```

Rules:

- small tables may include markdown in retrieval context
- medium and large tables should use summary text plus key rows
- huge tables must not be embedded as full raw text

---

## 9. Figure and Image Storage Strategy

Store figures in filesystem and metadata in PostgreSQL.

Recommended paths:

- `data/papers/assets/{paper_id}/figures/figure_001.png`
- `data/papers/assets/{paper_id}/thumbs/figure_001.jpg`

For retrieval, build a text representation from:

- figure number
- caption
- OCR text
- vision summary
- keywords

Example retrieval text:

```text
Figure 3. Model architecture.
The figure shows a dual-encoder retrieval pipeline followed by reranking.
OCR text includes: query encoder, document encoder, top-k retrieval, rerank.
Keywords: retrieval, reranker, architecture.
```

First implementation does not need image embeddings.
Textualized figure retrieval is enough for the first major redesign.

---

## 10. Redis Design

## 10.1 Keys

Recommended key patterns:

- `chat:recent:{session_id}`
- `paper:detail:{paper_id}`
- `paper:sections:{paper_id}`
- `chunk:neighbors:{chunk_id}`
- `search:paper:{query_hash}`
- `search:chat:{query_hash}`
- `paper:summary:{paper_id}`
- `paper:table_summary:{table_id}`
- `paper:figure_summary:{figure_id}`

## 10.2 TTL recommendations

- recent chat cache: 30-120 minutes
- paper detail cache: 6 hours
- section tree cache: 24 hours
- search result cache: 10-30 minutes
- chunk neighbor cache: 6 hours

## 10.3 Async queues

Use Redis-backed queues for:

- paper ingest
- OCR
- embedding
- figure summarization
- table summarization

Suggested queue names:

- `queue:paper_ingest`
- `queue:ocr`
- `queue:embed`
- `queue:figure_summary`
- `queue:table_summary`

---

## 11. JSONL Archive Design

Use JSONL only for append-only logs.

Recommended archive files:

- `data/archive/chat_events/{session_id}.jsonl`
- `data/archive/paper_ingest/{paper_id}.jsonl`
- `data/archive/retrieval_trace/{date}.jsonl`
- `data/archive/ocr_failures/{date}.jsonl`
- `data/archive/agent_trace/{date}.jsonl`

Online application paths must not depend on these files for primary reads.

---

## 12. Retrieval Flow

## 12.1 Paper question retrieval

For a query like:

"What is the method used in this paper?"

The retrieval flow must be:

1. identify target paper in PostgreSQL
2. fetch paper section tree from Redis or PostgreSQL
3. SQL pre-filter sections by title and scope
4. semantic search candidate chunks in Chroma
5. exact recall using SQL FTS/BM25 on `paper_chunks`
6. merge and rerank
7. hydrate neighbors from Redis or PostgreSQL
8. if figures or tables are hit, hydrate table and figure summaries
9. build final prompt

## 12.2 Session chat retrieval

For a query like:

"What did I upload just now?"

The retrieval flow must be:

1. fetch recent messages from Redis
2. fallback to PostgreSQL if Redis misses
3. if semantic history is needed, search `chat_memory`
4. if uploaded files are needed, search `session_upload_chunks`
5. hydrate exact message and attachment rows from PostgreSQL

---

## 13. Codebase Refactor Plan

## 13.1 New modules to create

Create:

- `tools/storage/sql/db.py`
- `tools/storage/sql/models.py`
- `tools/storage/sql/migrations/`
- `tools/storage/repos/paper_repo.py`
- `tools/storage/repos/chat_repo.py`
- `tools/storage/repos/asset_repo.py`
- `tools/storage/repos/job_repo.py`
- `tools/storage/repos/retrieval_repo.py`
- `tools/storage/vector/chroma_store.py`
- `tools/storage/redis/cache.py`
- `tools/storage/redis/keys.py`
- `tools/storage/redis/queue.py`
- `tools/storage/archive/jsonl_writer.py`
- `tools/storage/archive/event_logger.py`

## 13.2 Existing files to rewrite

### `tools/agent/paper_ingest.py`

Rewrite responsibilities:

- stop using thin SQLite-only paper metadata logic
- create or update `papers`
- parse sections, blocks, figures, and tables
- generate retrieval chunks
- bulk write all structured entities into PostgreSQL
- embed only retrieval chunks into Chroma
- warm Redis paper caches
- write JSONL ingest logs

### `tools/agent/session_file_embed.py`

Rewrite responsibilities:

- register uploaded file in `chat_attachments`
- create `session_upload_chunks`
- write attachment metadata into PostgreSQL
- embed file chunks into Chroma collection `session_upload_chunks`
- do not rely on conversation JSONL as the primary tracking source

### `tools/agent/conversation.py`

Rewrite responsibilities:

- primary write path goes to PostgreSQL
- recent session reads go to Redis first
- JSONL becomes append-only archive only
- `get_recent_messages` must read Redis then PostgreSQL
- `add_session_embed` must also persist structured attachment/session-ingest relations

### `tools/storage/long_memory.py`

Rewrite responsibilities:

- stop treating vector memory as the main conversation store
- only embed summarized memory, not all raw turns
- source memory rows from PostgreSQL

### `tools/storage/chat_turn_embed.py`

Rewrite responsibilities:

- convert into optional summarized chat memory writer
- do not embed every raw Q/A turn by default

### `tools/rag/knowledge.py`

Rewrite responsibilities:

- turn into a Chroma retrieval facade
- add query functions with PostgreSQL-aware filters
- support restricted retrieval by `paper_id`, `section_id`, `session_id`, `chunk_role`
- keep Chroma metadata minimal

### `tools/storage/papers_db.py`

Status:

- retire from primary path
- optionally keep as a compatibility shim during migration

### `tools/storage/paper_library.py`

Status:

- convert into a service facade backed by PostgreSQL repos

## 13.3 API changes in `main.py`

Add or refactor endpoints:

- `POST /papers/search_local`
- `GET /papers/{paper_id}/sections`
- `GET /papers/{paper_id}/tables`
- `GET /papers/{paper_id}/figures`
- `GET /sessions/{session_id}/messages`
- `GET /sessions/{session_id}/attachments`

Refactor existing endpoints to use repository layer, not direct JSONL or legacy SQLite access.

---

## 14. Migration Plan

## Phase 1

Introduce PostgreSQL, Redis, and repository layer.

Tasks:

- add new DB modules
- add migrations
- keep current code running
- start dual-write for papers and sessions

## Phase 2

Move chat to PostgreSQL primary storage.

Tasks:

- `conversation.py` writes SQL first
- Redis recent cache added
- JSONL switched to archive-only

## Phase 3

Move paper ingest to structured storage.

Tasks:

- parse PDF into sections, blocks, chunks, tables, figures
- stop relying on `papers_db.py` as main paper store
- Chroma becomes retrieval-only

## Phase 4

Switch retrieval to full layered path.

Tasks:

- Redis cache lookup
- SQL pre-filter
- Chroma semantic recall
- SQL FTS/BM25
- rerank
- SQL hydration

## Phase 5

Remove old direct storage paths.

Tasks:

- remove JSONL primary reads
- remove legacy paper index assumptions
- remove direct catch-all namespace writes where inappropriate

---

## 15. Implementation Priorities for Cursor

Cursor should execute in this order:

1. add new PostgreSQL connection and migration layer
2. define SQLAlchemy or equivalent models for all new tables
3. build repository layer for papers, chat, attachments, figures, tables, chunks
4. add Redis cache module and key helpers
5. refactor conversation persistence
6. refactor paper ingest pipeline
7. refactor Chroma wrapper
8. refactor retrieval flow in agent path
9. add new HTTP endpoints
10. add data migration scripts from old stores

---

## 16. Non-Negotiable Rules

- Do not use JSONL as primary online chat storage after migration.
- Do not use Chroma as the business source of truth.
- Do not store oversized metadata payloads in Chroma.
- Do not embed full huge raw tables directly.
- Do not store all chat turns as raw vector memory by default.
- Always use PostgreSQL ids as the canonical reference ids.
- Always hydrate final content from PostgreSQL after vector recall.

---

## 17. Expected Outcome

After this redesign:

- paper retrieval will be faster because SQL narrows scope before vector search
- session history will be faster because Redis and PostgreSQL replace JSONL primary reads
- table and figure retrieval will become explicit and accurate
- Chroma collections will become smaller and cleaner
- repeated queries will be much faster because of Redis caching
- the system will be easier to extend for multimodal and private-library features

---

## 18. Immediate Deliverables

The first deliverables Cursor should produce are:

1. PostgreSQL schema migration files
2. Redis cache and queue helpers
3. repository layer
4. refactored `conversation.py`
5. refactored `paper_ingest.py`
6. refactored `knowledge.py`
7. migration scripts from:
   - `data/conversations/*.jsonl`
   - `data/papers/papers.db`
   - old Chroma metadata references

