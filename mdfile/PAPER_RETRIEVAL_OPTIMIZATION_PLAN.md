# Paper Retrieval Optimization Plan

## 1. Goal

This document defines the new paper retrieval strategy for `PaperSearchAssistant2`.

The target is not only to make retrieval faster, but also to make it:

- more accurate
- more controllable
- better at handling papers, tables, figures, and long context
- easier to evaluate and tune

This plan assumes the database redesign in:

- `DATABASE_REDESIGN_PLAN.md`

This document focuses specifically on RAG and retrieval.

---

## 2. Retrieval Objectives

The new paper retrieval system must support:

- semantic retrieval with vectors
- exact keyword retrieval with BM25 / SQL FTS
- reranker-based reordering
- structured filtering by paper, section, year, and content type
- table-aware retrieval
- figure-aware retrieval
- conversation-aware but paper-first retrieval
- multi-stage recall
- context expansion around hit chunks

---

## 3. High-Level Retrieval Architecture

The new paper retrieval path must be a layered hybrid RAG pipeline:

1. query understanding
2. structured scope narrowing
3. multiple recall channels
4. recall fusion
5. reranking
6. context expansion
7. evidence packaging
8. answer generation

Recommended pipeline:

```text
User Query
  -> Query Understanding
  -> SQL Prefilter
  -> Vector Recall
  -> BM25 / FTS Recall
  -> Table Recall
  -> Figure Recall
  -> Fusion
  -> Reranker
  -> Neighbor Expansion
  -> Context Compression / Packing
  -> LLM Answer
```

---

## 4. Retrieval Strategy Overview

The retrieval system should support these strategies:

- `vector_only`
- `bm25_only`
- `hybrid`
- `hybrid_rerank`
- `paper_focused`
- `table_focused`
- `figure_focused`
- `multi_query_hybrid`
- `multi_stage_deep`

Recommended default:

- `hybrid_rerank`

Recommended advanced strategy for papers:

- `multi_query_hybrid`

Recommended best-quality strategy:

- `multi_stage_deep`

---

## 5. Query Understanding Layer

Before retrieval, classify the user query.

The system must infer:

- target paper or paper set
- likely section
- whether the query is asking for:
  - summary
  - method
  - experiment
  - result
  - conclusion
  - table content
  - figure content
  - citation-like factual lookup
- whether the query needs exact value retrieval
- whether the query is broad or narrow

Suggested query intent labels:

- `paper_summary`
- `paper_method`
- `paper_experiment`
- `paper_result`
- `paper_conclusion`
- `paper_comparison`
- `table_lookup`
- `figure_lookup`
- `metric_lookup`
- `citation_lookup`
- `open_ended_analysis`

Recommended implementation:

- start with rule-based detection
- optionally add lightweight LLM query router later

Examples:

- "这篇论文的方法是什么" -> `paper_method`
- "Table 3 结果如何" -> `table_lookup`
- "Figure 2 讲了什么" -> `figure_lookup`
- "这篇论文和 DPR 的区别是什么" -> `paper_comparison`

---

## 6. Structured Prefilter Layer

Before vector search, narrow the candidate set with structured constraints.

This is critical for speed.

Possible filters:

- `paper_id`
- `paper_ids`
- `section_id`
- `section_title`
- `chunk_role`
- `year`
- `has_table`
- `has_figure`
- `page range`

Examples:

- If the user is clearly asking about one paper, do not search the entire paper collection.
- If the user asks about methods, prioritize sections with titles matching:
  - method
  - approach
  - architecture
  - model
- If the user asks about results, prioritize:
  - experiment
  - result
  - evaluation
  - ablation

Implementation rule:

- SQL prefilter should reduce the candidate pool before semantic recall whenever possible.

---

## 7. Recall Channels

Use multiple parallel recall channels.

## 7.1 Vector Recall

Use Chroma for semantic retrieval over `paper_chunks`.

Good for:

- paraphrases
- conceptual matching
- long-form questions
- open-ended "what does this paper do" type queries

Recommended settings:

- top_k semantic recall: `20-50`
- embedding unit: `paper_chunks`
- use chunk metadata filters wherever possible

## 7.2 BM25 / SQL FTS Recall

Use PostgreSQL FTS or BM25-like lexical recall over:

- chunk content
- chunk summary
- table summaries
- figure summaries

Good for:

- exact terms
- acronyms
- metric names
- figure numbers
- table numbers
- dataset names
- algorithm names

Recommended settings:

- top_k lexical recall: `20-50`

## 7.3 Title / Abstract Recall

Add a dedicated recall path for:

- title
- abstract
- paper summary chunks

Useful when queries ask:

- "这篇论文主要做什么"
- "论文贡献是什么"

## 7.4 Table Recall

Search table summaries separately.

Use:

- caption text
- summary text
- key row summaries
- metric names

This recall path should be boosted if query intent is `table_lookup` or `metric_lookup`.

## 7.5 Figure Recall

Search figure summaries separately.

Use:

- caption text
- OCR text
- vision summary
- keywords

This recall path should be boosted if query intent is `figure_lookup`.

## 7.6 Session Upload Recall

If the current session uploaded documents or paper files, search them as a separate source.

Do not merge session uploads into the same logical pool too early.
Instead:

- retrieve them separately
- annotate source
- merge later

## 7.7 Conversation Memory Recall

Use only as a supplement, not as the primary evidence source for paper questions.

Rules:

- if the user asks about a paper, paper recall has priority
- memory recall is only for recovering prior conversation references

---

## 8. Recall Fusion

After collecting candidates from multiple channels, fuse them.

Recommended fusion methods:

- Reciprocal Rank Fusion (RRF)
- weighted score fusion

Recommended default:

- RRF

Suggested weighted boosts:

- vector recall: base weight `1.0`
- BM25 recall: base weight `1.0`
- title/abstract recall: `1.1`
- table recall when table intent: `1.4`
- figure recall when figure intent: `1.4`
- section-matched chunks: `1.2`
- session uploads when explicitly referenced: `1.3`

Recommended fusion steps:

1. deduplicate by `chunk_id`
2. keep best source scores and source labels
3. compute fused score
4. take top `30-80` candidates into reranker

---

## 9. Reranking Layer

Reranker is mandatory for high-quality paper retrieval.

Use a cross-encoder reranker after fusion.

Current good candidates:

- `BAAI/bge-reranker-v2-m3`
- `BAAI/bge-reranker-large`
- other cross-encoder rerankers depending on hardware

## 9.1 Why reranker is needed

Vector search is good at coarse recall, but it often:

- over-recalls semantically similar but not directly relevant chunks
- misses exact answer location among many related chunks
- ranks generic summary chunks too high

Reranker improves:

- local precision
- answer grounding quality
- evidence ordering

## 9.2 Reranker input

Each rerank item should include:

- chunk content
- section title
- paper title
- optional chunk role

Suggested rerank text format:

```text
Paper: {paper_title}
Section: {section_title}
Role: {chunk_role}
Content: {chunk_content}
```

## 9.3 Reranker output size

Recommended:

- rerank top input: `30-60`
- keep top final evidence: `5-10`

---

## 10. Chunking Strategy

Chunking quality determines retrieval quality.

The system must use structure-aware chunking.

## 10.1 Chunk goals

Chunks should be:

- semantically coherent
- locally answerable
- not too long
- not too fragmented
- easy to expand with neighbors

## 10.2 Recommended chunking rules

### Standard text chunks

- target size: `350-650 tokens`
- overlap: previous `1-2 blocks`
- preserve heading + nearby content
- never merge across unrelated major sections

### Special chunks

Always create separate chunks for:

- abstract
- conclusion
- method summary
- experiment summary
- figure summary
- table summary

### Chunk roles

Every chunk should have a role:

- `abstract`
- `intro`
- `method`
- `experiment`
- `result`
- `conclusion`
- `table`
- `figure`
- `appendix`
- `generic`

## 10.3 Why role labels matter

Role labels allow:

- prefiltering
- intent-aware boosting
- cheaper and more accurate retrieval

---

## 11. Context Expansion

After reranking, expand context around hit chunks.

This is important because:

- the hit chunk may not contain the full explanation
- a chunk may depend on the previous paragraph
- method and experiment descriptions often span adjacent chunks

Recommended expansion strategy:

- for each selected chunk, fetch:
  - self
  - previous chunk
  - next chunk
- only expand within same paper and same nearby section
- avoid excessive overlap

Recommended limits:

- max neighbor expansion depth: `1`
- final packed chunks after expansion: `6-12`

Boost rules:

- if chunk role is `table` or `figure`, also fetch the linked table or figure summary

---

## 12. Table Retrieval Plan

Tables must be first-class retrieval objects.

## 12.1 Representation

For each table store:

- caption
- summary
- key metrics
- optional markdown
- optional structured JSON

## 12.2 Table summary generation

Generate a table summary during ingest:

- describe what the table is about
- name important columns
- mention best result
- mention comparison baseline
- mention metric deltas if possible

## 12.3 Table-focused retrieval

When the query mentions:

- `table`
- `表`
- `指标`
- `结果`
- metric names like `F1`, `EM`, `BLEU`, `accuracy`

Then:

- increase table recall weight
- increase chunk role `table` boost
- prefer summaries over generic method chunks

---

## 13. Figure Retrieval Plan

Figures must also be first-class retrieval objects.

## 13.1 Representation

For each figure store:

- caption
- OCR text
- vision summary
- keywords

## 13.2 Figure summary generation

Generate a figure summary that captures:

- what the figure depicts
- main entities shown
- flow or architecture if present
- chart meaning if present

## 13.3 Figure-focused retrieval

When query mentions:

- `figure`
- `图`
- `架构图`
- `流程图`
- `图 2`

Then:

- increase figure recall weight
- boost chunks with role `figure`
- prioritize linked figure summary evidence

---

## 14. Additional RAG Methods to Add

Beyond vector + BM25 + rerank, the system should support these RAG enhancements.

## 14.1 Multi-Query Retrieval

Generate multiple rewritten retrieval queries for one user question.

Example:

Original:

- "这篇论文的方法有什么创新"

Generated:

- "method contribution of the paper"
- "novelty of the proposed approach"
- "paper method innovation"

Use:

- vector recall for all subqueries
- BM25 for selected subqueries
- merge results with RRF

This is one of the highest-value improvements.

## 14.2 Query Decomposition

Split compound questions into subquestions.

Example:

- "这篇论文的方法是什么，实验结果怎么样，和 DPR 有什么区别"

Decompose into:

- what is the method
- what are the experiment results
- how does it differ from DPR

Retrieve for each subquestion separately, then merge.

## 14.3 Hierarchical Retrieval

Use two-stage retrieval over paper structure:

1. retrieve relevant sections or summaries
2. retrieve chunks inside those sections

This is especially useful for long papers.

## 14.4 Parent-Child Retrieval

Store child chunks for retrieval but answer with parent context.

Example:

- child chunk = local paragraph-level chunk
- parent = section-level merged chunk

Retrieve child for precision, hydrate parent for context.

## 14.5 Contextual Compression

Before sending context to the LLM, compress low-value passages.

Possible methods:

- sentence filtering
- section-aware trimming
- LLM-based evidence summarization

Use only after retrieval, not before indexing.

## 14.6 Metadata-Aware Retrieval

Apply boosts or filters by:

- chunk role
- section title
- publication year
- whether chunk has table
- whether chunk has figure

## 14.7 Self-Query Retrieval

Use a lightweight query parser or LLM to extract filters:

- target paper
- section
- content type

Then convert them into structured filters plus semantic text query.

## 14.8 Hybrid Sparse-Dense Retrieval

This is the core recommended design:

- dense retrieval via Chroma
- sparse retrieval via SQL FTS/BM25
- fuse with RRF
- rerank with cross-encoder

## 14.9 Answer-Aware Retrieval Packing

Context packing should not be purely top-k.

It should try to maximize:

- coverage
- non-redundancy
- answerability

This can be implemented with:

- MMR
- role diversity constraints
- section diversity constraints

## 14.10 Citation-Centric Evidence Packaging

For final answer generation, every evidence block should be traceable to:

- paper title
- section
- page
- figure/table number if relevant

This improves grounded answering.

---

## 15. Recommended Retrieval Profiles

Implement multiple profiles.

## 15.1 Fast Profile

Use for quick lookup:

- SQL prefilter
- vector top 20
- BM25 top 20
- fuse
- rerank top 20
- output top 5

## 15.2 Balanced Profile

Recommended default:

- query rewrite
- SQL prefilter
- vector top 30
- BM25 top 30
- table/figure recall when applicable
- fuse with RRF
- rerank top 40
- neighbor expansion
- output top 8

## 15.3 Deep Research Profile

Use for difficult questions:

- query decomposition
- multi-query retrieval
- hierarchical retrieval
- vector + BM25 + table + figure recall
- fuse top 60
- rerank top 60
- MMR context packing
- output top 10-12 evidence blocks

---

## 16. Redis Caching Strategy for Retrieval

Add caching for these retrieval layers.

## 16.1 Query result cache

Key:

- `search:paper:{query_hash}`

Value:

- final top candidate ids

Use for:

- repeated questions
- repeated benchmark evaluation

## 16.2 Section tree cache

Key:

- `paper:sections:{paper_id}`

Use for:

- section lookup
- section title matching

## 16.3 Chunk neighbor cache

Key:

- `chunk:neighbors:{chunk_id}`

Use for:

- fast context expansion

## 16.4 Figure/table summary cache

Keys:

- `paper:table_summary:{table_id}`
- `paper:figure_summary:{figure_id}`

Use for:

- context hydration
- repeated citation assembly

---

## 17. Evaluation Plan

The retrieval redesign must be measurable.

Track these metrics:

- Recall@K
- Precision@K
- MRR
- nDCG
- hit rate by query type
- latency by stage
- answer grounding rate

Recommended evaluation buckets:

- summary questions
- method questions
- experiment questions
- metric lookup questions
- table questions
- figure questions

Suggested latency breakdown:

- query parsing time
- SQL prefilter time
- vector recall time
- lexical recall time
- rerank time
- context packing time

---

## 18. Recommended Config Additions

Add config options for:

- `RAG_PAPER_VECTOR_TOP_K`
- `RAG_PAPER_BM25_TOP_K`
- `RAG_PAPER_RERANK_TOP_K`
- `RAG_PAPER_RERANK_FETCH_K`
- `RAG_PAPER_ENABLE_MULTI_QUERY`
- `RAG_PAPER_ENABLE_QUERY_DECOMPOSITION`
- `RAG_PAPER_ENABLE_TABLE_RECALL`
- `RAG_PAPER_ENABLE_FIGURE_RECALL`
- `RAG_PAPER_ENABLE_PARENT_CHILD_RETRIEVAL`
- `RAG_PAPER_ENABLE_CONTEXT_EXPANSION`
- `RAG_PAPER_CONTEXT_EXPANSION_DEPTH`
- `RAG_PAPER_ENABLE_MMR_PACKING`
- `RAG_PAPER_SEARCH_PROFILE`

---

## 19. Code Refactor Targets

## 19.1 `tools/rag/knowledge.py`

Refactor into a true retrieval facade with:

- vector recall entry points
- hybrid recall entry points
- filtering support by `paper_id`, `section_id`, `chunk_role`
- no oversized metadata assumptions

## 19.2 New `tools/retrieval/paper_retriever.py`

Create a dedicated paper retrieval orchestrator.

Responsibilities:

- query analysis
- prefilter generation
- recall orchestration
- fusion
- reranking
- context expansion
- evidence packaging

## 19.3 New `tools/retrieval/query_understanding.py`

Responsibilities:

- query intent classification
- section hints
- figure/table hint extraction
- multi-query rewrite
- query decomposition

## 19.4 New `tools/retrieval/fusion.py`

Responsibilities:

- RRF
- weighted fusion
- deduplication

## 19.5 New `tools/retrieval/context_packing.py`

Responsibilities:

- neighbor expansion
- MMR packing
- role diversity
- token budget packing

## 19.6 `tools/agent/paper_ingest.py`

Must generate:

- chunk roles
- summary chunks
- table summaries
- figure summaries

## 19.7 `main.py`

Expose paper retrieval controls through API:

- strategy
- profile
- exact paper id
- section filter
- table-only or figure-only retrieval modes

---

## 20. Immediate Implementation Order for Cursor

Cursor should implement in this order:

1. create `tools/retrieval/` package
2. implement query understanding layer
3. implement SQL prefilter builder
4. implement vector recall wrapper
5. implement BM25 / FTS recall wrapper
6. implement table recall
7. implement figure recall
8. implement fusion layer
9. implement rerank layer integration
10. implement context expansion and packing
11. wire paper retrieval into `agent.py`
12. add config flags
13. add evaluation scripts

---

## 21. Recommended Default Production Strategy

For production default paper retrieval:

1. query understanding
2. SQL prefilter
3. multi-query generation
4. vector recall
5. BM25 recall
6. table or figure recall if applicable
7. RRF fusion
8. rerank top 40
9. expand neighbors
10. MMR pack final evidence

This should be the default retrieval profile for paper-focused RAG.

---

## 22. Non-Negotiable Rules

- Do not use vector-only retrieval as the default for paper QA.
- Do not ignore tables and figures in paper retrieval.
- Do not skip reranking for difficult paper queries.
- Do not pack raw top-k chunks directly into the prompt without neighbor/context logic.
- Do not let conversation memory dominate paper evidence.
- Always preserve citation traceability to section/page/object.

---

## 23. Expected Outcome

After implementing this plan:

- paper retrieval precision should improve significantly
- difficult paper questions should become more answerable
- table and figure questions should stop being blind spots
- repeated queries should become faster
- context fed to the LLM should become more relevant and less noisy

