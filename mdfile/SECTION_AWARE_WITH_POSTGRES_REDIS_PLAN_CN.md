# PaperSearchAssistant2 Section 感知存储检索与 PostgreSQL / Redis 结合方案

## 1. 文档目标

这份文档用于把：

- `SECTION_AWARE_STORAGE_AND_RETRIEVAL_PLAN_CN.md`

中的 section 感知存储与检索方案，进一步和：

- PostgreSQL
- Redis

结合起来，形成可落地的实现方案。

目标是让 Cursor 明确知道：

- section-aware 结构应该落在哪些 PostgreSQL 表里
- 哪些信息适合放到 Redis 做缓存
- 检索时 PostgreSQL 和 Redis 分别承担什么角色
- 如何支持“局部问题按 section 检索”和“全文问题跨 section 汇总”

---

## 2. 核心结论

Section-aware 方案要想真正落地，最佳做法是：

- PostgreSQL 负责存储 section 结构和 chunk 结构
- Redis 负责缓存 section 树、section summary、chunk 邻接和论文总览
- Chroma 负责向量召回

换句话说：

- **PostgreSQL 是 section-aware 的主存储**
- **Redis 是 section-aware 的加速层**
- **Chroma 是 section-aware 的语义召回层**

---

## 3. PostgreSQL 中如何建模 section-aware 结构

## 3.1 `papers`

保存论文主信息：

- `id`
- `title`
- `abstract`
- `authors_json`
- `published_at`
- `pdf_path`
- `source_url`

用途：

- 论文入口
- 论文详情
- 论文候选匹配

## 3.2 `paper_sections`

这是 section-aware 的核心表。

必须保存：

- `id`
- `paper_id`
- `parent_section_id`
- `section_level`
- `section_number`
- `title`
- `title_norm`
- `page_start`
- `page_end`
- `order_index`

作用：

- 表示论文目录结构
- 支持按 section 检索
- 支持 section fallback

## 3.3 `paper_chunks`

这是检索的核心表。

每个 chunk 必须绑定：

- `chunk_id`
- `paper_id`
- `section_id`
- `chunk_role`
- `chunk_index`
- `content`
- `summary_text`
- `page_from`
- `page_to`
- `prev_chunk_id`
- `next_chunk_id`
- `chroma_doc_id`

作用：

- 小粒度召回
- 绑定 section
- 支持 chunk 邻近扩展

## 3.4 `paper_sections_summary`

建议单独增加一张 section summary 表，或者把 summary 字段加到 `paper_sections`。

建议字段：

- `section_id`
- `paper_id`
- `section_role`
- `summary_text`
- `keywords_json`
- `updated_at`

作用：

- section-aware 预筛选
- 全文问题多 section 汇总
- parent context

## 3.5 `paper_summary_views`

建议增加论文级汇总表，或逻辑视图。

建议保存：

- `paper_id`
- `abstract_summary`
- `intro_summary`
- `method_summary`
- `result_summary`
- `conclusion_summary`
- `updated_at`

作用：

- 回答“这篇论文讲了什么”
- 避免每次现拼全文摘要

---

## 4. Redis 在 section-aware 中的作用

Redis 不是主存储，而是加速层。

## 4.1 缓存 section 树

推荐 key：

- `paper:sections:{paper_id}`

缓存内容：

- section 列表
- 标题
- section 层级
- page 范围
- section 顺序

用途：

- 用户问方法时快速找到 Method section
- 避免每次查 PostgreSQL

## 4.2 缓存 section summary

推荐 key：

- `paper:section_summary:{section_id}`

缓存内容：

- section title
- summary_text
- keywords
- section role

用途：

- 预筛选
- 全文问题汇总
- section fallback

## 4.3 缓存 chunk 邻接关系

推荐 key：

- `chunk:neighbors:{chunk_id}`

缓存内容：

- `prev_chunk_id`
- `next_chunk_id`
- 同一 section 下的邻近 chunk id

用途：

- 命中 chunk 后快速补前后文
- 缺失证据补全

## 4.4 缓存 paper summary bundle

推荐 key：

- `paper:summary_bundle:{paper_id}`

缓存内容：

- abstract summary
- intro summary
- method summary
- result summary
- conclusion summary

用途：

- 回答全文型问题
- 快速组织论文整体概览

## 4.5 缓存 section selection hint

推荐 key：

- `paper:section_roles:{paper_id}`

缓存内容：

- 哪些 section 属于：
  - method
  - experiment
  - result
  - conclusion

用途：

- 让 section-aware prefilter 更快

---

## 5. PostgreSQL、Redis、Chroma 三者分工

推荐明确分工如下：

## 5.1 PostgreSQL

负责：

- 论文结构化事实存储
- section、chunk、summary、对象关系
- 过滤与回填

## 5.2 Redis

负责：

- section 树缓存
- summary 缓存
- 邻接关系缓存
- 当前论文阅读态缓存

## 5.3 Chroma

负责：

- chunk 的向量检索
- section summary 的向量检索
- 表格 / 图片 summary 的向量检索

---

## 6. 检索时 PostgreSQL 和 Redis 怎么配合

## 6.1 局部问题：按 section 检索

例如：

- 这篇论文的方法是什么

推荐流程：

1. 路由识别出 `sub_intent = method`
2. 先从 Redis 读取：
   - `paper:sections:{paper_id}`
   - `paper:section_roles:{paper_id}`
3. 找到属于 `method` 的 section_id 列表
4. 用 PostgreSQL 进一步确认这些 section 对应 chunk
5. 在这些 chunk 上做 Chroma 向量召回 + BM25
6. 命中后从 Redis 读取 chunk 邻接关系补上下文

## 6.2 全文问题：跨多个 section 汇总

例如：

- 这篇论文讲了什么

推荐流程：

1. 识别为 `paper_summary`
2. 先尝试从 Redis 读取：
   - `paper:summary_bundle:{paper_id}`
3. 如果缓存存在，直接作为高优先级上下文
4. 如果缓存不存在，则从 PostgreSQL 读取：
   - abstract section summary
   - intro section summary
   - method summary
   - result summary
   - conclusion summary
5. 如有必要，再从 Chroma 召回对应 chunk 做补充

## 6.3 缺失证据时的 fallback

例如：

- 当前 top-k 里没有 Method 章节

推荐流程：

1. 从 Redis 快速拿 method section 列表
2. 从 PostgreSQL 读取该 section 的 chunk
3. 必要时再走 Chroma 做 section 内精排

也就是说：

- Redis 负责快速定位
- PostgreSQL 负责精确读取

---

## 7. 全文型问题如何结合 PostgreSQL 和 Redis

对于：

- 这篇论文讲了什么
- 这篇论文主要贡献是什么

不能只依赖一个 chunk，也不能整篇全文灌入。

推荐的实现方式是：

## 7.1 先从 Redis 取摘要包

如果有：

- `paper:summary_bundle:{paper_id}`

则优先取出：

- abstract summary
- intro summary
- method summary
- result summary
- conclusion summary

## 7.2 没有缓存则去 PostgreSQL 组装

从 PostgreSQL 中读取：

- 对应 section summary
- 必要时补一些关键 chunk

## 7.3 最后才向量补充

如果：

- summary 不足
- 问题更细

再调用 Chroma 进行向量召回和补充。

这个顺序比“每次都重新全文检索”更高效。

---

## 8. 推荐的 PostgreSQL 查询能力

为了支持 section-aware，PostgreSQL 需要支持这些查询：

## 8.1 查某篇论文的 section 树

输入：

- `paper_id`

输出：

- 全部 section
- section 层级
- 顺序

## 8.2 查某一类 section

输入：

- `paper_id`
- `section_role = method/result/conclusion`

输出：

- 匹配的 section 列表

## 8.3 查某个 section 的 chunk

输入：

- `section_id`

输出：

- 该 section 所有 chunk

## 8.4 查 chunk 邻接关系

输入：

- `chunk_id`

输出：

- prev / next / nearby chunk

## 8.5 查论文 summary bundle

输入：

- `paper_id`

输出：

- 多 section 摘要包

---

## 9. 推荐的 Redis 使用时机

## 9.1 在用户打开论文时

预热以下缓存：

- section tree
- section roles
- summary bundle

## 9.2 在论文入库完成时

预热以下缓存：

- section tree
- chunk neighbor cache
- paper summary bundle

## 9.3 在第一次回答全文问题后

将整理好的论文总览写入：

- `paper:summary_bundle:{paper_id}`

## 9.4 在 chunk 命中后

把 chunk 邻接结果缓存起来，供后续连续问答使用。

---

## 10. 推荐的能力边界

## 10.1 PostgreSQL 不做什么

PostgreSQL 不负责：

- 向量相似度召回
- 大规模 embedding 检索

## 10.2 Redis 不做什么

Redis 不负责：

- 永久真相存储
- 复杂结构化查询

## 10.3 Chroma 不做什么

Chroma 不负责：

- section 树结构
- 当前阅读态
- 复杂过滤主逻辑

---

## 11. 推荐新增模块

建议新增：

- `tools/storage/repos/section_repo.py`
- `tools/storage/repos/chunk_repo.py`
- `tools/storage/repos/summary_repo.py`
- `tools/storage/redis/section_cache.py`
- `tools/storage/redis/chunk_cache.py`

职责如下：

## 11.1 `section_repo.py`

- PostgreSQL 中 section 的读取与过滤

## 11.2 `chunk_repo.py`

- PostgreSQL 中 chunk 的读取、邻接扩展

## 11.3 `summary_repo.py`

- PostgreSQL 中 section summary、paper summary bundle 的读取

## 11.4 `section_cache.py`

- section tree 缓存
- section role 缓存

## 11.5 `chunk_cache.py`

- chunk 邻接缓存
- chunk context 缓存

---

## 12. 推荐修改的现有模块

## 12.1 `paper_ingest.py`

必须做到：

- 把解析出的 section 写入 PostgreSQL
- chunk 写入 PostgreSQL 并绑定 section_id
- 生成 section summary
- 预热 Redis 缓存

## 12.2 `paper_retriever.py`

必须做到：

- 先根据 Redis 判断可用 section
- 再根据 PostgreSQL 过滤 chunk
- 再走 Chroma 检索

## 12.3 `paper_summary_assembler.py`

必须做到：

- 优先读 Redis summary bundle
- 否则从 PostgreSQL 组装

---

## 13. 推荐给 Cursor 的实施顺序

建议 Cursor 按以下顺序实施：

1. 在 PostgreSQL 中落地 `paper_sections` 和 `paper_chunks`
2. 给 chunk 增加 `section_id`
3. 增加 section summary 表或字段
4. 增加 Redis section tree 缓存
5. 增加 Redis chunk neighbor 缓存
6. 增加 Redis paper summary bundle 缓存
7. 重构检索链路，接入 PostgreSQL + Redis + Chroma

---

## 14. 最重要的结论

Section-aware 方案和 PostgreSQL / Redis 的最佳结合方式是：

- **PostgreSQL 存结构**
- **Redis 存热点和上下文加速**
- **Chroma 做语义召回**

其中：

- 局部问题依赖 `section + chunk` 精确检索
- 全文问题依赖 `多 section summary + 少量关键 chunk` 汇总

因此，最终推荐方案不是：

- 只按 chunk 检索
- 或只按 section 检索

而是：

- **PostgreSQL 中按 section + chunk 双层建模**
- **Redis 中缓存 section tree / section summary / chunk neighbor**
- **Chroma 中保留 chunk 与 summary 的向量索引**

这才是适合论文阅读场景的 section-aware 实现方式。

