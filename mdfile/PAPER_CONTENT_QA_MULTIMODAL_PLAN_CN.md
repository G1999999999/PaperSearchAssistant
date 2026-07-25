# PaperSearchAssistant2 论文内容问答与多模态检索方案

## 1. 目标

这份文档专门用于指导 `PaperSearchAssistant2` 的论文内容问答能力重构。

这里的重点不是“找论文”，而是“用户已经在问某篇论文的内容”时，系统如何处理：

- 论文正文
- 表格
- 图片 / 图表

目标是建立一套完整链路，使系统能够回答：

- 这篇论文的方法是什么
- 这篇论文实验结果怎么样
- Table 3 讲了什么
- Figure 2 的结构图表示什么
- 这篇论文里某个指标是多少

本方案的核心思想是：

**论文内容问答必须采用多通道检索，不应把正文、表格、图片全部混成一种检索对象。**

---

## 2. 总体思路

当用户问论文内容时，系统需要把论文视为三类证据源：

1. 文字正文证据
2. 表格证据
3. 图片 / 图表证据

这三类证据各自有不同的：

- 存储方式
- 解析方式
- 检索方式
- 融合方式
- 回答组织方式

因此，需要设计一个“多通道问答链路”。

推荐的总体流程如下：

```text
用户问题
  -> 路由判断（正文 / 表格 / 图片 / 混合）
  -> 定位目标论文
  -> 并行发起三类召回：
       - 正文召回
       - 表格召回
       - 图片召回
  -> 多路融合
  -> reranker 重排
  -> 证据扩展
  -> 上下文打包
  -> LLM 回答生成
```

---

## 3. 用户问题分类

系统需要先判断用户在问哪类内容。

## 3.1 正文问题

典型问题：

- 这篇论文的方法是什么
- 这篇论文的核心贡献是什么
- 作者是怎么做实例分组的
- 论文结论是什么

这类问题以正文检索为主。

## 3.2 表格问题

典型问题：

- Table 2 说明了什么
- 表 3 里哪个模型最好
- 这个指标提升了多少
- 论文中的 F1 分数是多少

这类问题应优先走表格召回。

## 3.3 图片 / 图表问题

典型问题：

- Figure 3 讲了什么
- 图 2 的结构图是什么意思
- 这张图反映了什么趋势
- 模型架构图如何工作

这类问题应优先走图片 / 图表召回。

## 3.4 混合问题

典型问题：

- 这篇论文的方法是什么，实验结果如何
- Figure 2 的结构图和 Table 3 的结果能说明什么

这类问题需要正文、表格、图片联合召回。

---

## 4. 正文问答方案

## 4.1 正文存储

正文数据应来源于：

- `paper_sections`
- `paper_blocks`
- `paper_chunks`

正文的主检索单位应为：

- `paper_chunks`

每个 chunk 应具备：

- `chunk_role`
- `section_id`
- `page range`
- `paper_id`

## 4.2 正文切片策略

推荐规则：

- 结构化切片，不使用单纯固定 token 切片
- 每个 chunk 长度目标为 `350-650 tokens`
- overlap 使用 block overlap，而不是机械 token overlap
- 标题和后续段落尽量绑定
- 结论和摘要单独切片

建议 chunk role：

- `abstract`
- `intro`
- `method`
- `experiment`
- `result`
- `conclusion`
- `appendix`
- `generic`

## 4.3 正文检索方式

正文检索采用混合 RAG：

- 向量检索
- BM25 / FTS
- reranker

推荐流程：

1. 根据问题识别可能 section
2. SQL 预过滤对应的 chunk
3. 用 Chroma 做向量召回
4. 用 PostgreSQL FTS / BM25 做关键词召回
5. 结果融合
6. reranker 重排
7. 相邻 chunk 扩展

## 4.4 正文问题适合回答的内容

适合回答：

- 方法
- 贡献
- 总体思想
- 实验设置
- 对比分析
- 局限性

---

## 5. 表格问答方案

## 5.1 表格必须独立建模

表格不能只嵌在正文里处理，否则系统会：

- 漏掉关键数值
- 无法稳定回答“哪个最好”
- 无法稳定回答“提升了多少”

因此，表格必须作为一等实体单独存储。

推荐存储表：

- `paper_tables`

推荐字段包括：

- 表号
- 标题
- caption
- summary
- markdown
- json
- csv

## 5.2 表格的多种表示

每张表推荐保留以下表示方式：

1. 原始表格截图
2. 结构化 JSON
3. Markdown 表示
4. 表格摘要

其中最适合检索的是：

- 表格摘要

最适合精确数值回答的是：

- JSON / Markdown

## 5.3 表格摘要应如何生成

表格摘要建议描述：

- 这张表在比较什么
- 核心列有哪些
- 哪个模型最好
- 与 baseline 差多少
- 哪个指标最关键

例如：

```text
Table 3 比较了不同模型在 ScanNet 上的实例分组结果。
列包括方法、mIoU、AP 和推理速度。
Gaussian Grouping 在 mIoU 上取得最佳结果 78.4，相比 baseline 提高 3.2 个点。
```

## 5.4 表格检索方式

表格问题建议使用以下召回方式：

- caption 关键词检索
- summary 向量检索
- metric name 关键词检索
- table number 精确匹配

推荐加权：

- 如果问题中出现 `Table` / `表`
- 或者出现 `F1`、`mIoU`、`accuracy` 等指标词

则：

- 提高 table recall 权重
- 提高 table chunk 排名

## 5.5 表格问答时如何组织上下文

表格命中后，给模型的上下文应优先包含：

1. 表格编号
2. 表格标题 / caption
3. 表格摘要
4. 关键数值
5. 必要时附加部分 markdown 行列

不要直接把超大表原样塞进 prompt。

---

## 6. 图片 / 图表问答方案

## 6.1 图片 / 图表必须独立建模

图片和图表里经常包含正文之外的重要信息，例如：

- 模型结构图
- 算法流程图
- 结果趋势图
- 定性可视化

如果只检索正文，系统很容易回答不完整。

因此，图片 / 图表必须独立建模。

推荐表：

- `paper_figures`

推荐字段包括：

- 图号
- caption
- OCR 文本
- vision summary
- keywords
- 图片路径

## 6.2 图片 / 图表的文本化方案

为了支持检索，需要把图片变成文本化证据。

建议每张图构造以下内容：

- 图号
- 图注
- OCR 文本
- 视觉摘要
- 关键词

例如：

```text
Figure 2. Overall architecture.
该图展示了一个包含 query encoder、document encoder 和 reranker 的双塔检索结构。
OCR 文本包括 top-k retrieval、cross-attention、final answer。
关键词：architecture, retrieval, reranker, dual encoder。
```

## 6.3 图片 / 图表检索方式

推荐召回方式：

- caption 关键词检索
- vision summary 向量检索
- OCR 文本关键词检索
- figure number 精确匹配

如果问题中出现：

- `Figure`
- `图`
- `结构图`
- `流程图`
- `图 2`

则应：

- 提高 figure recall 权重
- 提高图像摘要类证据优先级

## 6.4 图片 / 图表问答时如何组织上下文

给模型的上下文建议包含：

1. 图号
2. 图注
3. 视觉摘要
4. OCR 关键文本
5. 必要时附加对应正文段落

特别是结构图问题，正文和图注往往需要一起给。

---

## 7. 多通道召回设计

## 7.1 为什么要多通道

论文问答不是单一文本检索问题。

一个问题可能同时依赖：

- 正文解释
- 表格数值
- 图片结构

如果只查正文，很多答案会缺失。

## 7.2 三类召回通道

建议并行维护三类召回：

### 正文召回通道

数据源：

- `paper_chunks`

方法：

- 向量检索
- BM25 / FTS
- reranker

### 表格召回通道

数据源：

- `paper_tables`
- `table summary chunks`

方法：

- caption 检索
- summary 向量检索
- 指标名 BM25

### 图片召回通道

数据源：

- `paper_figures`
- `figure summary chunks`

方法：

- caption 检索
- OCR 检索
- vision summary 向量检索

---

## 8. 多通道融合方案

三类召回结果不能直接拼一起用，必须融合和去重。

建议流程：

1. 各通道分别返回 top-k
2. 标注每条结果来源：
   - `text`
   - `table`
   - `figure`
3. 使用 RRF 或加权融合
4. 再送入 reranker
5. 最终保留若干条高质量证据

推荐融合规则：

- 正文问题：正文通道权重最高
- 表格问题：表格通道权重最高
- 图片问题：图片通道权重最高
- 混合问题：三通道较均衡

---

## 9. 重排序方案

多通道召回后，必须做统一 rerank。

推荐输入格式：

```text
SourceType: table
Paper: {paper_title}
Section: {section_title}
Object: Table 3
Content: {summary_or_chunk_text}
```

对于不同来源：

- 正文：内容是 chunk 文本
- 表格：内容是表格摘要
- 图片：内容是图像摘要

统一交给 cross-encoder reranker。

推荐做法：

- 初始召回总量：`30-60`
- rerank 后保留：`5-10`

---

## 10. 上下文扩展策略

## 10.1 正文扩展

正文证据命中后：

- 取当前 chunk
- 取前一个 chunk
- 取后一个 chunk

用于补齐上下文。

## 10.2 表格扩展

表格证据命中后：

- 取 table summary
- 取 table caption
- 取关键数值
- 如有必要，附加少量 markdown

## 10.3 图片扩展

图片证据命中后：

- 取 figure summary
- 取 caption
- 取 OCR 关键词
- 如有必要，附加相邻正文说明

---

## 11. 最终回答生成策略

## 11.1 回答不能只堆证据

模型最终回答时，应按照内容类型组织，而不是简单堆 chunk。

例如：

### 正文问题

优先输出：

- 核心结论
- 方法步骤
- 关键机制
- 必要时引用表格或图片作补充

### 表格问题

优先输出：

- 结论
- 核心数字
- 对比对象
- 指标提升情况

### 图片问题

优先输出：

- 图表达了什么
- 图中模块或流程如何工作
- 图与正文的对应关系

## 11.2 引用方式建议

建议最终证据中保留：

- 论文标题
- section
- page
- Table 编号
- Figure 编号

这样回答可追溯性更强。

---

## 12. 推荐默认行为

## 12.1 用户问正文内容

例如：

- `gaussian-grouping 的方法是什么`

默认：

- 主走正文通道
- 辅助表格/图片通道

## 12.2 用户问表格内容

例如：

- `Table 3 结果如何`

默认：

- 主走表格通道
- 辅助正文通道

## 12.3 用户问图片内容

例如：

- `Figure 2 讲了什么`

默认：

- 主走图片通道
- 辅助正文通道

## 12.4 用户问综合问题

例如：

- `这篇论文的方法和实验结果说明了什么`

默认：

- 正文、表格、图片三通道都参与
- 最后统一重排和上下文打包

---

## 13. 推荐新增模块

建议新增：

- `tools/retrieval/paper_content_qa.py`
- `tools/retrieval/table_retriever.py`
- `tools/retrieval/figure_retriever.py`
- `tools/retrieval/context_assembler.py`

职责如下：

### `paper_content_qa.py`

- 问答总控
- 调度正文、表格、图片三通道
- 合并结果

### `table_retriever.py`

- 表格召回
- caption 检索
- 指标词召回
- summary 检索

### `figure_retriever.py`

- 图片 / 图表召回
- caption 检索
- OCR 检索
- vision summary 检索

### `context_assembler.py`

- 最终上下文组织
- 去重
- token 控制
- 证据排序

---

## 14. 推荐修改的现有模块

## 14.1 `tools/agent/paper_ingest.py`

改造目标：

- 不只是 PDF 入向量库
- 要生成：
  - sections
  - blocks
  - chunks
  - tables
  - figures
  - table summary
  - figure summary

## 14.2 `tools/rag/knowledge.py`

改造目标：

- 不再只有泛化 chunk 检索
- 需要支持：
  - text retrieval
  - table retrieval
  - figure retrieval

## 14.3 `agent.py`

改造目标：

- 根据路由结果决定走哪类问答流程
- 文本、表格、图片不能都用同一种 prompt 组装方式

## 14.4 `prompts.py`

改造目标：

- 增加：
  - 正文问答 prompt
  - 表格问答 prompt
  - 图片问答 prompt
  - 多模态混合问答 prompt

---

## 15. 推荐配置项

建议新增：

- `RAG_PAPER_ENABLE_TABLE_QA`
- `RAG_PAPER_ENABLE_FIGURE_QA`
- `RAG_PAPER_TABLE_TOP_K`
- `RAG_PAPER_FIGURE_TOP_K`
- `RAG_PAPER_TEXT_TOP_K`
- `RAG_PAPER_MULTIMODAL_RERANK_TOP_K`
- `RAG_PAPER_ENABLE_CONTEXT_EXPANSION`
- `RAG_PAPER_ENABLE_MULTIMODAL_PACKING`

---

## 16. 给 Cursor 的实施顺序

建议 Cursor 按以下顺序实施：

1. 重构论文入库，新增表格和图片结构化存储
2. 新增 `table_retriever.py`
3. 新增 `figure_retriever.py`
4. 新增 `paper_content_qa.py`
5. 改造正文检索入口
6. 改造 `agent.py`，接入多通道检索
7. 改造 `prompts.py`，支持多模态问答
8. 增加日志和评测

---

## 17. 最重要的结论

论文内容问答不能只靠正文 chunk 检索。

要想把论文问答做好，必须建立这套结构：

1. 正文单独建模
2. 表格单独建模
3. 图片 / 图表单独建模
4. 三通道分别召回
5. 最后统一融合、重排、扩展和回答

最终推荐方案不是：

- “只做文本 RAG”

而是：

- **正文 RAG + 表格 RAG + 图片 / 图表 RAG + 多通道融合**

这才是适合论文问答场景的完整方案。

