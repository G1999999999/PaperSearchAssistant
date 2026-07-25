# PaperSearchAssistant2 Section 感知存储与检索方案

## 1. 目标

这份文档用于指导 `PaperSearchAssistant2` 把论文存储与检索升级为：

- 按 section 建结构
- 按 chunk 做召回
- chunk 绑定 section
- 检索时支持 section 过滤
- 回答时支持 section 扩展
- 同时保留“全文级问题”的回答能力

核心目标不是把系统改成“只能按 section 检索”，而是让系统既能：

- 精确回答局部问题
- 也能回答全局问题

---

## 2. 核心原则

## 2.1 不要只按 chunk 存

如果只有 chunk，没有 section 结构：

- 很难稳定回答“方法是什么”
- 很难自动补 `Method` 章节
- 很难识别某个 chunk 属于论文哪一部分

## 2.2 也不要只按 section 存

如果只按 section 存：

- 每个 section 太大
- 检索粒度太粗
- 容易引入太多无关内容

## 2.3 最佳方案：section + chunk 双层建模

推荐做法：

- section 是结构层
- chunk 是召回层
- 回答时可根据问题类型选择：
  - 只用局部 chunk
  - 扩展到整个 section
  - 汇总多个 section 回答全文问题

---

## 3. 数据建模方案

## 3.1 section 层

建议使用 `paper_sections` 表保存：

- `section_id`
- `paper_id`
- `parent_section_id`
- `section_level`
- `section_number`
- `section_title`
- `page_start`
- `page_end`
- `order_index`

典型 section 包括：

- Abstract
- Introduction
- Related Work
- Method
- Experiment
- Results
- Conclusion
- Appendix

## 3.2 chunk 层

建议使用 `paper_chunks` 表保存：

- `chunk_id`
- `paper_id`
- `section_id`
- `chunk_index`
- `chunk_role`
- `content`
- `summary_text`
- `prev_chunk_id`
- `next_chunk_id`
- `page_from`
- `page_to`
- `chroma_doc_id`

关键点：

- 每个 chunk 必须绑定 `section_id`
- chunk 仍然是检索的主要单位

## 3.3 section summary 层

建议为每个 section 额外生成一个 summary：

- `section_summary`
- `section_keywords`
- `section_role`

这层主要服务于：

- section 级预筛选
- 全文型问题摘要拼装
- 回答时的 parent context

---

## 4. 检索的三种粒度

引入 section 感知后，系统应支持三种粒度检索：

## 4.1 chunk 级检索

适合：

- 方法细节
- 实验细节
- 某个局部定义
- 某段具体解释

特点：

- 精确
- 噪音少
- 是默认召回层

## 4.2 section 级检索

适合：

- 方法整体介绍
- 实验章节整体内容
- 结论章节整体理解

特点：

- 比 chunk 更完整
- 比全文更聚焦

## 4.3 paper 级检索

适合：

- 这篇论文讲了什么
- 这篇论文主要贡献是什么
- 这篇论文整体思路是什么

特点：

- 需要汇总多个 section
- 不应该只靠单个 chunk 回答

---

## 5. 如何处理“全文型问题”

这是本方案最重要的补充。

像这样的问题：

- 这篇论文讲了什么
- 这篇论文主要贡献是什么
- 这篇论文整体思路是什么

确实不能只依赖某一个 section，也不能只依赖某几个随机 chunk。

因此，系统必须支持“全文级汇总检索”。

## 5.1 全文型问题不等于整篇全文灌入

虽然这类问题需要结合全文，但不意味着要把整篇论文所有 chunk 全部读出来。

更合理的做法是：

- 选择有代表性的 section
- 从这些 section 中挑高质量证据
- 再做汇总

## 5.2 全文型问题的推荐 section

对于“这篇论文讲了什么”，优先使用这些 section：

- Abstract
- Introduction
- Method summary
- Experiment / Result summary
- Conclusion

也就是说：

- 不是整篇全灌
- 而是“全篇代表性 section 汇总”

## 5.3 推荐的全文回答策略

对于全文型问题，建议：

1. 先取 abstract chunk
2. 再取 introduction 中最相关 chunk
3. 再取 method summary chunk
4. 再取 experiment / result summary chunk
5. 最后取 conclusion chunk

这样可以形成：

- 论文做什么
- 怎么做
- 结果如何
- 最终结论

的完整回答框架。

---

## 6. Section 感知检索策略

## 6.1 局部问题：先按 section 限定

当用户问题具有明确的 section 倾向时，应优先按 section 检索。

例如：

### 方法问题

- 这篇论文的方法是什么
- 模型结构怎么设计的

优先 section：

- Method
- Approach
- Model
- Architecture

### 实验问题

- 实验结果怎么样
- 在哪些数据集上做了测试

优先 section：

- Experiment
- Evaluation
- Result
- Ablation

### 结论问题

- 论文结论是什么

优先 section：

- Conclusion
- Discussion

## 6.2 全文问题：跨 section 汇总

当用户问：

- 这篇论文讲了什么
- 整体贡献是什么

则不要只限制到单个 section。

应采用：

- 多个代表性 section 共同召回

推荐 section 组合：

- Abstract
- Introduction
- Method summary
- Experiment summary
- Conclusion

## 6.3 表格和图片问题：对象优先 + section 补充

例如：

- Table 3 说明了什么
- Figure 2 讲了什么

做法：

1. 先召回表格 / 图片对象
2. 再补该对象所在 section 的正文解释

---

## 7. 推荐的检索链路

## 7.1 局部问题链路

适合：

- 方法是什么
- 实验结果如何
- 结论是什么

推荐链路：

1. 意图识别
2. 确定目标 section
3. 仅在相关 section chunk 中做粗排
4. reranker 精排
5. 补邻近 chunk
6. 必要时补整个 section

## 7.2 全文问题链路

适合：

- 这篇论文讲了什么
- 这篇论文的整体贡献是什么

推荐链路：

1. 意图识别为 `paper_summary`
2. 不限制单个 section
3. 从代表性 section 中分别召回
4. 对各 section 结果做融合
5. reranker 精排
6. 按“摘要-方法-结果-结论”组织回答

## 7.3 对象问题链路

适合：

- Table 2 说明了什么
- Figure 3 的结构图是什么意思

推荐链路：

1. 对象检索优先
2. 命中对象后补所在 section 正文
3. 再融合回答

---

## 8. 为什么 section 感知比纯 chunk 更好

引入 section 感知后，可以获得这些能力：

## 8.1 支持定向检索

用户问方法时，不必在整篇论文全局乱搜。

## 8.2 支持缺失补全

如果 `Method` 没命中，可以直接补整个 `Method` section。

## 8.3 支持更自然的全文总结

对于“这篇论文讲了什么”，可以从多个关键 section 中抽代表证据，而不是随机 top-k。

## 8.4 支持更好的证据引用

最终可以告诉用户：

- 方法见第 3 节
- 结论见第 6 节
- 图 2 位于方法部分

---

## 9. 推荐的 Parent-Child Retrieval

这套方案和 section 感知非常适合一起使用。

## 9.1 child

小 chunk，作为召回单位。

优点：

- 检索精确
- 粒度细

## 9.2 parent

section summary 或 section 内更大范围上下文。

优点：

- 回答完整
- 能补足局部片段的上下文

## 9.3 推荐模式

推荐：

- 用 child chunk 检索
- 用 parent section 回答

这比：

- 只用 chunk
- 或只用整节

都更平衡。

---

## 10. 推荐新增的能力

## 10.1 section-aware prefilter

根据问题意图先筛 section，再检索 chunk。

## 10.2 section summary retrieval

支持直接检索 section summary。

## 10.3 paper summary assembler

对全文型问题，自动从多个关键 section 拼论文总览。

## 10.4 section fallback

当前证据不足时，自动补目标 section。

---

## 11. 推荐新增模块

建议新增：

- `tools/retrieval/section_selector.py`
- `tools/retrieval/section_summary_builder.py`
- `tools/retrieval/paper_summary_assembler.py`

职责如下：

## 11.1 `section_selector.py`

- 根据问题意图选择相关 section

## 11.2 `section_summary_builder.py`

- 在入库时生成各个 section summary

## 11.3 `paper_summary_assembler.py`

- 针对“这篇论文讲了什么”这类问题
- 从多个 section 汇总上下文

---

## 12. 推荐修改的现有模块

## 12.1 `paper_ingest.py`

改造要求：

- 解析并保存 section
- chunk 必须绑定 section
- 生成 section summary

## 12.2 `paper_retriever.py`

改造要求：

- 支持按 section 检索
- 支持跨 section 汇总检索

## 12.3 `paper_content_qa.py`

改造要求：

- 对局部问题走 section-aware 检索
- 对全文问题走 multi-section summary 检索

---

## 13. 给 Cursor 的实施顺序

建议 Cursor 按以下顺序实施：

1. 新增 `paper_sections`
2. 让 chunk 绑定 `section_id`
3. 生成 `section_summary`
4. 新增 section-aware 检索逻辑
5. 增加全文型问题的 multi-section 汇总逻辑
6. 增加 section fallback

---

## 14. 最重要的结论

你完全可以按 section 来建模论文，但推荐的方式不是：

- 只按 section 存

而是：

- **按 section 建结构**
- **按 chunk 做召回**
- **chunk 绑定 section**
- **局部问题按 section 限定**
- **全文问题跨多个关键 section 汇总**

所以，对于：

- `这篇论文讲了什么`

这样的全文型问题，正确做法不是：

- 只查一个 chunk
- 或直接整篇全读

而是：

- **抽取 Abstract + Introduction + Method Summary + Result Summary + Conclusion**

来做论文级回答。

