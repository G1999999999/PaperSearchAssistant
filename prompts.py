"""
RAG 相关 Prompt 模板。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from config import (
    RAG_CHAT_HISTORY_MAX_IMAGE_TURNS,
    RAG_CONTEXT_MAX_CHARS_PER_DOC,
    RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC,
    RAG_CONTEXT_PAPER_METHOD_TOTAL_MAX_CHARS,
    RAG_CONTEXT_TOTAL_MAX_CHARS,
    RAG_MAX_IMAGES_PER_MESSAGE,
    RAG_USE_COMPACT_RAG_USER_PROMPT,
    RAG_USER_UPLOAD_MAX_IMAGES,
)
from tools.rag.multimodal_content import build_openai_multimodal_user_content


SYSTEM_PROMPT = (
    "你是一位乐于助人的论文检索与知识库助手。\n"
    "对于用户文档、论文、笔记等**事实性问题**：你必须优先且只能基于「本次提供的上下文检索片段」以及「对话中已出现的消息」作答；"
    "若上下文不足以支持答案，请明确说明你不知道或缺少何种材料。\n"
    "在证据充分时，**回答宜写得充实、有层次**：先给总括，再分点展开机制/步骤/模块关系；可解释关键术语、数据流或因果；"
    "若上下文有示例、类比、局限与注意事项，应一并纳入；避免用一两句话敷衍带过，除非用户明确要求极简。\n\n"
    "**例外（元问题）**：当用户问的是你自己的身份、模型名称、开发者、本应用能做什么、能否联网等**非文档事实**问题时，"
    "请直接、诚实、简洁地回答，**不要**回答「上下文中没有相关信息所以无法作答」，也不必强行从检索片段里找依据。"
)

# 当本轮“数据库上下文与图片不相关”，策略切换为“只基于图片回答”时使用。
SYSTEM_PROMPT_IMAGE_ONLY = (
    "你是一位乐于助人的图像解析助手。\n"
    "当本轮未提供或未使用数据库检索上下文时，你必须且只能基于：\n"
    "1）用户提供的图片内容；\n"
    "2）对话中已出现的文字信息；\n"
    "来回答用户问题。\n"
    "如果图片内容不足以支持答案，请明确说明缺什么信息，并建议用户补充更清晰的图片或提供必要文字。\n\n"
    "禁止编造图片中不存在的证据、数据、引用或结论。"
)

# 加强版提示词：在面试时可用来展示更细致的提示词设计思路。
# 在发起向量/BM25 检索前，可选地用 LLM 将口语问题改写成更利于召回的查询（见 tools/rag/language.py）。
QUERY_REWRITE_SYSTEM_PROMPT = """
你是一个「检索查询改写」助手。用户会给出自然语言问题，你需要改写成更适合向量检索与关键词检索的查询语句。

规则：
1. 只输出一行：改写后的检索查询；不要换行、不要编号列表、不要解释或前后缀。
2. 保留核心实体、任务类型与关键技术术语；去掉寒暄与冗余表述。
3. 可补充与主题强相关的同义技术词（用空格分隔），但不要编造无关实体。
4. 若用户用中文提问而知识库可能为英文，可在不改变含义的前提下加入简短英文关键词。
5. 若原问题已足够简短明确，可只做轻度清理。

6. 若用户问题包含 LaTeX 数学表达式（形如 `$...$`、`$$...$$`、`\(...\)`、`\[...\]` 或直接的 LaTeX 片段），必须**原样保留**这些数学部分：
   - 不要翻译数学符号含义
   - 不要删除或改写反斜杠 `\`
   - 不要把数学表达式拆散到多个 token 里（尽量保持原样）

只输出该行检索查询，不要输出其它任何文字。
""".strip()

# 将用户一次输入中的多个问题拆成独立子问题（见 tools/rag/language.split_compound_question）。
SUBQUESTION_SPLIT_SYSTEM_PROMPT = """
你是「问题拆分」助手。用户输入里可能包含多个独立问题（例如用分号、问号、以及/还有/另外等连接）。

任务：拆成若干条**完整、可单独回答**的子问题。

规则：
1. 只输出一个 JSON 数组，元素为字符串；不要 markdown 代码块、不要解释、不要序号前缀。
2. 若其实只有一个问题，输出仅含一个元素的数组，例如 ["……"]。
3. 保留每条子问题的关键实体与约束；不要合并不同主题的问题。
4. 子问题数量不超过 {max_subquestions}；若用户问题更多，请合并最相关的或保留前 {max_subquestions} 条。
5. 语言与用户输入保持一致（中文输入则子问题也用中文）。

示例输出：["RAG 是什么？","如何评估检索效果？"]
""".strip()

SYSTEM_PROMPT_DETAILED = """
你是一名 AI 助手，帮助用户理解并使用自己的知识库（笔记、文档、论文、报告等）。

你的行为必须遵循以下规则：

1. 事实依据（唯一真相）
   - 你必须且只能基于本次对话中“已提供的上下文内容”以及“在对话中出现过的先前消息”回答。
   - 如果上下文无法支持答案，你必须明确说明你不知道，并可选地指出为了回答需要哪些类型的文档或数据。

2. 回答风格
   - 在忠于上下文的前提下，优先给出**清晰且足够详细**的回答（先总述再分点展开）。
   - 总结文档/技术内容时，覆盖关键机制、模块关系与要点；若有局限性或前提，一并说明。
   - 除非用户明确要求极简，否则避免过度压缩成一两句空泛概括。

3. 引用与可追溯性
   - 在适用时，请使用所提供的 `source` 元数据进行引用（例如：文件名或 URL）。
   - 不要编造引用、ID 或 URL。

4. 语言
   - 如果用户的问题是中文，请用中文回答。
   - 如果用户的问题是英文，请用英文回答。

5. 安全与诚实
   - 不要假装自己能够访问工具与上下文之外的外部系统信息。
   - 对不确定或缺失的信息要说清楚。
""".strip()

MULTI_QUESTION_REASONING_TEMPLATE = """
请按照以下步骤来回答问题（允许一次输入包含多个子问题）：

步骤 1：理解问题
- 识别用户想解决的核心目标与限制条件。
- 如果问题隐含多个方面/子任务，请先把它们拆成“子问题列表”。

步骤 2：分析上下文证据
- 在提供的上下文中找出与每个子问题相关的要点。
- 指出关键证据来自哪一段上下文（若上下文中包含形如 `[来源: ...]` 的标注，请保留对应引用）。

步骤 3：推理与整合
- 基于上下文证据进行推断或归纳。
- 对缺失的关键证据要诚实说明，并告诉用户需要补充什么信息才能继续。

步骤 4：组织答案
- 按“子问题 -> 结论/要点”的结构分别输出；每个子问题的结论下可再拆小点，便于阅读。
- 最后给出 **「最终答案」**：这是对用户问题的**完整收束**，应比前面各条小结**更详实**，而不是简单重复一句话。

输出要求：
1）先输出：子问题列表（若只有一个就只列一个）。
2）再对每个子问题输出：推理要点（可稍展开，点出上下文依据或来源标注）+ 结论。
3）最后输出：**最终答案**（单独成段，建议用标题「### 最终答案」或「**最终答案：**」引出），并满足：
   - 用 2–5 个自然段或分层列表，把核心机制、模块职责、数据/条件如何流动讲清楚；
   - 若文档中有定义、公式名、模块名、训练/推理阶段等，应写具体，勿空泛概括；
   - 可保留文档中的类比或比喻（若有），并在其后用一两句技术话收束；
   - 若上下文提到局限、假设或未覆盖部分，应在最终答案末尾简要说明；
   - 仍严禁编造上下文未出现的事实、引用或数据。

上下文：
{context}

问题：
{question}
""".strip()

# 工具模式（answer_with_tools）最终合成：面向终端用户可读；知识对比类允许分点写详，与「调试/MCP 技术复盘」区分
TOOLS_FINAL_ANSWER_TEMPLATE = """
以下是本轮对话中**工具调用返回的文本结果**（可能来自天气、联网搜索、知识库、MCP 等）。请据此回答用户。

【工具输出】
{context}

【用户原问题】
{question}

写作要求（请严格遵守）：
1. **面向最终用户**：自然、好读；用户用中文则优先用中文。
2. **篇幅与结构（按问题类型自适应）**：
   - **极简事实**（如单个数值、一句确认类）：两三句即可（**不含**天气类，见下条）。
   - **天气 / 气温 / 湿度 / 风 / 出行气象**：必须先有 **1～2 段**把实况说清楚（温度、相对湿度、体感温度、风速、天气现象等；缺项不编造），**禁止**只用一句话敷衍；随后用 **分点** 给出 **4～6 条**实用建议（穿衣与增减衣、是否带伞或防晒、通勤与户外活动、老人幼儿注意事项、呼吸道或过敏相关提示等），每条宜有一句话点明**为何**如此建议。**整体不要太短**。
   - **对比 / 区别 / 优缺点 / 如何选择 / 什么是（概念阐释）**：必须先有 **1 段简短总述**，再按 **分点** 展开（用 `- ` 或 `1.` 均可），维度可包括：定位与范式、语法与类型、性能与资源、内存与并发、工程与生态、典型应用场景等（择与问题相关的 **4～8 条**，每条 **2～4 句**，写清楚「差异是什么、为什么重要」）。不要只甩链接标题；请把摘要里的**实质信息消化写进正文**。若有参考链接，可在文末加一小节「延伸阅读」列 **2～4 条** 即可。
3. **禁止**把回答写成「子问题列表」「推理要点」「MCP/SDK 调试步骤」、调用栈、模块职责、数据流等**技术复盘**，**除非**用户明确在问调试/实现原理。
4. 若工具输出**已是完整短事实**，可稍加润色；若材料较多，应**归纳合并**为上面的结构化正文，不要堆砌重复摘要。
5. 若工具报错或结果为空，说明原因并给可行建议（简短即可）。
6. 严禁编造工具未提供的事实、链接或数据。
""".strip()

# 简短版：不要求多步骤长文输出，适合易触发 max_tokens 截断的场景
COMPACT_RAG_USER_TEMPLATE = """
请仅根据下列上下文回答用户问题。若上下文不足以回答，请明确说明缺什么信息。
不必按「步骤1/2/3/4」展开，但**仍请写清楚、写充分**：先给 1–2 句总述，再分点说明关键机制、模块或流程；需要处可解释术语与相互关系。
若上下文有示例、类比或注意事项，应纳入；避免只有一句结论。最后可用小标题「**最终答案：**」收束一段稍详细的汇总（2–4 句以上）。

上下文：
{context}

问题：
{question}
""".strip()

IMAGE_ONLY_RAG_USER_TEMPLATE = """
请根据用户提供的图片内容以及问题回答。
如果图片不足以支持答案，请明确说明你需要哪些补充信息（例如更清晰的截图、完整表格、或关键文字）。
不要编造图片中不存在的证据或数据；能确定的就直接给出结论，不能确定就坦诚说明不确定。

问题：
{question}
""".strip()

# 论文内容问答（多通道）专用模板
PAPER_TEXT_QA_USER_TEMPLATE = """
你将回答“论文正文内容”问题。请优先根据正文证据给出结论，再补充关键机制/流程。
若证据不足请明确说明缺少哪一部分（方法段、实验段或结论段）。

问题：
{question}
""".strip()

PAPER_TABLE_QA_USER_TEMPLATE = """
你将回答「论文表格」类问题。
**范围约束**：只根据【当前论文】自身排版中的 Table / Tab. N 来解读（系统提示中已给出 arXiv id）。检索片段里若出现**其他论文**的表格或对其 Table N 的文字引用，那只是相关工作/对比实验段落，**不得**将其与本论文的 Table N 混成「两处都是本论文的 Table N」；除非用户本句明确要求多篇论文对照。
回答时优先给出：表题与量纲、行列/指标含义、可引用的核心数字、对比基线与提升幅度（仅证据中有的）。
禁止编造表格中不存在的数值；证据不足时请明确说明缺哪一类信息。

问题：
{question}
""".strip()

PAPER_FIGURE_QA_USER_TEMPLATE = """
你将回答“论文图片/图表内容”问题，必须严格做证据绑定：
1) 仅允许使用以下证据：命中的 figure caption / figure_number / 同页正文片段 / 用户提供图片。
2) 若用户指定 Fig./Figure N，先检查证据中是否存在对应 figure_number=N；若不存在，必须明确说“未检索到 Fig. N 的可靠证据”。
3) 禁止猜测图中物体类别、颜色含义、轨迹语义；证据不足时明确“不足以确认”。
4) 每个结论都要对应证据来源（caption/figure_number/page/正文片段）。
5) 先判断图类型：流程图/框架图、结果曲线、可视化渲染、对比图、表格截图或其他。
6) 若是流程图/框架图，按以下结构输出：
   1. 流程阶段（按从左到右/从上到下）
   2. 每个模块输入/输出
   3. 模块之间的连接关系
   4. 图中无法确认的信息
7) 若不是流程图/框架图，按以下结构输出：
   1. 图类型与用途
   2. 可确认的关键元素（对象/坐标轴/图例/对比项）
   3. 论文中该图支撑的结论（仅限证据可得）
   4. 图中无法确认的信息

问题：
{question}
""".strip()

PAPER_MULTIMODAL_QA_USER_TEMPLATE = """
你将综合正文、表格、图像三类证据回答论文问题。
请先给整体结论，再分别说明：
1) 正文证据
2) 表格证据（若有）
3) 图像证据（若有）
最后给出简短总结。

问题：
{question}
""".strip()

def _resolve_image_path(project_root: Path, raw: str) -> Path | None:
    p = Path(raw.strip())
    if not str(p):
        return None
    if not p.is_absolute():
        p = project_root / p
    try:
        p = p.resolve()
    except OSError:
        return None
    return p if p.is_file() else None


def _collect_retrieval_image_paths(
    retrieved_docs: Iterable[Tuple[object, float]],
    project_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for doc, _score in retrieved_docs:
        meta = getattr(doc, "metadata", {}) or {}
        ip = meta.get("image_path")
        if not ip:
            continue
        p = _resolve_image_path(project_root, str(ip))
        if p is None:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)
    return paths


def _merge_image_paths(
    user_paths: Sequence[str] | None,
    retrieval_paths: list[Path],
    project_root: Path,
) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()
    user_cap = min(RAG_USER_UPLOAD_MAX_IMAGES, RAG_MAX_IMAGES_PER_MESSAGE)

    for raw in user_paths or []:
        if len(merged) >= RAG_MAX_IMAGES_PER_MESSAGE or len(merged) >= user_cap:
            break
        p = _resolve_image_path(project_root, str(raw))
        if p is None:
            continue
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        merged.append(p)

    for p in retrieval_paths:
        if len(merged) >= RAG_MAX_IMAGES_PER_MESSAGE:
            break
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        merged.append(p)
    return merged


_FIGURE_Q_PAT = re.compile(r"(?:\bfig(?:ure)?\.?\s*\d+|图\s*\d+|Figure\s*\d+|Fig\.\s*\d+)", re.IGNORECASE)


def _looks_like_figure_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _FIGURE_Q_PAT.search(q):
        return True
    return bool(re.search(r"(图|figure|fig\.)", q, re.IGNORECASE))


def _looks_like_table_question(question: str) -> bool:
    from tools.retrieval.query_understanding import analyze_paper_query

    return bool(analyze_paper_query(question or "").wants_table)


def build_rag_prompt(
    question: str,
    retrieved_docs: Iterable[Tuple[object, float]],
    chat_history: List[dict] | None = None,
    *,
    user_image_paths: Sequence[str] | None = None,
    include_user_images_in_history: bool = True,
    project_root: Path | None = None,
    paper_glossary_web_supplement: bool = False,
    paper_scope_arxiv_id: str | None = None,
) -> List[dict]:
    """将检索到的文档与问题拼成 ChatCompletion 风格的 messages。

    chat_history: [{"role","content",...,"image_paths"?: [...]}, ...]；
    带图 user 轮次仅最近 RAG_CHAT_HISTORY_MAX_IMAGE_TURNS 轮重放像素。
    最后一条 user 可含多模态 content（文本 + 检索图 + 用户上传图）。
    """

    root = project_root or Path.cwd()

    retrieved_list = list(retrieved_docs)
    image_only_mode = bool(user_image_paths) and not retrieved_list

    context_parts: list[str] = []
    used_chars = 0
    has_focus_full = any(
        isinstance((getattr(d, "metadata", None) or {}), dict)
        and str((getattr(d, "metadata", None) or {}).get("type") or "").startswith("paper_focus_pg_full_")
        for d, _ in retrieved_list
    )
    has_table_inject = any(
        isinstance((getattr(d, "metadata", None) or {}), dict)
        and str((getattr(d, "metadata", None) or {}).get("type") or "") == "paper_table_pg"
        for d, _ in retrieved_list
    )
    total_cap = int(RAG_CONTEXT_TOTAL_MAX_CHARS or 0)
    if (
        (has_focus_full or has_table_inject)
        and int(RAG_CONTEXT_PAPER_METHOD_TOTAL_MAX_CHARS or 0) > 0
    ):
        total_cap = int(RAG_CONTEXT_PAPER_METHOD_TOTAL_MAX_CHARS)
    per_cap_default = int(RAG_CONTEXT_MAX_CHARS_PER_DOC or 0)
    sep = "\n\n---\n\n"

    for doc, score in retrieved_list:
        raw = getattr(doc, "page_content", str(doc)) or ""
        content = raw.strip()
        meta = getattr(doc, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        source = meta.get("source", "unknown")
        typ = str(meta.get("type") or "")
        role_l = str(meta.get("chunk_role") or "").strip().lower()
        is_full_method = typ.startswith("paper_focus_pg_full_")
        is_table_evidence = typ in ("paper_table_pg", "pdf_table") or role_l == "table"
        if is_full_method or is_table_evidence:
            per_cap = int(RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC)
        else:
            per_cap = per_cap_default
        if per_cap > 0 and len(content) > per_cap:
            trunc_note = (
                "RAG_CONTEXT_PAPER_METHOD_MAX_CHARS_PER_DOC"
                if (is_full_method or is_table_evidence)
                else "RAG_CONTEXT_MAX_CHARS_PER_DOC"
            )
            content = content[:per_cap] + f"\n\n...（单条检索片段已截断，见 {trunc_note}）"
        block = f"[来源: {source}, 相关度: {score:.3f}]\n{content}"
        if total_cap > 0:
            overhead = len(sep) if context_parts else 0
            if used_chars + overhead >= total_cap:
                context_parts.append(
                    "（已达到检索上下文总字数上限，其余片段已省略；可调大 RAG_CONTEXT_TOTAL_MAX_CHARS 或减小 k。）"
                )
                break
            room = total_cap - used_chars - overhead
            if room <= 0:
                context_parts.append(
                    "（已达到检索上下文总字数上限，其余片段已省略；可调大 RAG_CONTEXT_TOTAL_MAX_CHARS 或减小 k。）"
                )
                break
            if len(block) > room:
                block = block[:room] + "\n\n...（总字数预算截断，见 RAG_CONTEXT_TOTAL_MAX_CHARS）"
            used_chars += overhead + len(block)
        context_parts.append(block)

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "未检索到上下文。"

    retrieval_img = _collect_retrieval_image_paths(retrieved_list, root)
    final_paths = _merge_image_paths(user_image_paths, retrieval_img, root)

    sys_prompt = SYSTEM_PROMPT_IMAGE_ONLY if image_only_mode else SYSTEM_PROMPT
    ps_aid = (paper_scope_arxiv_id or "").strip()
    if (not image_only_mode) and ps_aid:
        sys_prompt = (
            str(sys_prompt)
            + "\n\n"
            + (
                "（**当前阅读论文范围**：本轮若无特别说明，用户所指 Table N / Figure N / 实验表 等，"
                f"均应优先对应 **arXiv:{ps_aid}** 论文自身中的编号；"
                "正文中引用他篇文献的表格或数字时，只能当作背景，不得把那条引用当成「本论文又有一处 Table N」或自行组织跨论文表内合并解读。）"
            )
        )
    if not image_only_mode and retrieved_list:
        try:
            types = {
                str((getattr(doc, "metadata", {}) or {}).get("type") or "").strip()
                for doc, _s in retrieved_list
            }
        except Exception:
            types = set()
        types.discard("")
        if types:
            notes: list[str] = []
            if "conversation_memory" in types:
                notes.append(
                    "（提示：本轮上下文包含**对话历史记忆**片段；它来自用户与助手先前对话的摘要/片段，"
                    "可能不完整或带口语省略。请结合当前问题与对话历史组织回答；不确定就说明不确定。）"
                )
            if "arxiv_search" in types:
                notes.append(
                    "（提示：本轮上下文包含**arXiv 在线检索**返回的标题/作者/摘要等信息；"
                    "请基于这些片段回答，并引用 URL；摘要可能不覆盖全部细节，需提醒用户核对原文。）"
                )
            if "web_search_nonpaper" in types or "web_search" in types:
                notes.append(
                    "（提示：本轮上下文包含**联网搜索摘要**片段；内容可能有时效性或不完全准确。"
                    "请综合多条来源回答，并提示用户核对关键结论的原始来源。）"
                )
            if paper_glossary_web_supplement:
                notes.append(
                    "（提示：用户在阅读论文语境下追问**术语/缩写/定义**；论文片段可能未覆盖该术语。"
                    "请先用联网摘要简明解释术语含义，再说明其与当前论文上下文的可能关系；"
                    "勿用网文细节覆盖论文中已明确陈述的事实。）"
                )
            if "session_upload" in types:
                notes.append(
                    "（提示：本轮上下文包含用户在本会话中**上传并入库**的文档片段；"
                    "请优先据此回答与文档相关的问题。）"
                )
            if notes:
                sys_prompt = str(sys_prompt) + "\n\n" + "\n".join(notes)
    messages: List[dict] = [{"role": "system", "content": sys_prompt}]

    hist = chat_history or []
    user_with_img_idx: list[int] = []
    if include_user_images_in_history:
        for i, msg in enumerate(hist):
            if msg.get("role") != "user":
                continue
            ips = msg.get("image_paths")
            if isinstance(ips, list) and any(str(x).strip() for x in ips):
                user_with_img_idx.append(i)
        allowed_hist_img = set(user_with_img_idx[-RAG_CHAT_HISTORY_MAX_IMAGE_TURNS :])
    else:
        # 禁止把历史里的用户图片重放进 prompt，避免无关图片干扰“仅基于数据库上下文”的回答策略
        allowed_hist_img = set()

    for i, msg in enumerate(hist):
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        text = str(msg.get("content") or "").strip()
        ips = msg.get("image_paths") if isinstance(msg.get("image_paths"), list) else []

        if role == "assistant":
            if text:
                messages.append({"role": "assistant", "content": text})
            continue

        if ips and i in allowed_hist_img:
            path_objs: list[Path] = []
            for raw in ips:
                p = _resolve_image_path(root, str(raw))
                if p is not None:
                    path_objs.append(p)
            content = build_openai_multimodal_user_content(
                text or "（用户上传了图片）",
                path_objs,
                max_images=min(len(path_objs), RAG_MAX_IMAGES_PER_MESSAGE),
            )
            messages.append({"role": "user", "content": content})
        elif ips:
            note = "[历史消息含图片，已省略图像以节省上下文]\n"
            messages.append({"role": "user", "content": note + text if text else note.strip()})
        elif text:
            messages.append({"role": "user", "content": text})

    if image_only_mode:
        user_body = IMAGE_ONLY_RAG_USER_TEMPLATE.format(question=question)
    elif RAG_USE_COMPACT_RAG_USER_PROMPT:
        user_body = COMPACT_RAG_USER_TEMPLATE.format(
            context=context_text, question=question
        )
    else:
        user_body = MULTI_QUESTION_REASONING_TEMPLATE.format(
            context=context_text, question=question
        )
    # 仅在“图像问答”场景追加图证据绑定约束；不改正文问答模板。
    try:
        has_figure_ctx = any(
            str((getattr(doc, "metadata", {}) or {}).get("type") or "").startswith("pdf_figure")
            or str((getattr(doc, "metadata", {}) or {}).get("chunk_role") or "").strip() == "figure"
            for doc, _s in retrieved_list
        )
    except Exception:
        has_figure_ctx = False
    if has_figure_ctx and _looks_like_figure_question(question):
        user_body = f"{user_body}\n\n{PAPER_FIGURE_QA_USER_TEMPLATE.format(question=question)}"

    try:
        has_table_ctx = any(
            str((getattr(doc, "metadata", {}) or {}).get("type") or "")
            in (
                "paper_table_pg",
                "pdf_table",
            )
            or str((getattr(doc, "metadata", {}) or {}).get("chunk_role") or "").strip().lower()
            == "table"
            for doc, _s in retrieved_list
        )
    except Exception:
        has_table_ctx = False
    if has_table_ctx and _looks_like_table_question(question):
        user_body = f"{user_body}\n\n{PAPER_TABLE_QA_USER_TEMPLATE.format(question=question)}"

    last_content = build_openai_multimodal_user_content(
        user_body,
        final_paths,
        max_images=RAG_MAX_IMAGES_PER_MESSAGE,
    )
    messages.append({"role": "user", "content": last_content})
    return messages

