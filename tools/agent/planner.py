"""
规划器：为自主执行过程生成结构化计划（steps）。

我们让它既稳健又便于调试：
- 要求 LLM 只输出 JSON。
- 对输出做校验与规范化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from models_qwen import qwen


@dataclass
class PlanStep:
    id: str
    title: str
    instruction: str
    done_criteria: str | None = None


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]


_PLANNER_SYSTEM = """你是自主助手的规划模块。
仅返回严格 JSON（不要使用 Markdown、不要代码围栏、不要解释性文字）。

规则：
- 生成 3-8 步，每一步都可执行且可验证。
- 每一步要足够小，能够在一次 tool/LLM 轮次内完成。
- 为每一步提供简短的 done_criteria（完成标准）。
- 输出架构（字段名必须严格一致）：
{
  "goal": "...",
  "steps": [
    {"id": "S1", "title": "...", "instruction": "...", "done_criteria": "..."},
    ...
  ]
}
""".strip()


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    # 先尝试直接解析。
    try:
        return json.loads(text)
    except Exception:
        pass
    # 再尝试从文本中截取第一个 JSON 对象。
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def make_plan(
    *,
    goal: str,
    context: str | None = None,
    max_steps: int = 8,
) -> Plan:
    user_content = f"目标：{goal.strip()}\n"
    if context:
        user_content += f"\n上下文：\n{context.strip()}\n"
    user_content += f"\n硬性限制：steps <= {max_steps}\n"

    resp = qwen.invoke(
        [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )
    raw = resp.content if hasattr(resp, "content") else str(resp)
    data = _safe_json_loads(raw) or {}
    steps_raw = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps_raw, list) or not steps_raw:
        # 兜底：给一个最小可用计划。
        return Plan(
            goal=goal,
            steps=[
                PlanStep(
                    id="S1",
                    title="直接回答",
                    instruction="如有需要可使用可用工具完成目标，然后给出最终回复。",
                    done_criteria="给出正确且完整的答案。",
                )
            ],
        )

    steps: list[PlanStep] = []
    for i, s in enumerate(steps_raw[:max_steps], start=1):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or f"S{i}")
        title = str(s.get("title") or f"步骤 {i}")
        instruction = str(s.get("instruction") or "").strip()
        if not instruction:
            instruction = f"处理：{title}"
        done_criteria = s.get("done_criteria")
        steps.append(
            PlanStep(
                id=sid,
                title=title,
                instruction=instruction,
                done_criteria=str(done_criteria).strip() if done_criteria else None,
            )
        )
    if not steps:
        steps = [
            PlanStep(
                id="S1",
                title="直接回答",
                instruction="如有需要可使用可用工具完成目标，然后给出最终回复。",
                done_criteria="给出正确且完整的答案。",
            )
        ]
    return Plan(goal=str(data.get("goal") or goal), steps=steps)

