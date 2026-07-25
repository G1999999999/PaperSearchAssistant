#!/usr/bin/env python3
"""独立 MCP 进程：天气查询（FastMCP + Open-Meteo）。

与 LangChain 进程内 `tool_weather` 对照：本文件通过 stdio JSON-RPC 暴露 MCP tools，
由 `mcp_runtime` 拉起子进程；业务逻辑复用 `tools.agent.weather.get_weather`。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 子进程工作目录通常为 PaperSearchAssistant；保证可 import 项目包
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.agent.weather import get_weather, weather_info_tool_text

mcp = FastMCP("OpenMeteoWeather")


@mcp.tool()
def query_weather(city: str) -> str:
    """根据城市名查询当前天气（Open-Meteo，无需 API Key）。

    城市名建议使用英文拼音小写（如 beijing、shanghai），与内置坐标表一致。
    """
    name = (city or "").strip() or "beijing"
    info = get_weather(name)
    return weather_info_tool_text(info)


@mcp.tool()
def season_weather_tips(season: str) -> str:
    """返回指定季节的简单穿衣与出行提示（演示用静态知识，非实时气象）。"""
    s = (season or "").strip().lower()
    tips = {
        "spring": "春季：温差大，建议洋葱式穿衣；花粉过敏者注意防护。",
        "summer": "夏季：注意防暑与补水；雷雨天气减少户外活动。",
        "autumn": "秋季：早晚偏凉，可备薄外套；空气干燥时注意保湿。",
        "winter": "冬季：注意保暖与路面防滑；室内取暖注意通风。",
    }
    key = s
    if s in ("春", "春天"):
        key = "spring"
    if s in ("夏", "夏天"):
        key = "summer"
    if s in ("秋", "秋天"):
        key = "autumn"
    if s in ("冬", "冬天"):
        key = "winter"
    return tips.get(key, "请使用 spring/summer/autumn/winter 或 春夏秋冬。")


if __name__ == "__main__":
    mcp.run(transport="stdio")
