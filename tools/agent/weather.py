from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import requests


@dataclass
class WeatherInfo:
    city: str
    description: str
    temperature_c: float
    wind_speed_kmh: float | None = None
    relative_humidity_pct: float | None = None
    apparent_temperature_c: float | None = None
    weather_condition_zh: str | None = None


def weather_info_tool_text(info: WeatherInfo) -> str:
    """供工具 / LLM 消费的简短结构化实况（非朗朗上口的最终答复）。"""
    lines = [
        f"城市：{info.city}",
        f"概况：{info.description}",
        f"气温：{info.temperature_c:.1f}°C",
    ]
    if info.relative_humidity_pct is not None:
        lines.append(f"相对湿度：{info.relative_humidity_pct:.0f}%")
    if info.apparent_temperature_c is not None:
        lines.append(f"体感温度：{info.apparent_temperature_c:.1f}°C")
    if info.wind_speed_kmh is not None:
        lines.append(f"风速：{info.wind_speed_kmh:.1f} km/h")
    if info.weather_condition_zh:
        lines.append(f"天气现象：{info.weather_condition_zh}")
    return "\n".join(lines)


# 简单的城市 -> (纬度, 经度) 映射，面试演示足够用
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "guangzhou": (23.1291, 113.2644),
    "shenzhen": (22.5431, 114.0579),
    "hangzhou": (30.2741, 120.1551),
}

def _geocode_city_to_latlon(city: str) -> Tuple[float, float] | None:
    """用 Open-Meteo 地理编码接口把城市名解析为经纬度。

    返回 None 表示无法解析（超时/解析失败/无结果）。
    """
    q = (city or "").strip()
    if not q:
        return None

    # Open-Meteo geocoding: 无需 API Key
    # docs: https://open-meteo.com/en/docs/geocoding-api
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        resp = requests.get(
            url,
            params={
                "name": q,
                "count": 1,
                "language": "zh",
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        if not results:
            return None
        first = results[0] or {}
        lat = first.get("latitude")
        lon = first.get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception:
        return None


def _wmo_weather_zh(code: int | None) -> str | None:
    """WMO weather_code 转简短中文（Open-Meteo 说明见文档）。"""
    if code is None:
        return None
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c == 0:
        return "晴"
    if c == 1:
        return "大部晴朗"
    if c == 2:
        return "多云"
    if c == 3:
        return "阴"
    if c in (45, 48):
        return "雾"
    if c in (51, 53, 55):
        return "毛毛雨"
    if c in (56, 57):
        return "冻雨"
    if c in (61, 63, 65):
        return "雨"
    if c in (66, 67):
        return "强冻雨"
    if c in (71, 73, 75):
        return "雪"
    if c == 77:
        return "雪粒"
    if c in (80, 81, 82):
        return "阵雨"
    if c in (85, 86):
        return "阵雪"
    if c == 95:
        return "雷阵雨"
    if c in (96, 99):
        return "雷阵雨伴冰雹"
    return "其他"


def get_weather(city: str) -> WeatherInfo:
    """
    使用 Open-Meteo API 查询当前气温。

    - 不需要 API Key
    - 通过内置的城市经纬度映射来演示“工具调用”
    """
    key = (city or "").lower()
    lat_lon = _CITY_COORDS.get(key)
    if lat_lon is None:
        # 自动地理编码：避免维护城市映射表
        lat_lon = _geocode_city_to_latlon(city)
    if lat_lon is None:
        # 兜底：找不到就退回到兜底数据
        return WeatherInfo(
            city=city,
            description="未知城市（兜底方案）",
            temperature_c=23.5,
            wind_speed_kmh=None,
            relative_humidity_pct=None,
            apparent_temperature_c=None,
            weather_condition_zh=None,
        )

    lat, lon = lat_lon
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        "weather_code,wind_speed_10m"
        "&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current") or {}
        temp_c = float(current.get("temperature_2m", 0.0))
        humid_raw = current.get("relative_humidity_2m")
        apparent_raw = current.get("apparent_temperature")
        code_raw = current.get("weather_code")
        wind_raw = current.get("wind_speed_10m")

        relative_humidity_pct: float | None = None
        try:
            if humid_raw is not None:
                relative_humidity_pct = float(humid_raw)
        except Exception:
            relative_humidity_pct = None

        apparent_temperature_c: float | None = None
        try:
            if apparent_raw is not None:
                apparent_temperature_c = float(apparent_raw)
        except Exception:
            apparent_temperature_c = None

        wind_speed_kmh: float | None = None
        try:
            if wind_raw is not None:
                wind_speed_kmh = float(wind_raw)
        except Exception:
            wind_speed_kmh = None

        weather_condition_zh = _wmo_weather_zh(code_raw)
        parts: list[str] = []
        if weather_condition_zh:
            parts.append(weather_condition_zh)
        parts.append(f"气温约 {temp_c:.1f}°C")
        if relative_humidity_pct is not None:
            parts.append(f"相对湿度约 {relative_humidity_pct:.0f}%")
        if apparent_temperature_c is not None:
            parts.append(f"体感约 {apparent_temperature_c:.1f}°C")
        if wind_speed_kmh is not None:
            parts.append(f"风速约 {wind_speed_kmh:.1f} km/h")
        elif not parts:
            parts.append("实况数据不完整")
        description = "，".join(parts)

        return WeatherInfo(
            city=city,
            description=description,
            temperature_c=temp_c,
            wind_speed_kmh=wind_speed_kmh,
            relative_humidity_pct=relative_humidity_pct,
            apparent_temperature_c=apparent_temperature_c,
            weather_condition_zh=weather_condition_zh,
        )
    except Exception:
        # 任何失败都退回到兜底数据，避免让调用方处理太多异常
        return WeatherInfo(
            city=city,
            description="天气服务不可用（兜底方案）",
            temperature_c=23.5,
            wind_speed_kmh=None,
            relative_humidity_pct=None,
            apparent_temperature_c=None,
            weather_condition_zh=None,
        )