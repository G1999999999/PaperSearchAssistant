"""
相对时间表述（今天/明天等）与联网检索：把「今天」落实为具体日期，便于搜到当日赛程等实时信息。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

# 含这些词时，问题依赖「当前时刻」，需要在 system 里锚定日期
_REL_TIME_CN = (
    "今天",
    "今日",
    "这天",
    "今晚",
    "今早",
    "明早",
    "明天",
    "明日",
    "后天",
    "大后天",
    "昨天",
    "昨日",
    "前天",
    "这周",
    "本周",
    "下周",
    "现在",
    "目前",
    "当前",
    "最新",
    "实时",
)
_REL_TIME_EN = (
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "this week",
    "now",
    "current",
)


def _now_for_display() -> datetime:
    from config import RAG_DISPLAY_TIMEZONE

    tz_name = (RAG_DISPLAY_TIMEZONE or "").strip()
    if not tz_name:
        return datetime.now()
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def needs_temporal_anchor(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    ql = q.lower()
    if any(k in q for k in _REL_TIME_CN):
        return True
    return any(k in ql for k in _REL_TIME_EN)


def format_temporal_system_note() -> str:
    """供 RAG system 追加：让模型知道「今天」对应哪一天（服务器/配置的时区）。"""
    now = _now_for_display()
    d = now.date()
    return (
        f"时间锚定（用于理解用户说的「今天」「明天」等）："
        f"当前日期 {d.isoformat()}（{d.year}年{d.month}月{d.day}日），"
        f"本地时间约 {now.strftime('%H:%M')}。"
        f"请据此理解相对日期；若上下文仍无该日的具体赛程/比分，应明确说明无法从摘要中确认并建议用户查看官网比分页。"
    )


def _add_date_parts(parts: list[str], d: date) -> None:
    parts.append(f"{d.year}年{d.month}月{d.day}日")
    parts.append(d.isoformat())


def expand_query_for_web_search(question: str) -> str:
    """在发起网页搜索前扩展 query：把相对日期写成具体数字日期，提高命中当日赛程摘要的概率。"""
    q = (question or "").strip()
    if not q:
        return q

    now = _now_for_display()
    today = now.date()
    parts: list[str] = [q]

    if any(x in q for x in ("今天", "今日", "这天", "今晚", "今早")):
        _add_date_parts(parts, today)
        parts.append("today schedule scores")
    if "明早" in q or "明天" in q or "明日" in q:
        _add_date_parts(parts, today + timedelta(days=1))
        parts.append("tomorrow schedule scores")
    if "后天" in q:
        _add_date_parts(parts, today + timedelta(days=2))
    if "大后天" in q:
        _add_date_parts(parts, today + timedelta(days=3))
    if "昨天" in q or "昨日" in q:
        _add_date_parts(parts, today - timedelta(days=1))
    if "前天" in q:
        _add_date_parts(parts, today - timedelta(days=2))

    ql = q.lower()
    if "today" in ql or "tonight" in ql:
        _add_date_parts(parts, today)
    if "tomorrow" in ql:
        _add_date_parts(parts, today + timedelta(days=1))
    if "yesterday" in ql:
        _add_date_parts(parts, today - timedelta(days=1))

    # 体育赛事当日赛程类：加强英文检索词
    if any(k in ql for k in ("nba", "nfl", "mlb", "nhl", "cba", "英超", "西甲", "欧冠")):
        parts.append("schedule game results scores")
    if any(k in q for k in ("比赛", "赛程", "比分", "对阵")):
        parts.append("赛程 比分")

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


def looks_like_live_schedule_query(question: str) -> bool:
    """是否像「查当天有没有比赛」这类，值得尝试拉取网页正文补充摘要。"""
    q = (question or "").lower()
    zh = question or ""
    sport = any(
        k in q
        for k in (
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "cba",
            "足球",
            "篮球",
            "英超",
            "西甲",
            "欧冠",
            "意甲",
        )
    )
    sched = any(
        k in zh or k in q
        for k in ("比赛", "赛程", "比分", "对阵", "schedule", "game", "score", "match")
    )
    return sport and sched
