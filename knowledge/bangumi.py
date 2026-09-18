"""bangumi SQLite 库的检索接口（返回结构与 KnowledgeIndex.search 一致）。

供插件侧统一调用：`search(query, top_k, max_chars)` → [{title, text, score}]
- title 形如 "条目#8491 进击的巨人"、"角色#1 鲁路修·兰佩路基"
- 库为只读打开（分发给插件后随包携带，不在运行时写入）
- 分词/查询清洗复用 `text`，保证与萌百科库的查询行为一致
"""
from __future__ import annotations

import datetime
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

from . import text
from .paths import KB_DATA_DIR

DB_PATH = KB_DATA_DIR / "bangumi.db"
_KIND_LABELS = {"subject": "条目", "character": "角色", "person": "人物"}

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def available() -> bool:
    """库文件是否存在（未构建时插件应跳过该工具）。"""
    return DB_PATH.is_file()


def _get_conn() -> sqlite3.Connection:
    """懒加载只读连接（插件通过 asyncio.to_thread 调用，加锁共用）。"""
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(
                f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
            )
        return _conn


def reset_connection() -> None:
    """丢弃缓存的连接（库被重建后调用）。"""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# 可安全用于 FTS5 前缀查询的词元：文字/数字/下划线（中文也在 \w 内）
_SAFE_FTS_TOKEN_RE = re.compile(r"^\w+$")


def _build_match_expr(tokens: List[str]) -> str:
    """构造 FTS5 匹配表达式：每个词元做「精确 OR 前缀」。

    前缀兜底用于缩写比词条名短的情况（"京吹" → 别名"京吹部"、"马娘" →
    "赛马娘"）；对常规查询实测无副作用（召回集合不变），只补回前缀型漏召。
    含特殊字符的词元退化为引号短语，避免破坏 FTS5 语法。
    """
    parts: List[str] = []
    for token in tokens:
        if _SAFE_FTS_TOKEN_RE.match(token):
            parts.append(f"({token} OR {token}*)")
        else:
            parts.append('"' + token.replace('"', '""') + '"')
    return " AND ".join(parts)


def search(query: str, top_k: int = 4, max_chars: int = 300) -> List[Dict[str, Any]]:
    """FTS5 全文检索。返回 [{title, text, score}]，score 越大越相关。"""
    if not available():
        return []
    query = str(query or "").strip()
    if not query:
        return []
    tokens = list(dict.fromkeys(text.tokenize(query, for_query=True)))
    if not tokens:
        return []
    top_k = max(1, min(int(top_k or 4), 12))
    conn = _get_conn()
    rows = conn.execute(
        "SELECT d.kind, d.source_id, d.name, d.name_cn, d.summary, d.relations, d.extra, "
        "       d.infobox, bm25(fts, 8.0, 1.0) AS score "
        "FROM fts JOIN docs d ON d.rowid = fts.rowid "
        "WHERE fts MATCH ? "
        "ORDER BY (d.name_cn = ? OR d.name = ?) DESC, score LIMIT ?",
        (_build_match_expr(tokens), query, query, top_k),
    ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        kind = str(row[0])
        label = _KIND_LABELS.get(kind, kind)
        # 角色/人物的属性资料（性别/生日/事务所…）只存在 infobox 里，去链接后补进展示；
        # subject 的 infobox 与 relations/extra 重复，构建端已不再存，这里自然为空
        info = text.strip_links(row[7]) if kind in ("character", "person") else ""
        results.append(
            {
                "title": f"{label}#{row[1]} {row[3] or row[2]}",
                "text": _compose_text(row[4], row[5], row[6], max_chars, info),
                "score": round(-float(row[8]), 3),  # bm25 为负值（越小越相关），取正统一口径
            }
        )
    return results


def _compose_text(summary: str, relations: str, extra: str, limit: int, info: str = "") -> str:
    """（角色/人物）资料卡 + 简介 + 关联信息 + 补充信息，按 max_chars 截断。

    资料卡放最前：角色/人物的 summary 常是上千字的长传记，若排在后面，
    生日/事务所/性别这些"别处没有"的属性会被 max_chars 截掉、等于没接上。
    subject 不传 info，顺序无变化。
    """
    parts = [
        part.strip()
        for part in (info or "", summary or "", relations or "", extra or "")
        if str(part or "").strip()
    ]
    text_body = "\n".join(parts)
    if len(text_body) > limit:
        text_body = text_body[:limit].rstrip() + "…"
    return text_body


def _season_extra(extra: str) -> str:
    """新番列表的补充信息：去掉与行首重复的"类型/发行日期/评分"，留平台/排名/标签。"""
    text_body = str(extra or "")
    if text_body.startswith("类型：动画"):
        text_body = text_body[len("类型：动画") :].lstrip(" /")
    text_body = re.sub(r"^发行日期：[^/]*/\s*", "", text_body)
    return re.sub(r"评分：[^/]*/\s*", "", text_body, count=1).strip()


def season(year: int = 0, month: int = 0, limit: int = 20) -> Dict[str, Any]:
    """某季度的动画新番列表（7/8/9 月都归入 7 月季度，4/5/6 归入 4 月番）。

    year/month 不传用当前日期；month 传季度内任意一月均可。
    返回 {"year", "start_month", "label", "total", "items"}，
    items 按追番人数（收藏数）降序，取前 limit 条。
    """
    today = datetime.date.today()
    year = max(1900, min(int(year or today.year), 2100))
    month = max(1, min(int(month or today.month), 12))
    start = ((month - 1) // 3) * 3 + 1
    months = [f"{year}-{start + offset:02d}" for offset in range(3)]
    result: Dict[str, Any] = {
        "year": year,
        "start_month": start,
        "label": f"{year} 年 {start} 月新番",
        "total": 0,
        "items": [],
    }
    if not available():
        return result
    limit = max(1, min(int(limit or 20), 40))
    conn = _get_conn()
    marks = ",".join("?" for _ in months)
    where = f"kind='subject' AND subject_type=2 AND ym IN ({marks})"
    result["total"] = int(
        conn.execute(f"SELECT COUNT(*) FROM docs WHERE {where}", months).fetchone()[0] or 0
    )
    rows = conn.execute(
        "SELECT source_id, name, name_cn, date, score, favorite, extra "
        f"FROM docs WHERE {where} "
        "ORDER BY favorite DESC, score DESC, date DESC LIMIT ?",
        (*months, limit),
    ).fetchall()
    result["items"] = [
        {
            "title": f"条目#{source_id} {name_cn or name}",
            "name": name_cn or name,
            "date": date or "",
            "score": round(float(score or 0), 1),
            "favorite": int(favorite or 0),
            "extra": _season_extra(extra)[:120],
        }
        for source_id, name, name_cn, date, score, favorite, extra in rows
    ]
    return result


# 中文类型名 → subject_type（书籍/漫画同为一个类型）
_SUBJECT_TYPES_CN = {"书籍": 1, "漫画": 1, "动画": 2, "音乐": 3, "游戏": 4, "三次元": 6}
# 排序字段白名单（值直接进 SQL，必须白名单）
_SORT_COLUMNS = {"score": "d.score", "favorite": "d.favorite", "date": "d.ym", "rank": "d.rank"}


def subject_type_of(name: str) -> int:
    """中文类型名 → subject_type 编号（未知按动画处理）。"""
    return _SUBJECT_TYPES_CN.get(str(name or "").strip(), 2)


def browse(
    tag: str = "",
    keyword: str = "",
    subject_type: int = 2,
    year_from: int = 0,
    year_to: int = 0,
    min_score: float = 0.0,
    sort: str = "score",
    limit: int = 15,
    max_chars: int = 220,
) -> List[Dict[str, Any]]:
    """按条件筛选作品（"最近的高分喜剧动画"这类发现型查询）。

    先用 FTS 缩小候选（毫秒级），再按类型/年份/评分过滤、按指定字段排序；
    tag 额外要求出现在标签字段里，避免"简介里顺带提到"的噪音。
    返回 [{title, text, score}]，score 为评分（方便展示）。
    """
    if not available():
        return []
    tag = str(tag or "").strip()
    keyword = str(keyword or "").strip()
    probe = f"{tag} {keyword}".strip()
    if not probe:
        return []
    tokens = list(dict.fromkeys(text.tokenize(probe, for_query=True)))
    if not tokens:
        return []
    limit = max(1, min(int(limit or 15), 40))
    order = _SORT_COLUMNS.get(str(sort or "score").strip().lower(), "d.score")

    sql = (
        "SELECT d.kind, d.source_id, d.name, d.name_cn, d.summary, d.extra, "
        "d.score, d.favorite, d.ym "
        "FROM fts JOIN docs d ON d.rowid = fts.rowid "
        "WHERE fts MATCH ? AND d.kind = 'subject' AND d.subject_type = ?"
    )
    params: List[Any] = [_build_match_expr(tokens), int(subject_type or 2)]
    if tag:
        sql += " AND d.extra LIKE ?"
        params.append(f"%{tag}%")
    if year_from:
        sql += " AND d.ym >= ?"
        params.append(f"{int(year_from):04d}-01")
    if year_to:
        sql += " AND d.ym <= ?"
        params.append(f"{int(year_to):04d}-12")
    if min_score:
        sql += " AND d.score >= ?"
        params.append(float(min_score))
    sql += f" ORDER BY {order} DESC, d.favorite DESC LIMIT ?"
    params.append(limit)

    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    results: List[Dict[str, Any]] = []
    for kind, source_id, name, name_cn, summary, extra, score, favorite, ym in rows:
        meta = " | ".join(
            part
            for part in (
                f"开播 {ym}" if ym else "",
                f"评分 {score}" if score else "",
                f"追番 {favorite}" if favorite else "",
            )
            if part
        )
        body = str(extra or "")
        if "标签：" in body:
            body = "标签：" + body.split("标签：")[-1]
        text_body = "\n".join(part for part in (meta, body, str(summary or "").strip()) if part)
        if len(text_body) > max_chars:
            text_body = text_body[:max_chars].rstrip() + "…"
        results.append(
            {
                "title": f"条目#{source_id} {name_cn or name}",
                "text": text_body,
                "score": round(float(score or 0), 1),
            }
        )
    return results


def stats() -> Dict[str, Any]:
    """库概览（调试用）：各类型条数、文件大小与数据源日期。

    ``source_date`` 优先取构建时写入的 ``meta`` 表（``source_date`` 键，即数据源
    导出/更新日期，如 bangumi/Archive 的导出日）；旧库没有 meta 表时回退为文件
    最后修改日期，保证任何情况下都显示一个真实日期。
    """
    if not available():
        return {}
    conn = _get_conn()
    counts = {
        kind: count
        for kind, count in conn.execute("SELECT kind, COUNT(*) FROM docs GROUP BY kind")
    }
    source_date = ""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='source_date'").fetchone()
        source_date = str(row[0]).strip() if row else ""
    except sqlite3.Error:
        source_date = ""
    if not source_date:
        source_date = datetime.date.fromtimestamp(DB_PATH.stat().st_mtime).isoformat()
    return {
        "counts": counts,
        "size_mb": round(DB_PATH.stat().st_size / 1048576, 1),
        "source_date": source_date,
    }
