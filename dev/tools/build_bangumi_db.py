"""从 Bangumi JSONL 构建 SQLite FTS5 检索库（1.7GB 级数据源的方案）。

数据源：topics/bangumi/jsonlines/*.jsonlines（Bangumi Archive 导出）
产物：  kb_data/bangumi.db（SQLite，FTS5 全文索引）

为什么不用 knowledge 的内存 BM25 索引：
    668 万行 ≈ 1.7GB，全内存方案需要 10GB+ 内存、索引文件数 GB、加载几分钟；
    SQLite FTS5 按需读页（内存 MB 级）、百万行毫秒级查询。

中文检索：jieba 预分词后存 FTS5（unicode61）。
    不用 FTS5 原生 trigram：它要求查询 ≥3 字，"崩铁""原神"这类 2 字查询会失效；
    预分词同时复用了 knowledge.text 的停用词/查询清洗口径。

用法：
    python dev/tools/build_bangumi_db.py --limit 3000    # 小样本试跑（验证速度与效果）
    python dev/tools/build_bangumi_db.py                 # 全量构建
    python dev/tools/build_bangumi_db.py --query "鲁路修 声优"   # 检索测试
"""
from __future__ import annotations

import argparse
import collections
import datetime
import functools
import json
import pathlib
import re
import sqlite3
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from knowledge import text as kb_text  # 复用运行时分词，保证与检索侧口径一致  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent.parent.parent  # 插件根（kb_data 所在）
DEV_DIR = BASE / "dev"  # 开发侧内容（源数据 / 构建工具）
JSONL_DIR = DEV_DIR / "topics" / "bangumi" / "jsonlines"
DB_PATH = BASE / "kb_data" / "bangumi.db"

# 支持的词条类型（episode/关联表暂不入库）
_KINDS = ("subject", "character", "person")
_KIND_NAMES = {"subject": "条目", "character": "角色", "person": "人物"}
_SUBJECT_TYPES = {1: "书籍", 2: "动画", 3: "音乐", 4: "游戏", 6: "三次元"}

# 每行 jsonlines 以 {"id":N,... 开头：先正则取 id 判保留，不被保留的行跳过 json.loads
_ID_PREFIX_RE = re.compile(r'^\{"id":(\d+)')

_SCHEMA = """
CREATE TABLE docs (
    rowid INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    name TEXT,
    name_cn TEXT,
    summary TEXT,
    infobox TEXT,
    relations TEXT,
    extra TEXT,
    subject_type INTEGER,
    date TEXT,
    ym TEXT,
    platform INTEGER,
    rank INTEGER,
    score REAL,
    favorite INTEGER
);
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE VIRTUAL TABLE fts USING fts5(
    name, body, content='', contentless_delete=1, tokenize='unicode61'
);
"""

# 结构化关联边 + 逐集数据：当前的四个工具都不读（运行时只查 docs+fts），
# 默认不产出以省约 120MB；将来做"完整反查/追番进度"用 --with-graph 重建即可。
_SCHEMA_GRAPH = """
CREATE TABLE links (
    owner_kind TEXT NOT NULL,
    owner_id INTEGER NOT NULL,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    meta TEXT
);
CREATE INDEX idx_links_owner ON links(owner_kind, owner_id);
CREATE INDEX idx_links_target ON links(target_kind, target_id);
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    name TEXT,
    name_cn TEXT,
    airdate TEXT,
    ep_type INTEGER,
    sort REAL
);
CREATE INDEX idx_episodes_subject ON episodes(subject_id);
"""


def _votes(obj: dict) -> int:
    """打分人数：score_details 各档人数求和。

    这是比收藏数更硬的热度指标（"想看"点一下即可，打分是主动评价行为）。
    rank 只在 6% 的条目上存在，不能作为过滤依据。
    """
    details = obj.get("score_details")
    return sum(int(v or 0) for v in details.values()) if isinstance(details, dict) else 0


def _infobox_parts(raw) -> list[tuple[str, str]]:
    """原始 wiki infobox → [(键, 值)]（值内多行列表用顿号连接，去掉模板符号）。"""
    if not raw:
        return []
    text = str(raw).strip()
    text = re.sub(r"^\{\{[^\n]*\n?", "", text).rstrip().rstrip("}").rstrip()
    parts: list[tuple[str, str]] = []
    for block in text.split("|")[1:]:
        if "=" not in block:
            continue
        key, _, value = block.partition("=")
        key = re.sub(r"[{}\[\]]", "", key).strip()
        items = [
            re.sub(r"[{}\[\]]", "", item).strip() for item in value.splitlines()
        ]
        items = [item for item in items if item]
        if key and items:
            parts.append((key, "、".join(items)))
    return parts


def _infobox_text(raw) -> str:
    """infobox 全量文本（存入 docs 供展示，不进全文索引）。"""
    return " / ".join(f"{key}：{value}" for key, value in _infobox_parts(raw))


# 名称类字段：值要进全文索引（用户会用别名、译名、缩写来查）
_NAME_FIELD_KEYS = ("罗马字", "罗马音", "原名", "简称", "缩写", "俗称", "通称", "别称")


def _alias_text(raw) -> str:
    """infobox 里的名称类字段（别名/译名/缩写）——这部分要进全文索引。

    否则"鲁路修"搜不到名字写作"ルルーシュ"的角色条目，
    动画的常用缩写（SAO 这类写在"简称"字段的）也搜不到。
    """
    values = [
        value
        for key, value in _infobox_parts(raw)
        if key.endswith("名") or key in _NAME_FIELD_KEYS
    ]
    return "、".join(values)


def _parse_date(raw) -> datetime.date | None:
    """解析 date 字段（ISO 日期，可能带时间或为空）。"""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _favorite_count(obj: dict) -> int:
    """收藏人次（各状态求和）——用于筛掉零收藏的占位条目。"""
    favorite = obj.get("favorite")
    return sum(int(v or 0) for v in favorite.values()) if isinstance(favorite, dict) else 0


def _is_new(obj: dict, cutoff: datetime.date, horizon: datetime.date) -> bool:
    """是否豁免"打分人数"过滤的新作品。

    - 已上映：发行日期在最近 new_months 个月内；
    - 未上映：距今不超过 3 年（horizon），且收藏 > 0
      —— 零收藏的是无人关注的占位条目，远期日期多为录入错误或空壳，都不收；
    - 无日期不算新（走打分过滤）。
    """
    date_value = _parse_date(obj.get("date"))
    if date_value is None:
        return False
    if date_value > datetime.date.today():
        return date_value <= horizon and _favorite_count(obj) > 0
    return date_value >= cutoff


def _year_requirement(
    date_value: datetime.date | None,
    base: int,
    oldest: int,
    oldest_year: int,
) -> int:
    """按发行年份插值的热度门槛（打分人数/收藏人次共用）：越早的作品要求越高。

    oldest_year 及以前 = oldest（如 2000 年及更早要 100 人打分 / 1000 人收藏）；
    此后线性下降，至今年 = base（如打分 10 / 收藏 100）。
    老作品已积累多年评价，门槛必须比新作高，否则会把整个长尾放进来。
    """
    if date_value is None:
        return base
    current_year = datetime.date.today().year
    year = date_value.year
    if year <= oldest_year:
        return oldest
    if year >= current_year:
        return base
    ratio = (year - oldest_year) / (current_year - oldest_year)
    return round(oldest - (oldest - base) * ratio)


def _build_heat_map(
    cutoff: datetime.date,
    horizon: datetime.date,
    base_votes: int,
    oldest_votes: int,
    oldest_year: int,
    base_favorites: int,
    oldest_favorites: int,
) -> dict[str, dict[int, bool]]:
    """保留判定表：各条目是否达标（打分或收藏过阈值，或新作豁免）。

    - subject：打分人数 ≥ 按发行年份插值的打分门槛，**或**收藏人次 ≥ 同曲线的
      收藏门槛（打分是主动评价、收藏只是标记，收藏门槛为打分的 10 倍），
      或命中新作豁免；
    - character/person：关联作品中任一条达标即保留
      （自身没有评分字段，关注度来自登场作品）。
    """
    subject_ok: dict[int, bool] = {}
    with (JSONL_DIR / "subject.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            key = int(obj.get("id") or 0)
            date_value = _parse_date(obj.get("date"))
            vote_threshold = _year_requirement(
                date_value, base_votes, oldest_votes, oldest_year
            )
            fav_threshold = _year_requirement(
                date_value, base_favorites, oldest_favorites, oldest_year
            )
            subject_ok[key] = (
                _votes(obj) >= vote_threshold
                or _favorite_count(obj) >= fav_threshold
                or _is_new(obj, cutoff, horizon)
            )
    keep: dict[str, dict[int, bool]] = {"subject": subject_ok}
    for kind, filename, id_key in (
        ("character", "subject-characters.jsonlines", "character_id"),
        ("person", "subject-persons.jsonlines", "person_id"),
    ):
        linked: dict[int, bool] = {}
        with (JSONL_DIR / filename).open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                subject_id = int(obj.get("subject_id") or 0)
                key = int(obj.get(id_key) or 0)
                if subject_ok.get(subject_id):
                    linked[key] = True
        keep[kind] = linked
    return keep


def _tags_text(tags) -> str:
    """标签列表 [{"name": …}] → 顿号连接。"""
    if not isinstance(tags, list):
        return ""
    names = [
        str(item.get("name", "")).strip()
        for item in tags
        if isinstance(item, dict) and item.get("name")
    ]
    return "、".join(names)


# 平台标识符 → 展示名（subject_platforms.yml 只有英文标识符，没有中文名）
_PLATFORM_CN = {
    "tv": "TV",
    "ova": "OVA",
    "movie": "剧场版",
    "short_film": "短片",
    "web": "网络",
    "anime_comic": "动画漫画",
    "comic": "漫画",
    "novel": "小说",
    "illustration": "画集",
    "picture": "绘本",
    "photo": "写真集",
    "official": "公式书",
    "jp": "日剧",
    "en": "欧美剧",
    "cn": "华语剧",
    "live": "真人秀",
    "show": "综艺",
    "album": "专辑",
    "drama": "广播剧",
    "audio": "有声书",
    "radio": "广播",
    "games": "游戏",
    "software": "软件",
    "dlc": "DLC",
    "demo": "试玩",
    "table": "桌游",
}


@functools.lru_cache(maxsize=1)
def _load_platform_map() -> dict[int, dict[int, str]]:
    """platform 数字 → 展示名，按作品类型分组。

    同一个数字在不同类型组含义不同（anime 的 1=TV、book 的 1001=漫画），
    所以必须带上作品类型一起查。结构（缩进固定）：
        types:
          anime:
            tv: &PLATFORM_TV 1
    """
    path = _MAPS_DIR / "subject_platforms.yml"
    if not path.is_file():
        return {}
    result: dict[int, dict[int, str]] = {}
    kind: int | None = None
    in_types = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_types:
            in_types = line.startswith("  types:")
            continue
        if line and not line.startswith(" "):
            break  # 顶层键（defaults: 等）→ types 段结束
        match = re.match(r"^    ([a-z_]+):\s*$", line)
        if match:
            kind = _TYPE_IDS.get(match.group(1))
            if kind is not None:
                result.setdefault(kind, {})
            continue
        match = re.match(r"^      ([a-z_]+):\s*&\w+\s+(\d+)\s*$", line)
        if match and kind is not None:
            name = match.group(1)
            if name != "none":  # none = 未指定平台
                result[kind][int(match.group(2))] = _PLATFORM_CN.get(name, name.upper())
    return result


def _extra_text(kind: str, obj: dict) -> str:
    """结构化补充信息（非全文索引内容，返回结果时展示）。"""
    parts: list[str] = []
    if kind == "subject":
        type_name = _SUBJECT_TYPES.get(obj.get("type"))
        if type_name:
            parts.append(f"类型：{type_name}")
        if obj.get("date"):
            parts.append(f"发行日期：{obj['date']}")
        # 游戏的 platform 枚举是"游戏/软件/DLC/试玩"等形态（与类型重复），不是平台名
        if obj.get("type") != 4:
            platforms = _load_platform_map().get(obj.get("type"), {})
            if obj.get("platform") and platforms.get(obj.get("platform")):
                parts.append(f"平台：{platforms[obj['platform']]}")
        if obj.get("score"):
            parts.append(f"评分：{obj['score']}")
        if obj.get("rank"):
            parts.append(f"排名：{obj['rank']}")
        tags = _tags_text(obj.get("tags"))
        if tags:
            parts.append(f"标签：{tags}")
    elif kind == "person":
        career = obj.get("career")
        if isinstance(career, list) and career:
            parts.append("职业：" + "、".join(str(c) for c in career))
    return " / ".join(parts)


# ---- 关联信息（登场角色 / 制作人员 / 关联作品 / 登场作品 / 参与作品）----

_MAPS_DIR = DEV_DIR / "topics" / "bangumi" / "maps"
if not _MAPS_DIR.is_dir():
    # CI（GitHub Actions）环境没有 dev/topics/（1.7GB 源数据不入库）：映射表
    # （四个 yml，约 70 KB）随仓库放在 dev/tools/bangumi_maps/
    _MAPS_DIR = DEV_DIR / "tools" / "bangumi_maps"
_TYPE_IDS = {"book": 1, "anime": 2, "music": 3, "game": 4, "real": 6}
_ROLE_NAMES = {1: "主角", 2: "配角", 3: "客串"}


def _load_enum_map(
    filename: str, key_ids: dict[str, int] | None = None
) -> dict[int, dict[int, str]]:
    """解析 bangumi/common 的枚举 YAML → {分组: {编号: 中文名}}。

    文件结构规整（缩进固定），按行解析，不引入 yaml 依赖：
        define:
          types:
            anime: &ANCHOR
              1:
                en: "..."
                cn: "原作"
    默认分组键是作品类型（anime/game/… → _TYPE_IDS）；person_relations.yml
    的分组键不同，通过 key_ids 传入（{"person": 0, "character": 1}）。
    缺失文件返回空表（"职位 2004" 这类裸数字没有入库价值，宁可不拼）。
    """
    keys = key_ids if key_ids is not None else _TYPE_IDS
    path = _MAPS_DIR / filename
    if not path.is_file():
        return {}
    result: dict[int, dict[int, str]] = {}
    kind: int | None = None
    number: int | None = None
    in_types = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_types:
            in_types = line.startswith("  types:")
            continue
        if line and not line.startswith(" "):  # 顶层键（staffs: 等）→ types 段结束
            break
        match = re.match(r"^    ([a-z_]+): &", line)
        if match:
            kind = keys.get(match.group(1))
            number = None
            if kind is not None:
                result.setdefault(kind, {})
            continue
        match = re.match(r"^      (\d+):\s*$", line)
        if match and kind is not None:
            number = int(match.group(1))
            continue
        match = re.match(r'^\s+cn:\s*"([^"]*)"', line)
        if match and kind is not None and number is not None:
            name = match.group(1).strip()
            if name:
                result[kind][number] = name
            number = None
    return result


def _top_join(items: list[str], limit: int) -> str:
    """去重取前 limit 条，顿号连接。"""
    seen = list(dict.fromkeys(item for item in items if item))
    return "、".join(seen[:limit])


def _load_staff_priority() -> dict[tuple[int, int], int]:
    """从 subject_staffs.yml 的 presentation 段解析职位展示优先级（越小越靠前）。

    groups 的排列顺序就是官方展示顺序（原作 → 导演 → 脚本 → …），
    摘要里的"制作人员"按它排序，保证导演/原作这类关键职位排前面。
    """
    path = _MAPS_DIR / "subject_staffs.yml"
    if not path.is_file():
        return {}
    result: dict[tuple[int, int], int] = {}
    kind: int | None = None
    group_rank = 0
    in_presentation = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_presentation:
            in_presentation = line.startswith("presentation:")
            continue
        match = re.match(r"^  \*(\w+) :", line)
        if match:
            kind = _TYPE_IDS.get(match.group(1).lower().replace("type_", ""))
            group_rank = 0
            continue
        match = re.match(r"^      (\w+): \[([^\]]*)\]", line)
        if match and kind is not None:
            group_rank += 1
            for index, raw in enumerate(match.group(2).split(",")):
                try:
                    position = int(raw.strip().strip('"'))
                except ValueError:
                    continue
                result[(kind, position)] = group_rank * 100 + index
    return result


def _build_relations(
    keep: dict[str, dict[int, bool]],
    collect_links: bool = True,
) -> tuple[dict[str, dict[int, str]], list[tuple[str, int, str, int, str]]]:
    """聚合关联信息：返回 (摘要文本表, links 边列表)。

    - 摘要文本：每条目一段（登场角色/制作人员/关联作品…），进全文索引与展示；
      截断前按关键度排序（角色主角优先、制作按职位优先级、作品按热度）
    - links：结构化边（只保留**终点也在库内**的记录），供详情工具做完整反查；
      collect_links=False 时不收集（不建 links 表的默认构建省去 160 万条边的内存与耗时）
    """
    subject_name: dict[int, str] = {}
    subject_type: dict[int, int] = {}
    subject_score: dict[int, float] = {}
    with (JSONL_DIR / "subject.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            key = int(obj.get("id") or 0)
            subject_name[key] = _name_text("subject", obj)[0]
            subject_type[key] = int(obj.get("type") or 0)
            try:
                subject_score[key] = float(obj.get("score") or 0)
            except (TypeError, ValueError):
                subject_score[key] = 0.0
    character_name: dict[int, str] = {}
    with (JSONL_DIR / "character.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            character_name[int(obj.get("id") or 0)] = _name_text("character", obj)[0]
    person_name: dict[int, str] = {}
    with (JSONL_DIR / "person.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            person_name[int(obj.get("id") or 0)] = _name_text("person", obj)[0]

    staff_map = _load_enum_map("subject_staffs.yml")
    relation_map = _load_enum_map("subject_relations.yml")
    person_relation_map = _load_enum_map(
        "person_relations.yml", {"person": 0, "character": 1}
    )
    staff_priority = _load_staff_priority()

    keep_subject = keep.get("subject", {})
    keep_character = keep.get("character", {})
    keep_person = keep.get("person", {})

    subject_chars: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    subject_staff: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    subject_rel: dict[int, list[str]] = collections.defaultdict(list)
    char_works: dict[int, list[tuple[float, str]]] = collections.defaultdict(list)
    person_works: dict[int, list[tuple[float, str]]] = collections.defaultdict(list)
    char_actors: dict[int, list[str]] = collections.defaultdict(list)
    links: list[tuple[str, int, str, int, str]] = []
    # collect_links=False（不建 links 表）时，add_link 退化成空操作：关联摘要照常聚合，
    # 只是不再累积 160 万条结构化边，省内存与耗时
    add_link = links.append if collect_links else (lambda *_a: None)

    field_pattern = re.compile(r'"(\w+)":\s*(\d+)')
    with (JSONL_DIR / "subject-characters.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(field_pattern.findall(line))
            sid = int(fields.get("subject_id", 0))
            if not keep_subject.get(sid):
                continue  # 作品不在库：其角色关联不聚合（只展示库内数据）
            cid = int(fields.get("character_id", 0))
            cname = character_name.get(cid, "")
            if not cname:
                continue
            role_id = int(fields.get("type", 0))
            role = _ROLE_NAMES.get(role_id, "")
            subject_chars[sid].append((role_id, f"{cname}（{role}）" if role else cname))
            char_works[cid].append((subject_score.get(sid, 0.0), subject_name.get(sid, "")))
            if keep_character.get(cid):
                add_link(("subject", sid, "character", cid, role or "登场"))

    with (JSONL_DIR / "subject-persons.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(field_pattern.findall(line))
            sid = int(fields.get("subject_id", 0))
            if not keep_subject.get(sid):
                continue  # 作品不在库：其制作人员关联不聚合
            pid = int(fields.get("person_id", 0))
            pname = person_name.get(pid, "")
            if not pname:
                continue
            subject_kind = subject_type.get(sid, 0)
            position = int(fields.get("position", 0))
            staff_name = staff_map.get(subject_kind, {}).get(position, "")
            priority = staff_priority.get((subject_kind, position), 9999)
            label = f"{staff_name} {pname}" if staff_name else pname
            subject_staff[sid].append((priority, label))
            sname = subject_name.get(sid, "")
            person_works[pid].append(
                (
                    subject_score.get(sid, 0.0),
                    f"{sname}（{staff_name}）" if staff_name else sname,
                )
            )
            if keep_person.get(pid):
                add_link(("subject", sid, "person", pid, staff_name or "参与"))

    with (JSONL_DIR / "subject-relations.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(field_pattern.findall(line))
            sid = int(fields.get("subject_id", 0))
            if not keep_subject.get(sid):
                continue  # 作品不在库：其关联作品不聚合
            rid = int(fields.get("related_subject_id", 0))
            rname = subject_name.get(rid, "")
            if not rname:
                continue
            relation_name = relation_map.get(subject_type.get(sid, 0), {}).get(
                int(fields.get("relation_type", 0)), ""
            )
            subject_rel[sid].append(f"{rname}（{relation_name}）" if relation_name else rname)
            if keep_subject.get(rid):
                add_link(("subject", sid, "subject", rid, relation_name or "相关"))

    # person-characters：谁在某作品里饰演某角色（"角色 → 声优"反查的来源）
    with (JSONL_DIR / "person-characters.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(field_pattern.findall(line))
            cid = int(fields.get("character_id", 0))
            if not keep_character.get(cid):
                continue  # 角色不在库：其饰演信息不聚合
            pid = int(fields.get("person_id", 0))
            sid = int(fields.get("subject_id", 0))
            pname = person_name.get(pid, "")
            cname = character_name.get(cid, "")
            if not pname or not cname:
                continue
            sname = subject_name.get(sid, "")
            char_actors[cid].append(f"{pname}（{sname}）" if sname else pname)
            if keep_person.get(pid):
                add_link(
                    ("character", cid, "person", pid, f"饰演（{sname}）" if sname else "饰演")
                )

    # person-relations：人物之间 / 角色之间的关系
    with (JSONL_DIR / "person-relations.jsonlines").open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(field_pattern.findall(line))
            match = re.search(r'"person_type":\s*"(\w+)"', line)
            group = 0 if (match.group(1) if match else "prsn") == "prsn" else 1
            pid = int(fields.get("person_id", 0))
            rid = int(fields.get("related_person_id", 0))
            relation_name = person_relation_map.get(group, {}).get(
                int(fields.get("relation_type", 0)), ""
            )
            if not relation_name:
                continue
            if group == 0:
                if keep_person.get(pid) and keep_person.get(rid):
                    add_link(("person", pid, "person", rid, relation_name))
            elif keep_character.get(pid) and keep_character.get(rid):
                add_link(("character", pid, "character", rid, relation_name))

    result: dict[str, dict[int, str]] = {"subject": {}, "character": {}, "person": {}}
    for sid in set(subject_chars) | set(subject_staff) | set(subject_rel):
        parts: list[str] = []
        chars_list = sorted(subject_chars.get(sid, []), key=lambda item: item[0])
        chars = _top_join([text for _, text in chars_list], 20)
        if chars:
            parts.append(f"登场角色：{chars}")
        staff_list = sorted(subject_staff.get(sid, []), key=lambda item: item[0])
        staff = _top_join([text for _, text in staff_list], 20)
        if staff:
            parts.append(f"制作人员：{staff}")
        related = _top_join(subject_rel.get(sid, []), 12)
        if related:
            parts.append(f"关联作品：{related}")
        if parts:
            result["subject"][sid] = " / ".join(parts)
    for cid, works in char_works.items():
        works.sort(key=lambda item: item[0], reverse=True)
        parts = []
        text = _top_join([item[1] for item in works], 12)
        if text:
            parts.append(f"登场作品：{text}")
        actors = _top_join(char_actors.get(cid, []), 5)
        if actors:
            parts.append(f"饰演者：{actors}")
        if parts:
            result["character"][cid] = " / ".join(parts)
    for pid, works in person_works.items():
        works.sort(key=lambda item: item[0], reverse=True)
        text = _top_join([item[1] for item in works], 12)
        if text:
            result["person"][pid] = f"参与作品：{text}"
    return result, links


def _cn_from_infobox(raw) -> str:
    """从 infobox 里取中文名。

    character/person 没有 name_cn 字段，中文名写在 infobox 的
    "简体中文名/中文名" 里（如 鲁路修·兰佩路基）。
    """
    if not raw:
        return ""
    for key in ("简体中文名", "中文名"):
        match = re.search(rf"\|\s*{key}\s*=\s*([^\n|]+)", str(raw))
        if match:
            return re.sub(r"[{}\[\]]", "", match.group(1)).strip()
    return ""


def _name_text(kind: str, obj: dict) -> tuple[str, str]:
    """(展示名, 检索用名文本)。"""
    name = str(obj.get("name") or "").strip()
    name_cn = str(obj.get("name_cn") or "").strip() or _cn_from_infobox(obj.get("infobox"))
    display = name_cn or name
    if name and name_cn:
        return display, f"{name_cn} {name}"
    return display, display


def _build_episodes(conn: sqlite3.Connection, keep_subject: dict[int, bool]) -> int:
    """写入保留作品的剧集（含 OP/ED 等，type 字段区分，查询端过滤）。

    episode.jsonlines 有 169 万行；只留库内作品的部分，供"某条目有哪些集"。
    """
    path = JSONL_DIR / "episode.jsonlines"
    if not path.is_file():
        return 0
    batch: list[tuple] = []
    count = 0

    def _flush() -> None:
        nonlocal count
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)", batch
            )
            count += len(batch)
            batch.clear()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = int(obj.get("subject_id") or 0)
            if not keep_subject.get(sid):
                continue
            name = str(obj.get("name") or "").strip()
            name_cn = str(obj.get("name_cn") or "").strip()
            if not name and not name_cn:
                continue
            try:
                sort_value = float(obj.get("sort") or 0)
            except (TypeError, ValueError):
                sort_value = 0.0
            batch.append(
                (
                    int(obj.get("id") or 0),
                    sid,
                    name,
                    name_cn,
                    str(obj.get("airdate") or ""),
                    int(obj.get("type") or 0),
                    sort_value,
                )
            )
            if len(batch) >= 5000:
                _flush()
    _flush()
    return count


def _build(
    limit: int,
    kinds: tuple[str, ...],
    min_votes: int,
    new_months: int,
    oldest_votes: int,
    oldest_year: int,
    min_favorites: int = 100,
    oldest_favorites: int = 1000,
    with_graph: bool = False,
    with_subject_infobox: bool = False,
    source_date: str = "",
) -> int:
    """流式读取 jsonlines 并写入 SQLite。

    保留规则 = 打分人数 ≥ 按发行年份插值的门槛（2000 年及更早要 oldest_votes，
    今年要 min_votes），**或**收藏人次 ≥ 同曲线的收藏门槛（2000 年及更早要
    oldest_favorites，今年要 min_favorites），或命中新作豁免（近 new_months
    个月已上映；未上映限 3 年内且收藏>0）。char/person 关联作品中任一达标即保留。

    体积开关（默认都关，省空间；需要时用命令行再开）：
    - with_graph：建 links/episodes（结构化反查、逐集数据），当前检索工具不读；
    - with_subject_infobox：存 subject 的整块 infobox；其有用信息（别名→FTS name、
      导演→relations、中文名→name_cn、官网等）已各归其位，默认不重复存。
      角色/人物的 infobox 始终是资料卡（性别/生日/事务所…别处没有），一直保留。
    """
    overall_start = time.perf_counter()
    # 连 -wal/-shm 一起清掉：残留的 WAL 会被新建的库重放，污染数据
    for suffix in ("", "-wal", "-shm"):
        pathlib.Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=new_months * 30)
    horizon = today + datetime.timedelta(days=3 * 365)
    print(
        f"[保留判定] 打分门槛 {min_votes}（今年）→ {oldest_votes}（{oldest_year} 年及更早）、"
        f"收藏门槛 {min_favorites}（今年）→ {oldest_favorites}（{oldest_year} 年及更早）、"
        f"新作窗口 {new_months} 个月、未上映限 3 年内且收藏>0 ...",
        flush=True,
    )
    stage = time.perf_counter()
    keep = _build_heat_map(
        cutoff,
        horizon,
        min_votes,
        oldest_votes,
        oldest_year,
        min_favorites,
        oldest_favorites,
    )
    print(f"[保留判定] 完成（{time.perf_counter() - stage:.0f}s）", flush=True)

    print("[关联信息] 聚合登场角色 / 制作人员 / 关联作品 ...", flush=True)
    stage = time.perf_counter()
    relations, links = _build_relations(keep, collect_links=with_graph)
    edge_note = f"{len(links)} 条" if with_graph else "（未建 links 表，跳过）"
    print(
        f"[关联信息] 摘要 {sum(len(v) for v in relations.values())} 条、"
        f"结构化边 {edge_note}（{time.perf_counter() - stage:.0f}s）",
        flush=True,
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    # 建库是一次性离线写，页缓存开大、临时表放内存，减少刷盘（配合 wal_checkpoint 收尾）
    conn.execute("PRAGMA cache_size=-262144")  # ~256 MB 页缓存
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.executescript(_SCHEMA)
    if with_graph:
        conn.executescript(_SCHEMA_GRAPH)

    kept = skipped = 0
    started = time.perf_counter()
    for kind in kinds:
        path = JSONL_DIR / f"{kind}.jsonlines"
        if not path.is_file():
            print(f"[跳过] {path.name} 不存在")
            continue
        keep_map = keep.get(kind, {})
        relation_map = relations.get(kind, {})
        done = 0
        stage = time.perf_counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit and done >= limit:
                    break
                # 预过滤：行首 {"id":N 正则取 id，先判保留；被打分门槛刷掉的行（占多数）
                # 连 json.loads 都省掉，只在命中后才整行解析
                head = _ID_PREFIX_RE.match(line)
                key = int(head.group(1)) if head else 0
                if not keep_map.get(key, False):
                    skipped += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                display, name_text = _name_text(kind, obj)
                if not display:
                    continue
                summary = str(obj.get("summary") or "").strip()
                alias = _alias_text(obj.get("infobox"))
                relation = relation_map.get(key, "")
                tags_text = _tags_text(obj.get("tags"))
                # infobox 落库口径：subject 默认不存整块（别名/导演/中文名等已各归其位），
                # 角色/人物始终存——那是它们唯一承载性别/生日/事务所等资料的地方，去链接后入库
                if kind == "subject":
                    infobox_stored = (
                        kb_text.strip_links(_infobox_text(obj.get("infobox")))
                        if with_subject_infobox
                        else ""
                    )
                else:
                    infobox_stored = kb_text.strip_links(_infobox_text(obj.get("infobox")))
                # 结构化列（仅 subject 有值）
                date_text = str(obj.get("date") or "").strip() if kind == "subject" else ""
                subject_kind = int(obj.get("type") or 0) if kind == "subject" else None
                platform = int(obj.get("platform") or 0) if kind == "subject" else 0
                rank_value = int(obj.get("rank") or 0) if kind == "subject" else 0
                try:
                    score_value = float(obj.get("score") or 0) if kind == "subject" else 0.0
                except (TypeError, ValueError):
                    score_value = 0.0
                cursor = conn.execute(
                    "INSERT INTO docs(kind, source_id, name, name_cn, summary, infobox, "
                    "relations, extra, subject_type, date, ym, platform, rank, score, favorite) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        kind,
                        int(obj.get("id") or 0),
                        str(obj.get("name") or ""),
                        display,
                        summary,
                        infobox_stored,
                        relation,
                        _extra_text(kind, obj),
                        subject_kind,
                        date_text,
                        date_text[:7] if len(date_text) >= 7 else None,
                        platform or None,
                        rank_value or None,
                        score_value or None,
                        _favorite_count(obj) if kind == "subject" else None,
                    ),
                )
                rowid = cursor.lastrowid
                # 索引文本＝名称+别名+简介+关联+标签：职位/关系名与标签都要可检索
                # （"这部动画的导演是谁"靠"制作人员：导演 XXX"命中，"催泪"靠标签命中）；
                # infobox 其余字段只存 docs 供展示，不进全文索引
                conn.execute(
                    "INSERT INTO fts(rowid, name, body) VALUES (?,?,?)",
                    (
                        rowid,
                        " ".join(kb_text.tokenize(f"{name_text} {alias}")),
                        " ".join(
                            kb_text.tokenize(f"{summary} {relation} {tags_text}".strip())
                        ),
                    ),
                )
                done += 1
                kept += 1
                if done % 2000 == 0:
                    conn.commit()
                    speed = done / max(0.001, time.perf_counter() - started)
                    print(f"  [{kind}] 入库 {done} 条（{speed:.0f} 条/秒）", flush=True)
        conn.commit()
        print(
            f"[{_KIND_NAMES.get(kind, kind)}] 入库 {done} 条"
            f"（{time.perf_counter() - stage:.0f}s）"
        )

    if with_graph and links:
        print(f"[关联索引] 写入 {len(links)} 条结构化边 ...", flush=True)
        stage = time.perf_counter()
        conn.executemany("INSERT INTO links VALUES (?,?,?,?,?)", links)
        conn.commit()
        print(f"[关联索引] 写入完成（{time.perf_counter() - stage:.0f}s）", flush=True)

    if with_graph:
        print("[剧集] 写入保留作品的剧集 ...", flush=True)
        stage = time.perf_counter()
        episode_count = _build_episodes(conn, keep.get("subject", {}))
        conn.commit()
        print(f"[剧集] {episode_count} 条（{time.perf_counter() - stage:.0f}s）", flush=True)

    elapsed = time.perf_counter() - overall_start
    # 数据源日期（/kb_stats 展示用）：参数未给时取 jsonlines 的最后修改日期
    # （≈ 数据源导出落地日），再兜底为今天；写入 meta 表随库分发
    if not source_date:
        try:
            source_date = datetime.date.fromtimestamp(
                max(p.stat().st_mtime for p in JSONL_DIR.glob("*.jsonlines"))
            ).isoformat()
        except (ValueError, OSError):
            source_date = datetime.date.today().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('source_date', ?)",
        (source_date,),
    )
    conn.commit()
    # 合并 FTS 分段（分批写入会留下很多小段，optimize 合成单段、顺带回收），
    # 再把 WAL 并回主库转单文件模式，最后 VACUUM 压实空闲页：产出更小的单文件，
    # 且此后只读查询不会再生成 -wal/-shm（WAL 下只读打开也会生成这两个附属文件）
    conn.commit()
    conn.execute("INSERT INTO fts(fts) VALUES('optimize')")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    for suffix in ("-shm", "-wal"):
        pathlib.Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
    size_mb = DB_PATH.stat().st_size / 1048576
    print(
        f"[完成] 入库 {kept} 条（打分过滤 {skipped} 条）→ {DB_PATH.name}"
        f"（{size_mb:.1f} MB，耗时 {elapsed:.0f}s）"
    )
    return 0


def _search(query: str, top_k: int = 5) -> int:
    """检索测试。"""
    tokens = " ".join(kb_text.tokenize(query, for_query=True))
    if not tokens:
        print("查询未产生有效词元")
        return 1
    print(f"查询: {query!r} → 词元: {tokens}")
    conn = sqlite3.connect(DB_PATH)
    # 排序：名称精确匹配优先（"水树奈奈"直接命中的条目/人物排在"简介里提到"的前面），
    # 其余按加权 bm25（name 列权重 8，名字命中优先于简介里顺带提到）
    rows = conn.execute(
        "SELECT d.kind, d.source_id, d.name, d.name_cn, d.extra, d.relations, "
        "bm25(fts, 8.0, 1.0) AS score "
        "FROM fts JOIN docs d ON d.rowid = fts.rowid "
        "WHERE fts MATCH ? "
        "ORDER BY (d.name_cn = ? OR d.name = ?) DESC, score LIMIT ?",
        (tokens, query, query, top_k),
    ).fetchall()
    for kind, source_id, name, name_cn, extra, relations, score in rows:
        print(f"  [{_KIND_NAMES.get(kind, kind)}#{source_id}] {name_cn or name} ({name})")
        if extra:
            print(f"      {extra[:100]}")
        if relations:
            print(f"      {relations[:170]}")
    if not rows:
        print("  未命中")
    return 0


def _delete(spec: str) -> int:
    """按 "kind:id" 删除条目（逗号分隔多个，如 subject:8,character:22）。

    docs 与 fts 成对删除；删完用 --vacuum 回收空间。
    """
    if not DB_PATH.is_file():
        print(f"[错误] 库不存在：{DB_PATH}")
        return 1
    targets: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kind, _, ident = part.partition(":")
        try:
            targets.append((kind.strip(), int(ident)))
        except ValueError:
            print(f"[跳过] 无法解析：{part}（格式应为 kind:id，如 subject:8）")
    if not targets:
        print("没有可删除的目标")
        return 1
    conn = sqlite3.connect(DB_PATH)
    removed = 0
    for kind, ident in targets:
        rows = conn.execute(
            "SELECT rowid FROM docs WHERE kind = ? AND source_id = ?", (kind, ident)
        ).fetchall()
        if not rows:
            print(f"[未找到] {kind}:{ident}")
            continue
        for (rowid,) in rows:
            conn.execute("DELETE FROM fts WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM docs WHERE rowid = ?", (rowid,))
            removed += 1
        print(f"[删除] {kind}:{ident} → {len(rows)} 条")
    conn.commit()
    conn.close()
    print(f"[完成] 共删除 {removed} 条（用 --vacuum 回收空间）")
    return 0


def _vacuum() -> int:
    """回收数据库文件空间（删除条目后使用，大库需几十秒）。"""
    if not DB_PATH.is_file():
        print(f"[错误] 库不存在：{DB_PATH}")
        return 1
    before = DB_PATH.stat().st_size / 1048576
    conn = sqlite3.connect(DB_PATH)
    conn.execute("VACUUM")
    conn.close()
    after = DB_PATH.stat().st_size / 1048576
    print(f"[完成] VACUUM：{before:.1f} MB → {after:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 bangumi SQLite FTS5 检索库")
    parser.add_argument("--limit", type=int, default=0, help="每类最多入库多少条（试跑用）")
    parser.add_argument(
        "--types", default=",".join(_KINDS), help="要处理的类型，逗号分隔（subject,character,person）"
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=10,
        help="打分门槛（今年及更新作品）；越早的作品门槛按年份线性递增",
    )
    parser.add_argument(
        "--oldest-votes",
        type=int,
        default=100,
        help="最早年份及更早作品的打分门槛（默认 100）",
    )
    parser.add_argument(
        "--min-favorites",
        type=int,
        default=100,
        help="收藏门槛（今年及更新作品）；与打分门槛为「或」关系，越早按年份线性递增",
    )
    parser.add_argument(
        "--oldest-favorites",
        type=int,
        default=1000,
        help="最早年份及更早作品的收藏门槛（默认 1000）",
    )
    parser.add_argument(
        "--oldest-year",
        type=int,
        default=2000,
        help="门槛曲线起点年份（该年及更早用 oldest-votes）",
    )
    parser.add_argument(
        "--new-months",
        type=int,
        default=3,
        help="新作品豁免窗口（月）：此窗口内已上映的条目不受打分门槛限制",
    )
    parser.add_argument(
        "--source-date",
        default="",
        help="数据源日期（如 2026-09-08），写入 meta 供 /kb_stats 显示；默认取 jsonlines 最后修改日期",
    )
    parser.add_argument("--query", default="", help="检索测试（不构建）")
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="额外建 links/episodes 表（结构化反查、逐集数据）；当前检索工具不读，默认不建",
    )
    parser.add_argument(
        "--with-subject-infobox",
        action="store_true",
        help="额外存 subject 的整块 infobox（其有用信息已各归其位，默认不重复存）",
    )
    parser.add_argument(
        "--delete",
        default="",
        help='删除条目：kind:id 逗号分隔（如 "subject:8,character:22"）',
    )
    parser.add_argument("--vacuum", action="store_true", help="回收数据库空间（删除条目后使用）")
    args = parser.parse_args()

    if args.query:
        return _search(args.query)
    if args.delete:
        return _delete(args.delete)
    if args.vacuum:
        return _vacuum()
    kinds = tuple(part.strip() for part in args.types.split(",") if part.strip())
    return _build(
        args.limit,
        kinds,
        args.min_votes,
        args.new_months,
        args.oldest_votes,
        args.oldest_year,
        min_favorites=args.min_favorites,
        oldest_favorites=args.oldest_favorites,
        with_graph=args.with_graph,
        with_subject_infobox=args.with_subject_infobox,
        source_date=args.source_date,
    )


if __name__ == "__main__":
    sys.exit(main())
