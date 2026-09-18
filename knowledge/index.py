"""运行时检索引擎：BM25 内存索引的加载、合并与检索。

只负责"读索引 → 打分 → 返回片段"这条运行时链路；txt → 索引的离线构建在
`builder.py` / `cleaning.py` / `parsing.py`，运行时不导入那些模块。
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .text import (
    _OVERVIEW_SECTIONS,
    _chars_near_in_order,
    _clean_query,
    _query_phrases,
    _truncate_around,
    _wants_overview,
    jieba,
    tokenize,
)

ENGINE_VERSION = 1
# 索引格式版本：
#   1 = chunks 只有 title/text，加载时重新分词（已知慢：数千片段要 5 秒级）
#   2 = chunks 附带词频表（text_tf/title_tf），加载直接重建倒排，不重新分词
# v1 文件仍可加载（回退到重新分词），构建一次后即升级为 v2。
INDEX_FORMAT_VERSION = 2

# 索引文件的元信息排在 chunks 之前；主题名（display_name）就在最前面的几十字节里。
# 只为拼 Tool 描述而完整解析十几 MB 的索引（含词频表）是浪费，故按前缀轻量读取。
_THEME_PREFIX_CHARS = 8192
_DISPLAY_NAME_RE = re.compile(r'"display_name"\s*:\s*"((?:[^"\\]|\\.)*)"')

# 主观偏好类问法（"哪个角色可爱/好看""你喜欢谁"）不含"萌点"这类知识库维度词，
# 直接检索会漏掉萌点数据；识别后扩展出"萌点"参与检索，
# 命中构建期合成的「角色萌点一览」目录片段与各角色基本信息的萌点行
_SUBJECTIVE_ASPECT_RE = re.compile(r"可爱|好看|漂亮|萌|颜值|魅力|喜欢")

# "萌王/燃王"是萌战（世萌/B萌）对冠军的称呼，而知识库的历年得主表里写的是"冠军"，
# 扩展出来让"世萌 萌王是谁"能命中历年得主表
_MOE_CROWN_RE = re.compile(r"萌王|燃王")

_ROSTER_TERM = "角色"  # "主题名 + 角色"形态的查询词（"绝区零 角色"）


def _is_roster_query(query_terms: List[str], theme_names: set) -> Optional[str]:
    """判断是否为"XX 有哪些角色"式的列表查询，命中则返回对应主题名。

    查询里除"角色"外的词都落在某个主题名内（"崩坏 星穹铁道 角色" ↔
    "崩坏：星穹铁道"）；简称/别称（"崩铁""星铁"）按字符顺序近似匹配。
    具体角色名（"三月七 角色"）不是列表意图——用户要看的是该角色本人。
    """
    return _match_theme_query(query_terms, theme_names, lambda term: term == _ROSTER_TERM)


def _is_meme_query(query_terms: List[str], theme_names: set) -> Optional[str]:
    """判断是否为"XX 有什么梗"式的梗查询，命中则返回对应主题名。

    判断方式与列表查询相同，只是意图词从"角色"换成含"梗"的词
    （"有什么梗"经疑问词清理后通常是"梗"单字）。
    """
    return _match_theme_query(
        query_terms, theme_names, lambda term: "梗" in term
    )


def _match_theme_query(
    query_terms: List[str], theme_names: set, is_intent: Any
) -> Optional[str]:
    """列表/梗查询的公共判定：查询里除意图词外的词都落在某个主题名内
    （简称按字符顺序近似匹配），命中则返回该主题名。
    """
    rest = [term for term in query_terms if not is_intent(term)]
    if not rest or not theme_names:
        return None
    for flat in theme_names:
        if all(
            term in flat or (len(term) >= 2 and _chars_near_in_order(term, flat))
            for term in rest
        ):
            return flat
    return None


class KnowledgeIndex:
    """基于 BM25 的内存知识索引，支持 JSON 持久化。"""

    BM25_K1 = 1.5
    BM25_B = 0.75
    TITLE_BOOST = 2.2  # 小节标题命中额外权重（"实体 / 属性"式查询靠它精确落位）
    PHRASE_BONUS = 0.18  # 查询短语在原文整串命中的加成
    PHRASE_BONUS_MAX = 3  # 最多叠加几个短语（防止长查询堆分）
    CHAR_FALLBACK_SCORE = 1.8  # 词面不重合时按字符近似匹配的基础分（×倍率使用）
    OVERVIEW_BOOST = 1.6  # "X 是什么"提问下，概述性小节（简介/基本信息）的权重
    PIN_TITLE_LIMIT = 2  # 置顶意图下，同名片段最多几条（防列表霸榜）

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[int]] = {}  # term → 片段下标，检索时裁剪候选
        self.avg_len = 1.0
        self.avg_title_len = 1.0
        self.built_at = ""
        self.display_name = ""
        self.theme_names: set = set()  # 各主题 display_name（去空白），供列表意图判断
        self.sources: List[Dict[str, Any]] = []
        self.topics: List[str] = []  # 一级章节主题清单（按文档顺序去重）
        self.broken_files: List[str] = []  # 合并加载时无法读取/不兼容而跳过的文件名

    @property
    def size(self) -> int:
        """索引内片段数量。"""
        return len(self.entries)

    # ── 构建 ──

    @staticmethod
    def _register_terms(titles: List[str]) -> None:
        """把小节标题里的专有名词加进 jieba 词典。

        否则"云无留迹的过客""冬去煦至"这类专名会被切碎，查询时整词命中不到。
        重复添加是幂等的。
        """
        for title in titles:
            for part in title.split(" / "):
                part = part.strip()
                if (
                    2 <= len(part) <= 12
                    and re.search(r"[\u4e00-\u9fff]", part)
                    and not part.isdigit()
                ):
                    jieba.add_word(part)

    @staticmethod
    def _make_entry(
        title: str, text: str, text_tokens: List[str], title_tokens: List[str]
    ) -> Dict[str, Any]:
        """由分词结果构造片段条目（词频表随索引持久化，加载时免于重新分词）。"""
        return {
            "title": title,
            "text": text,
            "text_tf": dict(Counter(text_tokens)),
            "title_tf": dict(Counter(title_tokens)),
            "len": max(1, len(text_tokens)),
            "title_len": max(1, len(title_tokens)),
            "lower": text.lower(),
        }

    @staticmethod
    def _entry_from_payload(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """由持久化片段（带词频表）还原条目，不重新分词。"""
        text = str(chunk.get("text", "")).strip()
        if not text:
            return None
        title = str(chunk.get("title", "")).strip()
        text_tf = {str(k): int(v) for k, v in dict(chunk.get("text_tf") or {}).items()}
        title_tf = {str(k): int(v) for k, v in dict(chunk.get("title_tf") or {}).items()}
        if not text_tf and not title_tf:
            return None
        return {
            "title": title,
            "text": text,
            "text_tf": text_tf,
            "title_tf": title_tf,
            "len": max(1, sum(text_tf.values())),
            "title_len": max(1, sum(title_tf.values())),
            "lower": text.lower(),
        }

    def _rebuild_statistics(self) -> None:
        """由 entries 的词频表重建倒排表与 BM25 统计（不涉及分词）。

        正文与小节标题分开统计：标题命中单独加权（比把它当重复 token 更可控，
        也让"三月七 / 星魂"这类"实体 / 属性"查询能精确落在对应小节）。
        """
        total = len(self.entries)
        document_freq: Counter = Counter()
        postings: Dict[str, List[int]] = {}
        for position, entry in enumerate(self.entries):
            for term in set(entry["text_tf"]) | set(entry["title_tf"]):
                document_freq[term] += 1
                # 用紧凑数组存片段下标：六十万个倒排项从 ~20 MB 降到 ~2 MB
                postings.setdefault(term, array("i")).append(position)
        self.postings = postings
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }
        self.avg_len = (sum(entry["len"] for entry in self.entries) / total) if total else 1.0
        self.avg_title_len = (
            (sum(entry["title_len"] for entry in self.entries) / total) if total else 1.0
        )
        topics: List[str] = []
        for entry in self.entries:
            chapter = str(entry.get("title", "")).split(" / ")[0].strip()
            if chapter and chapter != "前言" and chapter not in topics:
                topics.append(chapter)
        self.topics = topics

    def _load_from_payload(self, chunks: List[Dict[str, Any]]) -> None:
        """由持久化片段（带词频表）恢复索引：只重建倒排，不重新分词。"""
        # 词表仍要注册进 jieba：查询侧分词口径必须与索引一致，
        # 否则"云无留迹的过客"这类专名会被切碎，整词命中不到
        self._register_terms([str(chunk.get("title", "")) for chunk in chunks])
        self.entries = []
        for chunk in chunks:
            entry = self._entry_from_payload(chunk)
            if entry is not None:
                self.entries.append(entry)
        self._rebuild_statistics()

    def build(self, chunks: List[Dict[str, str]], sources: List[Dict[str, Any]]) -> None:
        """由知识片段列表构建倒排统计（需要分词，用于 txt → 索引）。"""
        # 先把小节标题里的专有名词教给 jieba，再分词——顺序不能反，
        # 否则索引里的 token 与查询侧的分词口径会不一致
        self._register_terms([str(chunk.get("title", "")) for chunk in chunks])

        self.entries = []
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            title = str(chunk.get("title", "")).strip()
            text_tokens = tokenize(text)
            title_tokens = tokenize(title)
            if not text_tokens and not title_tokens:
                continue
            self.entries.append(self._make_entry(title, text, text_tokens, title_tokens))

        self._rebuild_statistics()
        self.built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.sources = list(sources)

    # ── 持久化 ──

    def save(self, path: Path) -> None:
        """把索引写入文件（v2：连同词频表一起存，加载时免于重新分词）。

        词频表让文件变大（gzip 后反而比 v1 还小）；.gz 后缀自动压缩。
        """
        payload = {
            "format": INDEX_FORMAT_VERSION,
            "engine_version": ENGINE_VERSION,
            "built_at": self.built_at,
            "display_name": self.display_name,
            "sources": self.sources,
            "chunks": [
                {
                    "title": entry["title"],
                    "text": entry["text"],
                    "text_tf": entry["text_tf"],
                    "title_tf": entry["title_tf"],
                }
                for entry in self.entries
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, ensure_ascii=False)
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(data)
        else:
            path.write_text(data, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeIndex":
        """从索引文件加载（v2 直接读词频表，v1 回退重新分词，均可加载）。"""
        raw = read_index_file(path)
        fmt = int(raw.get("format", 0))
        if fmt not in (1, INDEX_FORMAT_VERSION):
            raise ValueError("索引格式版本不兼容，需要重建")
        if int(raw.get("engine_version", 0)) != ENGINE_VERSION:
            raise ValueError("索引引擎版本不兼容，需要重建")
        index = cls()
        index.built_at = str(raw.get("built_at", ""))
        index.display_name = str(raw.get("display_name", ""))
        index.theme_names = {re.sub(r"\s+", "", index.display_name)}
        index.sources = [item for item in raw.get("sources", []) if isinstance(item, dict)]
        items = [item for item in raw.get("chunks", []) if isinstance(item, dict)]
        if fmt >= 2:
            index._load_from_payload(items)
        else:
            legacy_chunks = [
                {"title": str(item.get("title", "")), "text": str(item.get("text", ""))}
                for item in items
            ]
            index.build(legacy_chunks, index.sources)
        return index

    @classmethod
    def load_combined(cls, paths: List[Path]) -> "KnowledgeIndex":
        """合并加载多个索引文件（萌百科各主题），重建统一的 BM25 统计。

        v2 文件自带词频表，合并时不重新分词；v1 文件回退分词（下次构建
        自动升级为 v2）。两种格式混用也能正确合并。

        容错：单个文件损坏 / 格式不兼容时跳过并记入 ``broken_files``（由调用方
        打日志），不影响其余文件；全部失败时返回空索引，调用方按"未收录数据"处理。
        """
        index = cls()
        entries: List[Dict[str, Any]] = []
        legacy_chunks: List[Dict[str, str]] = []
        sources: List[Dict[str, Any]] = []
        built_times: List[str] = []

        for path in paths:
            try:
                raw = read_index_file(path)
                fmt = int(raw.get("format", 0))
                if fmt not in (1, INDEX_FORMAT_VERSION) or int(
                    raw.get("engine_version", 0)
                ) != ENGINE_VERSION:
                    raise ValueError("索引格式不兼容（需要重建）")
            except Exception as e:  # noqa: BLE001 — 任何坏文件都只跳过，不拖垮整体
                index.broken_files.append(f"{path.name}（{e}）")
                continue
            index.theme_names.add(re.sub(r"\s+", "", str(raw.get("display_name", ""))))
            index.theme_names.discard("")
            for item in raw.get("chunks", []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                if fmt >= 2:
                    entry = cls._entry_from_payload(item)
                    if entry is None:
                        continue
                    entries.append(entry)
                else:
                    legacy_chunks.append(
                        {"title": str(item.get("title", "")), "text": text}
                    )
            for source in raw.get("sources", []):
                entry = dict(source) if isinstance(source, dict) else {"name": str(source)}
                entry["origin"] = path.name
                sources.append(entry)
            if raw.get("built_at"):
                built_times.append(str(raw.get("built_at")))

        # 词表先注册（查询侧口径），再对旧格式片段补分词
        index._register_terms(
            [entry["title"] for entry in entries]
            + [chunk["title"] for chunk in legacy_chunks]
        )
        for chunk in legacy_chunks:
            text_tokens = tokenize(chunk["text"])
            title_tokens = tokenize(chunk["title"])
            if not text_tokens and not title_tokens:
                continue
            entries.append(
                cls._make_entry(chunk["title"], chunk["text"], text_tokens, title_tokens)
            )
        index.entries = entries
        index._rebuild_statistics()
        index.sources = sources
        index.built_at = max(built_times) if built_times else time.strftime("%Y-%m-%d %H:%M:%S")
        return index

    # ── 检索 ──

    def search(self, query: str, top_k: int = 4, max_chars: int = 300) -> List[Dict[str, Any]]:
        """BM25 检索，返回 [{title, text, score}]。

        在经典 BM25 之上做了几件针对性的事：
        - 正文与小节标题分开打分，标题命中额外加权（"三月七 星魂"能精确落到对应小节）；
        - 用倒排表裁剪候选，只给命中过查询词的片段算分；
        - 对未登录词（简称/别称，如"崩铁""星铁"）退到字级匹配兜底；
        - 原文短语整串命中额外加分；
        - 返回文本围绕命中位置开窗，避免答案被截在片段之外。
        """
        query = str(query or "").strip()
        if not query or not self.entries:
            return []
        # 索引是静态的，"今年/去年"这类相对时间词必须换成具体年份才能命中
        # （"今年人气角色" → "2026人气角色"）
        current_year = time.localtime().tm_year
        query = re.sub(r"今年|本年度|本届", f"{current_year}年", query)
        query = re.sub(r"去年|上年度|上届", f"{current_year - 1}年", query)
        # 先剥离疑问/语气成分再分词，否则"崩铁是什么游戏"会被切成"崩铁是"+"什么游戏"
        cleaned_query = _clean_query(query)
        query_terms = list(dict.fromkeys(tokenize(cleaned_query, for_query=True)))
        if not query_terms:
            query_terms = list(dict.fromkeys(tokenize(query, for_query=True)))
        if not query_terms:
            return []
        # 主观问法（"哪个可爱""你喜欢谁"）补上"萌点"维度词，让它参与候选与打分
        if _SUBJECTIVE_ASPECT_RE.search(query) and "萌点" not in query_terms:
            query_terms.append("萌点")
        # 萌战冠军问法（"萌王/燃王"直说，或"B萌 2026 冠军"这类"萌战词+冠军"）
        # 补上"冠军"是得主表表头用词，让各类冠军问法都能命中得主/结果表
        if (
            _MOE_CROWN_RE.search(query)
            or ("冠军" in query_terms and re.search(r"世萌|B萌|b萌|萌战", query, re.IGNORECASE))
        ) and "冠军" not in query_terms:
            query_terms.append("冠军")

        flat_query = re.sub(r"\s+", "", cleaned_query).lower()
        phrases = _query_phrases(cleaned_query)
        wants_overview = _wants_overview(query)
        # 置顶意图："绝区零 角色"→登场角色列表；"绝区零 有什么梗"→用语与梗；
        # "世萌 萌王是谁"→历年得主表。pin_theme 限定主题（萌王不限）。
        pin_re = None
        pin_theme = _is_roster_query(query_terms, self.theme_names)
        if pin_theme is not None:
            # 置顶两个合成目录：登场角色名单 + 萌点目录——它们含该主题全部角色，
            # 单独靠 BM25 会被主词条列表块压出去。条数缩到 2（名单+目录即全量
            # 信息），单片段窗口放宽避免名单截断
            pin_re = re.compile("登场角色一览|角色萌点一览")
            top_k = min(top_k, 2)
            max_chars = max(max_chars, 700)
        else:
            meme_theme = _is_meme_query(query_terms, self.theme_names)
            if meme_theme is not None:
                pin_re, pin_theme = re.compile("用语与梗"), meme_theme
            elif _MOE_CROWN_RE.search(query) or (
                "冠军" in query_terms
                and re.search(r"世萌|B萌|b萌|萌战", query, re.IGNORECASE)
            ):
                pin_re = re.compile("得主|历年结果|举办日期及结果")
            elif "人气" in query_terms and re.search(
                r"世萌|B萌|b萌|萌战", query, re.IGNORECASE
            ):
                # "今年世萌谁人气高"→进行中赛季的小组赛名次表（当前人气）
                pin_re = re.compile("小组赛|赛果")

        candidates: set[int] = set()
        unmatched: List[str] = []
        for term in query_terms:
            posting = self.postings.get(term)
            if posting:
                candidates.update(posting)
            else:
                unmatched.append(term)

        scored: List[tuple[int, float]] = []
        for position in candidates:
            entry = self.entries[position]
            score = self._score_entry(entry, query_terms, flat_query, phrases)
            # "X 是什么"优先给概述性小节（"崩铁是什么游戏" → 词条的"基本信息"节，
            # 那里写着"常用译名：崩铁、星铁"，而不是正文里顺带提到崩铁的梗条目）
            if wants_overview and any(
                part in entry["title"] for part in _OVERVIEW_SECTIONS
            ):
                score *= self.OVERVIEW_BOOST
            # 简称/别称兜底：查询词与小节标题字面不重合时（"崩铁""星铁" ↔
            # "崩坏：星穹铁道"），按字符顺序与间距做近似匹配。标题很短，
            # 直接扫字符串即可，不必预存字符集合（那是上百 MB 的内存开销）
            title = entry["title"]
            fallback_terms = set(unmatched)
            for term in query_terms:
                if len(term) < 2 or entry["title_tf"].get(term):
                    continue  # 标题已整词命中，无需近似
                if _chars_near_in_order(term, title):
                    score += self.CHAR_FALLBACK_SCORE * 3.0
                elif term in fallback_terms and all(char in title for char in term):
                    score += self.CHAR_FALLBACK_SCORE * 1.8
            if score > 0.0:
                scored.append((position, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        # 查询即角色全名（"艾莲·乔"、"艾莲 乔"）时，把该角色页的基本信息
        # 片段提到最前——搜角色应先看到本名/别号/萌点等键值行，而不是台词
        # 或角色相关节。判定：查询去分隔符后与某片段的章节名完全一致
        # （后缀匹配会把"星神"这类种族/词条名误当角色名，故只认精确相等）
        if pin_re is None:
            merged_query = re.sub(r"[\s·・\-—]+", "", flat_query)
            if len(merged_query) >= 2:
                profile: tuple[int, float] | None = None
                for item in scored:
                    head, sep, rest = self.entries[item[0]]["title"].partition(" / ")
                    if not sep or rest.split(" / ")[0] != "基本信息":
                        continue
                    if re.sub(r"[\s·・\-—]+", "", head) == merged_query:
                        profile = item
                        break
                if profile is not None and scored[0][0] != profile[0]:
                    scored.remove(profile)
                    scored.insert(0, profile)
        limit = max(1, top_k)
        if pin_re is not None:
            # 置顶意图：命中置顶特征的片段排最前（如简称查询里登场角色片段
            # BM25 基础分薄，靠加权排不上来），同名片段最多几条防霸榜，
            # 其余按分补足
            title_counts: Dict[str, int] = {}
            ranked: List[tuple[int, float]] = []
            for position, score in scored:
                entry = self.entries[position]
                title = entry["title"]
                if not pin_re.search(title):
                    continue
                if pin_theme is not None and pin_theme not in title:
                    continue
                if len(entry["text"]) < 30:
                    continue  # 纯 meta 片段（如"CV 排序说明"）没有可引用内容
                if title_counts.get(title, 0) >= self.PIN_TITLE_LIMIT:
                    continue
                title_counts[title] = title_counts.get(title, 0) + 1
                ranked.append((position, score))
                if len(ranked) >= limit:
                    break
            if len(ranked) < limit:
                picked = {position for position, _ in ranked}
                for position, score in scored:
                    if position not in picked:
                        ranked.append((position, score))
                        if len(ranked) >= limit:
                            break
        else:
            ranked = scored[:limit]
        results: List[Dict[str, Any]] = []
        for position, score in ranked:
            entry = self.entries[position]
            results.append(
                {
                    "title": entry["title"],
                    "text": _truncate_around(
                        entry["text"], self._window_needle(entry, flat_query, query_terms),
                        max(50, max_chars),
                    ),
                    "score": round(score, 3),
                }
            )
        return results

    def _score_entry(
        self,
        entry: Dict[str, Any],
        query_terms: List[str],
        flat_query: str,
        phrases: List[str],
    ) -> float:
        """对单个片段打分：BM25（正文 + 标题）+ 覆盖加成 + 短语命中加成。"""
        k1, b = self.BM25_K1, self.BM25_B
        score = 0.0
        matched = 0
        for term in query_terms:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            text_freq = entry["text_tf"].get(term, 0)
            title_freq = entry["title_tf"].get(term, 0)
            if not text_freq and not title_freq:
                continue
            matched += 1
            if text_freq:
                score += idf * _bm25_weight(text_freq, entry["len"], self.avg_len, k1, b)
            if title_freq:
                score += (
                    self.TITLE_BOOST
                    * idf
                    * _bm25_weight(
                        title_freq, entry["title_len"], self.avg_title_len, k1, b
                    )
                )
        if not matched:
            return 0.0
        # 多词覆盖加成：命中的词占比越高越接近原分，
        # 避免单个冷门词的高 IDF 压过同时命中多个词的片段
        score *= 0.4 + 0.6 * (matched / len(query_terms))
        # 查询词全部落在小节标题里：基本可以断定就是这个节点
        # （"三月七 时装" → "三月七 / 时装"，而不是正文里顺带提到"时装"的长片段）
        if matched == len(query_terms) and all(
            entry["title_tf"].get(term) for term in query_terms
        ):
            score *= 1.8
        base = max(score, 1.0)
        if len(flat_query) >= 2 and flat_query in entry["lower"]:
            score += base * 0.5
        for phrase in phrases[: self.PHRASE_BONUS_MAX]:
            if phrase != flat_query and phrase in entry["lower"]:
                score += base * self.PHRASE_BONUS
        return score

    @staticmethod
    def _window_needle(entry: Dict[str, Any], flat_query: str, query_terms: List[str]) -> str:
        """选一个用于定位截断窗口的命中串：优先整串查询，其次第一个命中的词。"""
        if len(flat_query) >= 2 and flat_query in entry["lower"]:
            return flat_query
        for term in sorted(query_terms, key=len, reverse=True):
            if term in entry["lower"]:
                return term
        return ""


def _bm25_weight(freq: int, length: int, average: float, k1: float, b: float) -> float:
    """BM25 的词频权重部分（长度归一化）。"""
    return freq * (k1 + 1.0) / (freq + k1 * (1.0 - b + b * length / max(average, 1e-6)))


def _open_index_text(path: Path):
    """按扩展名打开索引文件文本流（.gz 自动解压）。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_index_file(path: Path) -> Dict[str, Any]:
    """读取索引文件（.gz 后缀自动解压，其余按纯 JSON 读）。"""
    with _open_index_text(path) as handle:
        return json.load(handle)


def read_index_theme(path: Path) -> str:
    """轻量读取索引的显示主题名（display_name），不解析整包 chunks。

    元信息排在 chunks 之前，只读文件前缀即可拿到 display_name；
    取不到（缺字段或为空串）时返回空串，由调用方决定是否回退整包解析。
    """
    try:
        with _open_index_text(path) as handle:
            head = handle.read(_THEME_PREFIX_CHARS)
    except (OSError, UnicodeDecodeError, EOFError):
        return ""
    match = _DISPLAY_NAME_RE.search(head)
    if match is None:
        return ""
    try:
        value = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return ""
    return str(value).strip()
