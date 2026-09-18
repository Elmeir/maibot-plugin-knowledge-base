"""检索侧文本处理：jieba 分词、查询清洗、简称近似匹配、围绕命中的开窗截断。

只放运行时检索需要的文本逻辑；构建期的 wiki 转载清洗在 `cleaning.py`。
jieba 在此统一导入并压低日志（首次分词会构建前缀词典、输出进度日志）。
"""

from __future__ import annotations

import logging
import re
import sys
from typing import List

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError as e:  # pragma: no cover
    raise ImportError("知识库插件需要 jieba，请安装：pip install jieba>=0.42.1") from e

# 句子切分（运行时截断与构建期长段切分共用）
_SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；\n]?")

_STOPWORDS = set(
    "的 了 是 在 我 你 他 她 它 我们 你们 他们 咱 它们 这 那 这些 那些 这个 那个 "
    "也 都 就 还 并 与 或 及 和 对 对于 关于 被 把 让 使 从 向 往 于 到 "
    "中 上 下 里 去 来 会 能 可以 要 想 说 有 无 没 不 很 挺 太 更 最 "
    "一个 一些 什么 怎么 如何 为什么 因为 所以 但是 而且 然后 因此 以及 或者 并且 如果 虽然 "
    "等等 之类 等 吗 呢 吧 啊 哦 嗯 呀 嘛 啦 哈哈 呵呵".split()
)

_PUNCT_TOKEN_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# 查询侧额外过滤的词：自然语言问句里的疑问词、语气词与"泛需求词"。
# 它们不代表检索意图，留在查询词集合里只会压低真正关键词的覆盖占比。
_QUERY_NOISE = set(
    "谁 哪些 哪个 哪里 哪儿 多少 几 怎么 怎样 如何 为什么 为何 是否 有没有 吗 呢 吧 啊 呀 "
    "介绍 一下 请问 请 告诉 说说 讲讲 名字 叫 叫做 称为 简称 全称 "
    "相关 有关 关于 内容 资料 信息 意思 含义 情况 方面 东西 地方 时候 记得 知道".split()
)

# 查询里的疑问/语气模式：先按串剥离，否则 jieba 会把"崩铁是什么游戏"切成
# "崩铁是"+"什么游戏"，真正要检索的"崩铁"反而消失
_QUERY_PATTERN_RE = re.compile(
    r"是什么|是啥|有什么|有哪些|有那些|什么是|是谁|在哪|怎么|怎样|如何|为什么|多少|"
    r"叫什么名字|叫什么|叫啥|名字|介绍一下|介绍下|介绍|告诉我|请问|一下|哪些|哪个"
)

# "X 是什么"这类提问：优先返回该条目的概述性小节，而不是正文里顺带提到 X 的段落
_OVERVIEW_QUERY_RE = re.compile(r"是什么|什么是|是啥|是谁|介绍一下|介绍下|讲讲|说说")
_OVERVIEW_SECTIONS = ("简介", "基本信息", "介绍", "概述", "游戏简介", "设定")


def tokenize(text: str, *, for_query: bool = False) -> List[str]:
    """jieba 搜索引擎模式分词，过滤停用词与标点。

    搜索引擎模式会额外切出子词（"三月七" → "三月"、"月七"、"三月七"），
    召回更全；查询侧再滤掉疑问词、语气词这类没有检索意图的词。
    """
    tokens: List[str] = []
    for word in jieba.lcut_for_search(str(text).lower()):
        word = word.strip()
        if not word or word in _STOPWORDS:
            continue
        if for_query and word in _QUERY_NOISE:
            continue
        if _PUNCT_TOKEN_RE.match(word):
            continue
        # intern：同一词在全库共用一份字符串对象，词频表省下大量内存，
        # 后续比较也走指针相等，更快
        tokens.append(sys.intern(word) if len(word) <= 32 else word)
    return tokens


def _wants_overview(query: str) -> bool:
    return bool(_OVERVIEW_QUERY_RE.search(query))


def _clean_query(query: str) -> str:
    """剥离疑问/语气成分，留下真正的检索意图（"崩铁是什么游戏" → "崩铁 游戏"）。"""
    cleaned = re.sub(r"\s+", " ", _QUERY_PATTERN_RE.sub(" ", query)).strip()
    return cleaned or query.strip()


def _chars_near_in_order(term: str, text: str, max_gap: int = 6) -> bool:
    """term 的字符是否按序且相邻字符间隔不大地出现在 text 中。

    用于简称/别称与全称的对应："崩铁""星铁" ↔ "崩坏：星穹铁道"。
    限制间隔是为了避免"三月七"这类短词在长标题里被七零八落地"凑"出来。
    """
    position = text.find(term[0])
    if position < 0:
        return False
    for char in term[1:]:
        found = text.find(char, position + 1)
        if found < 0 or found - position > max_gap:
            return False
        position = found
    return True


def _query_phrases(query: str) -> List[str]:
    """查询里按空格/标点切出的短语（"三月七 时装" → ["三月七", "时装"]）。

    用于原文整串命中加成：短语越具体，命中越说明相关；过长的整串不可能是原文子串，跳过。
    """
    phrases: List[str] = []
    for piece in re.split(r"[\s，。、,？?！!：:；;]+", query):
        piece = piece.strip().lower()
        if 2 <= len(piece) <= 8 and piece not in _QUERY_NOISE:
            phrases.append(piece)
    return list(dict.fromkeys(phrases))


def _truncate_at_sentence(text: str, limit: int) -> str:
    """把文本截断到 limit 内，优先在句子边界收尾，避免半句话喂给模型。"""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # 换行优先：行边界是最强的结构边界（目录/表格一行一个实体，绝不切半行）
    for sep in ("\n", "。", "！", "？", "；"):
        position = cut.rfind(sep)
        if position >= limit * 0.5:
            return cut[: position + 1]
    return cut


def _truncate_around(text: str, needle: str, limit: int) -> str:
    """截断片段时优先保证命中词可见：围绕命中位置开窗，而不是从开头截。

    多行结构片段（角色目录、数据表等一行一个实体）按行边界对齐窗口——
    单行要么完整给出要么整行去掉，不喂半行；普通段落维持字符级开窗。
    """
    if len(text) <= limit:
        return _truncate_at_sentence(text, limit)
    position = text.lower().find(needle.lower()) if needle else -1
    if position < 0:
        return _truncate_at_sentence(text, limit)
    start = max(0, position - int(limit * 0.35))
    end = min(len(text), start + limit)
    if "\n" in text:
        line_start = text.find("\n", start) + 1 if start > 0 else 0
        line_end = text.rfind("\n", start, end)
        adjusted_start, adjusted_end = line_start, line_end if line_end > start else end
        # 行对齐把窗口挤得太小（如整段只有一两个换行）就退回字符级
        if adjusted_end - adjusted_start >= min(limit, 80):
            start, end = adjusted_start, adjusted_end
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


# infobox 里的链接清理：值多为 URL / 裸域名；展示用（不进 FTS），宁可保守——
# 命中链接形态才删，正常属性（身高：153cm、血型：O型、唱片公司：King Records、
# 原作：5pb.）一律保留。构建端与运行端共用此口径。
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_HOSTNAME_RE = re.compile(
    r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?:com|net|org|jp|cn|tw|hk|kr|us|uk|de|fr|io|co|me|tv|cc|xyz|info|biz|club"
    r"|live|app|dev|gg|im|so|sh|ly|wiki|fm|to|is)\b(?:/\S*)?",
    re.IGNORECASE,
)
# 社交 / 主页 / 来源类键：值恒为链接或 @ 号，无检索价值，整段丢弃（别名等正常键不受影响）
_LINK_KEY_RE = re.compile(
    r"官方网站?|官方?主页|首页|主页|引用来源|来源|外部链接|参考资料|链接"
    r"|twitter|推特|微博|weibo|facebook|ins|instagram|pixiv|p站|youtube|youtu"
    r"|bilibili|b站|nico(?:nico)?|blog|博客|个人博客|hp|homepage|web"
    r"|fanclub|粉丝俱乐部|事务所资料页|imdb|dmmid|twitch|discord|tiktok|抖音|快手",
    re.IGNORECASE,
)
# 键名里嵌了社交/主页 token 的变体写法（如"X（Twitter）""Twitter(临时)""pixiv id"）：
# 只用多字符 token 做子串匹配，避免误伤"爱好：Instagram"这类把平台名写进【值】的正常键
_LINK_KEY_SUBSTR_RE = re.compile(
    r"twitter|推特|weibo|微博|facebook|instagram|pixiv|p站|youtube|bilibili|b站"
    r"|fanclub|粉丝俱乐部|博客|blog|链接|主页|首页|官网|官方网站|事务所资料页"
    r"|imdb|tiktok|抖音|discord|twitch|快手",
    re.IGNORECASE,
)
# 被 " / " 截断只剩协议头的残值（原始 URL 里的 / 撞上了拍平分隔符）
_SCHEME_ONLY_RE = re.compile(r"^(?:https?|ftp|www)[:/]?$", re.IGNORECASE)


def _is_link_key(key: str) -> bool:
    """键本身是社交/主页/来源类（精确命中，或含明确社交 token 的变体写法）。"""
    return bool(_LINK_KEY_RE.fullmatch(key)) or bool(_LINK_KEY_SUBSTR_RE.search(key))


def strip_links(value: str) -> str:
    """从"键：值 / 键：值"串里删掉 URL、裸域名、社交/主页键，并清理随之变空的段。"""
    s = str(value or "")
    if not s:
        return ""
    s = _URL_RE.sub("", s)
    s = _HOSTNAME_RE.sub("", s)
    # 按分隔符切段：丢空段、只剩协议头的残值、社交/主页键、以及值被清空的"键："段
    kept = []
    for seg in s.split(" / "):
        seg = seg.strip().strip("/").strip()
        if not seg:
            continue
        match = re.match(r"^([^：]{1,16})：(.*)$", seg)
        if match:
            key, val = match.group(1).strip(), match.group(2).strip()
            if _is_link_key(key) or _SCHEME_ONLY_RE.match(val):
                continue
            if not val:
                continue
        kept.append(seg)
    return " / ".join(kept)
