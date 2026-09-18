"""构建期文本清洗：把 wiki 转载残留规整成干净的知识文本。

只在离线构建（txt → 索引）时用，运行时不导入。集中放这里，与运行时检索
（`index.py` / `text.py`）解耦——检索侧不需要这些抓取产物清洗规则。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

# 装饰性制表字符（制表框线、色块等），整行仅由这些字符组成时视为分隔线
_DECORATION_CHARS = set(
    " \u3000═━─–—―＝=﹏＿~＊*#·•●○■□▪▫▬▭▮▯═║╔╗╚╝▓░▸►▶▷◆◇★☆✦✧"
    "┃│┌┐└┘├┤┬┴┼╭╮╯╰▁▂▃▄▅▆▇█|｜+＋"
)
# 【括号标题】一级小节（抓取脚本把页面 h2 与 wiki 小节标题转成这种形式）
_BRACKET_HEADER_RE = re.compile(r"^【([^【】]{1,30})】$")
# 〔括号子标题〕二级小节（h3 及更深的层级）
_SUB_HEADER_RE = re.compile(r"^〔([^〔〕]{1,30})〕$")
# 〖括号三级标题〗（正文 <dt> 术语，fetch 渲染产出）
_TERM_HEADER_RE = re.compile(r"^〖([^〖〗]{1,30})〗$")

# 一行内粘连的多个标题标记（【】/〔〕/〖〗混排），拆开各占一行
_HEADER_TOKEN_RE = r"[【〔〖][^【】〔〖〗]{1,30}[】〕〗]"
_MERGED_HEADERS_RE = re.compile(rf"({_HEADER_TOKEN_RE})(?={_HEADER_TOKEN_RE})")

# ──── 内容清洗（wiki 转载残留） ────

# 站内交叉引用："（参考：崩坏：星穹铁道/光锥）"
_REF_LINK_RE = re.compile(r"（参考：[^）]*）")
# 表格分隔符连跑：" / / / / "（空单元格连续出现）
_SLASH_RUN_RE = re.compile(r"(?:\s*/\s*){2,}")
# 空单元格渲染残留在行首/行尾：" / 生命值 635"、"来历 /"
_EDGE_SLASH_RE = re.compile(r"^\s*(?:/\s*)+|(?:\s*/)+\s*$")
# 表格单元格渲染产生的重复字："毁灭毁灭 巡猎巡猎"（仅当行内出现 ≥2 组时才折叠，
# 避免误伤"考虑考虑"类正常叠词）
_DOUBLING_RE = re.compile(r"([\u4e00-\u9fff]{2})\1")
# 连续空格
_SPACE_RUN_RE = re.compile(r"[ \u3000]{2,}")
# wiki 图注行："游戏图标，图标中角色为三月七"
_CAPTION_RE = re.compile(r"^(游戏|项目|作品)?图标[，,、]")
# wiki 指引行（"主条目：丰饶玄鹿"、"见上文。"）：只起跳转作用，无知识内容。
# 只匹配整行就是指引的，带正文的（"技能与原版一致，见上文。"）照旧保留
_NAV_HINT_RE = re.compile(r"^(?:主条目|另见|参见)[：:]|^.{0,4}见上文[。.]?$")
# wiki 表格徽章残留："【适龄提示12+ |】"（【】内含竖线分隔的徽章文本）
_BADGE_RE = re.compile(r"【[^】|]*\|[^】]*】")
# wiki 列表标记残留："配音导演：* 彭博"、"：* 通过开拓等级奖励"
_COLON_STAR_RE = re.compile(r"[：:]\s*\*+\s*")
# 重复标点折叠：仅收拢几乎必为渲染残留的句读（"。。"→"。"、"，，"→"，"）。
# 感叹号/问号不参与（"！！！""？？？"常是合法强调，如三月七别号"？？？"）；"……"省略号本就不匹配。
_PUNCT_RUN_RE = re.compile(r"([。，、；])\1+")
# 信息框内联 CSS 残留（heimu 黑幕样式规则被当正文吞入），可整行或嵌在行中间
_CSS_RESIDUE_RE = re.compile(r"\.[\w-]+(?:\s+[\w.:>\[\]\"'-]+)*\s*\{[^{}]*\}")

# 独立引号行（wiki 引用框模板渲染出的装饰符号），整行删除
_QUOTE_ONLY_RE = re.compile(r"^[\"“”‘’']+$")
# 脚注标记："[1]"、"[12]"
_FOOTNOTE_RE = re.compile(r"\[\d+\]")
# 视频/折叠控件的可见文案（播放器与折叠条的外壳文字，无知识内容）
# "来历"是旧管线留下的折叠按钮标签（新解析已按 mw-customtoggle 跳过）
_UI_BOILERPLATE_RE = re.compile(r"^\s*(?:展开|折叠|待补充|（待补充）|暂无|来历)\s*$")
# 播放器外壳短行（旧管线把"宽屏模式"与"显示视频"并成一行）：短行内出现即丢弃
_UI_CONTROL_RE = re.compile(r"宽屏模式|显示视频|点击展开|点击折叠")
# 未渲染出来的模板参数："{{UserName}}"
_TEMPLATE_RESIDUE_RE = re.compile(r"\{\{[^{}]{0,40}\}\}")
# 不可见字符：零宽字符（不间断空格另按空格处理）
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

# 萌百信息框（infobox3）键名：仅在【基本信息】节内与后续值行合并为"键：值"
_INFOBOX_KEYS = {
    "本名", "外文名", "别名", "别号", "昵称", "称号", "头衔",
    "原文名", "罗马字", "片假名",
    "性别", "年龄", "种族", "族群", "种类", "物种",
    "发色", "瞳色", "外貌", "特征",
    "生日", "忌日", "出身地", "出身地区", "活动范围", "常驻位置", "所属",
    "所属团体", "所属组织", "身份", "职业", "阵营", "信仰",
    "属性", "命途", "弱点", "数值",
    "配音", "声优", "演员", "角色设计", "作画", "音乐",
    "萌点", "个人状态", "相关人士", "登场作品", "登场", "适龄提示",
    "短信签名", "座右铭", "口头禅",
    "全称", "别称", "简称", "成立时间", "总部",
    "组织名称", "组织种类", "组织领导人", "代表角色",
    "种族名称", "歌曲名", "作曲", "编曲", "作词", "演唱", "收录专辑", "出品",
    "制作人", "发行", "发行商", "发行日期", "发行时间",
    "原名", "官方译名", "常用译名", "类型", "平台", "开发", "开发商",
    "引擎", "模式", "系列", "分级", "题材", "玩家人数", "相关作品",
    "官网", "画面", "主角", "监督", "编剧", "历法", "货币", "族群",
}
# 信息框内的分组表头行（非键名），直接丢弃
_INFOBOX_DROP_LINES = {"基本资料", "基本信息", "资料"}
# 注释/外链类样板节（脚注说明、参考链接列表、同名条目消歧义导航），整节无知识价值
_BOILERPLATE_SECTION_RE = re.compile(r"注释|外部链接|参考资料|参考链接|相关消歧义页")


def _is_header_line(line: str) -> bool:
    """整行是否为小节标题（一级【】、二级〔〕或三级〖〗）。"""
    return bool(
        _BRACKET_HEADER_RE.match(line)
        or _SUB_HEADER_RE.match(line)
        or _TERM_HEADER_RE.match(line)
    )


def _is_decoration_line(line: str) -> bool:
    """判断整行是否为纯装饰分隔线。"""
    return bool(line) and all(ch in _DECORATION_CHARS for ch in line)


def _is_chapter_header(line: str) -> bool:
    """判断是否为章标题行（"第X章"或抓取 txt 首行的词条标题）。"""
    return bool(line) and len(line) <= 60 and not re.search(r"[。！？；，,]", line)


def _collapse_line_doubling(line: str) -> str:
    """整行就是一个叠词（筛选器标签把"毁灭"渲染两遍）时收拢一半。"""
    doubled = re.fullmatch(r"(.{2,8})\1", line)
    return doubled.group(1) if doubled else line


def _collapse_doubled(value: str) -> str:
    """值整体为叠词（模板徽章渲染产物，如"欢愉欢愉"）时收拢一半。"""
    doubled = re.fullmatch(r"(.{2,8})\1", value.replace(" ", ""))
    return doubled.group(1) if doubled else value


def _fold_infobox_value(key: str, values: List[str]) -> str:
    """信息框键值折叠："键" + 值行列表 → "键：值"。

    仅作为旧版 txt（键名/值逐行平铺）的兜底；新抓取管线在抓取阶段就按
    信息卡 DOM 结构输出"键：值"行，并列项以顿号连接。
    """
    value = _collapse_doubled(" ".join(values).strip())
    return f"{key}：{value}" if value else key


def _clean_infobox_value(value: str) -> str:
    """已折叠"键：值"行的值清洗：去徽章残留前导横线、收拢叠词。"""
    value = re.sub(r"^[-－—]\s*", "", value.strip())
    return _collapse_doubled(value)


def _has_readable_content(text: str) -> bool:
    """判断文本是否含实际可读内容（≥2 个中日韩文字或 ≥3 个拉丁字母）。"""
    readable = sum(
        1 for ch in text
        if "\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"  # 假名/汉字/谚文
    )
    if readable >= 2:
        return True
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return latin >= 3


def _clean_body_line(line: str) -> str:
    """清洗单行正文：剔除 wiki 转载残留噪音，规整空白。"""
    # 信息框内联 CSS 残留（如 ".mw-parser-output .infobox3 span.heimu{...}"），
    # 可能独占一行，也可能夹在别号列表中间，一律剔除并收拾残留顿号
    if "{" in line and "}" in line and _CSS_RESIDUE_RE.search(line):
        line = _CSS_RESIDUE_RE.sub("", line)
        line = re.sub(r"、{2,}", "、", line).strip("、 ")
        if not line:
            return ""
    # 消歧义导航句（"……详见「xxx」"）、wiki 指引行（"主条目：xxx"）与图注行，无知识价值
    if "详见「" in line or _NAV_HINT_RE.search(line) or _CAPTION_RE.match(line):
        return ""
    line = _REF_LINK_RE.sub("", line)
    line = _BADGE_RE.sub("", line)
    line = _COLON_STAR_RE.sub("：", line)
    # 行首 wiki 列表标记："* 体力：「开拓力」"
    if line.startswith("*"):
        line = line.lstrip("*").strip()
    # 空名字段："▸ -：登上星穹列车" → "▸ 登上星穹列车"
    line = re.sub(r"([-▸▪◆●]\s*)[-－]\s*([：:])", r"\1", line)
    line = _SLASH_RUN_RE.sub(" / ", line)
    line = _EDGE_SLASH_RE.sub("", line)
    line = _collapse_line_doubling(line)
    if len(_DOUBLING_RE.findall(line)) >= 2:
        previous = None
        while previous != line:
            previous = line
            line = _DOUBLING_RE.sub(r"\1", line)
    line = _PUNCT_RUN_RE.sub(r"\1", line)
    return _SPACE_RUN_RE.sub(" ", line).strip()


def _is_low_value_chunk(text: str, title: str = "") -> bool:
    """判断片段是否为低价值孤行（标题残留、标语等，无知识密度）。

    小节归属明确的短内容要留（如"三月七 / 时装 → 冬去煦至（冰·存护）"，
    它就是这个时装小节的唯一信息），没有章节路径的孤行才按短文本丢弃。
    """
    stripped = text.strip()
    if "\n" in stripped:
        return False
    minimum = 8 if " / " in title else 12
    if len(stripped) < minimum:
        return True
    # 无 CJK、无句读的短行（如英文文档标题残留）
    cjk = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    return cjk == 0 and len(stripped) < 60 and not re.search(r"[。！？；]", stripped)


def _finalize_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """分块后处理：过滤无可读内容的片段并按归一化文本去重。"""
    seen = set()
    result: List[Dict[str, str]] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        title = str(chunk.get("title", ""))
        if not _has_readable_content(text) or _is_low_value_chunk(text, title):
            continue
        key = re.sub(r"\s+", "", text)
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def normalize_article_text(text: str) -> str:
    r"""规范化抓取词条 txt 的版式（不改动正文内容）：

    - 拆开同一行粘连的多个【标题】（旧抓取管线残留）；
    - 【基本信息】节的"键名行+值行"折叠为"键：值"（别号、种族等）；
    - 删除"基本资料"分组表头、独立引号装饰行、脚注标记 [1]；
    - 丢弃注释/外部链接/参考资料类样板节；
    - 合并相邻重复行（信息卡模板会把导语渲染两遍）。
    """
    text = _MERGED_HEADERS_RE.sub(r"\1\n", text)
    lines: List[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        stripped = _INVISIBLE_RE.sub("", stripped).replace("\xa0", " ")
        stripped = _FOOTNOTE_RE.sub("", stripped)
        stripped = _TEMPLATE_RESIDUE_RE.sub("", stripped)
        stripped = _SLASH_RUN_RE.sub(" / ", _EDGE_SLASH_RE.sub("", stripped)).strip()
        if not stripped or _QUOTE_ONLY_RE.match(stripped):
            continue
        if _UI_BOILERPLATE_RE.match(stripped) or _is_decoration_line(stripped):
            continue
        if len(stripped) <= 20 and _UI_CONTROL_RE.search(stripped):
            continue
        if _NAV_HINT_RE.search(stripped):
            continue
        stripped = _collapse_line_doubling(stripped)
        if not stripped:
            continue
        # 已折叠成"键：值"的信息框行：去徽章残留横线、收拢叠词值
        folded = re.match(r"^([^：]{1,8})：(.+)$", stripped)
        if folded and folded.group(1) in _INFOBOX_KEYS:
            stripped = f"{folded.group(1)}：{_clean_infobox_value(folded.group(2))}"
        lines.append(stripped)

    out: List[str] = []
    in_infobox = False
    in_boilerplate = False
    pending_key = ""
    pending_values: List[str] = []

    def _flush() -> None:
        nonlocal pending_key, pending_values
        if pending_key:
            out.append(_fold_infobox_value(pending_key, pending_values))
            pending_key = ""
            pending_values = []

    for line in lines:
        if not line:
            # 信息框内部折叠后是密集键值块，不需要空行分隔
            if not in_infobox and out and out[-1] != "":
                out.append("")
            continue
        heading = _BRACKET_HEADER_RE.match(line)
        if heading is not None:
            _flush()
            section_name = heading.group(1).strip()
            in_infobox = section_name == "基本信息"
            in_boilerplate = bool(_BOILERPLATE_SECTION_RE.search(section_name))
            if not in_boilerplate:
                out.append(line)
            continue
        sub_heading = _SUB_HEADER_RE.match(line)
        if sub_heading is not None:
            _flush()
            in_boilerplate = in_boilerplate or bool(
                _BOILERPLATE_SECTION_RE.search(sub_heading.group(1))
            )
            if not in_boilerplate:
                out.append(line)
            continue
        if in_boilerplate:
            continue
        if in_infobox:
            if line in _INFOBOX_DROP_LINES:
                continue
            if line in _INFOBOX_KEYS:
                _flush()
                pending_key = line
                continue
            if pending_key and "。" not in line:
                pending_values.append(line)
                continue
        _flush()
        out.append(line)
    _flush()
    out = _drop_empty_sections(out)

    # 合并相邻完全重复的内容行
    deduped: List[str] = []
    last_content = ""
    for line in out:
        if line and line == last_content:
            continue
        if line:
            last_content = line
        deduped.append(line)
    return "\n".join(deduped).rstrip() + "\n"


def _header_level(line: str) -> int | None:
    """标题行层级：【】=1、〔〕=2、〖〗=3；非标题行返回 None。"""
    if _BRACKET_HEADER_RE.match(line):
        return 1
    if _SUB_HEADER_RE.match(line):
        return 2
    if _TERM_HEADER_RE.match(line):
        return 3
    return None


def _drop_empty_sections(lines: List[str]) -> List[str]:
    """丢弃后面没有任何内容的标题行（【一级】/〔二级〕/〖三级〗）。

    标题后紧跟更深一级子标题（【A】〖B〗内容）不算空；只有到下一个同级或
    更浅标题、或结尾都没有内容时才丢（纯图片表格删空后只剩标题的场景）。
    """
    kept = list(lines)
    for index in range(len(kept) - 1, -1, -1):
        level = _header_level(kept[index])
        if level is None:
            continue
        empty = True
        for candidate in kept[index + 1 :]:
            depth = _header_level(candidate)
            if depth is not None:
                if depth <= level:
                    break  # 到同级/更浅标题仍无内容：空小节
                continue  # 更深的子标题：本节有子结构，继续找内容
            if candidate:
                empty = False
                break
        if empty:
            del kept[index]
    return kept


def _read_text_file(path: Path) -> str:
    """读取 txt 文本，兼容 UTF-8（含 BOM）与 GBK 系编码。"""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
