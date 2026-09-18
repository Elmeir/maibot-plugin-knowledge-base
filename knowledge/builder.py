"""离线构建入口：把一个或多个 txt 转换为随插件分发的内置索引（kb_data/*.index.json）。

组合"清洗/解析/分块（`cleaning` + `parsing`）→ BM25 索引（`index`）"整条管线。
构建结果随插件仓库分发，宿主启动时直接加载，运行时不依赖原始 txt。
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .cleaning import _finalize_chunks, _read_text_file
from .index import KnowledgeIndex
from .parsing import chunk_plain_text, chunk_sections, parse_sections


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """源文件指纹：文件名 + 字节数 + 内容 sha256，写入索引 sources 供追溯。"""
    data = path.read_bytes()
    return {
        "name": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


# 登场角色行尾的"属性+特性/命途"后缀（"猫宫又奈物理强攻""星见雅烈霜异常"），
# 剥掉后剩下的才是角色名
_PLAYABLE_SUFFIX_RE = re.compile(
    r"(?:物理|火|冰|雷|风|量子|虚数|以太|烈霜|霜烈|电|玄炎)?"
    r"(?:毁灭|巡猎|智识|同谐|虚无|存护|记忆|欢愉|强攻|击破|异常|防护|支援)$"
)


def _extract_playable_names(text: str) -> set[str]:
    """从主词条【登场角色】节（原始 txt 按行扫描）提取可控角色名（名单白名单）。

    数据驱动，三种行形态：
    - 崩铁："姬子 火智识"——名字后跟属性命途（NPC/星神没有，天然过滤）
    - 绝区零 Random Play 组："哲"独立行、下一行是"CV：…"
    - 绝区零 代理人组："猫宫又奈物理强攻 / CV：…"——本名+属性特性后缀与 CV 同行
    虚狩、钥匙等阵营不在主词条登场角色节里，天然被排除。
    """
    names: set[str] = set()
    in_roster = False
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("【"):
            in_roster = s == "【登场角色】"
            continue
        if not in_roster or not s or s.startswith(("〔", "CV")):
            continue
        # 崩铁形态："姬子 火智识"（名字与属性以空格分隔）
        head = re.match(r"^(\S{1,12}) (?:物理|火|冰|雷|风|量子|虚数)", s)
        if head:
            names.add(head.group(1))
            continue
        # 绝区零 Random Play 形态："哲"独立行紧跟 CV 行
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt.startswith("CV："):
            names.add(_PLAYABLE_SUFFIX_RE.sub("", s).strip())
            continue
        # 绝区零 代理人形态："猫宫又奈物理强攻 / CV：…"（本名与 CV 同行）
        m = re.match(
            r"^(\S{1,20}?)(?:物理|火|冰|雷|风|量子|虚数|以太|烈霜|霜烈|电|玄炎)?"
            r"(?:强攻|击破|异常|防护|支援) / CV：",
            s,
        )
        if m:
            names.add(html.unescape(m.group(1)))
    return names


def _extract_character_directory(
    sections: List[Any],
) -> tuple[str, str] | None:
    """从单篇角色 txt 的【基本信息】提取 (角色名, "萌点…（发色·瞳色）")。

    非角色页（无萌点行）返回 None。sections 为 parse_sections 的输出，
    基本信息节内的信息框行是"键：值"折叠格式。
    """
    name = ""
    moe = hair = eye = ""
    for title, paragraphs in sections:
        if title.split(" / ")[-1] != "基本信息":
            continue
        if not name:
            name = title.split(" / ")[0]
        for para in paragraphs:
            for line in para.split("\n"):
                if line.startswith("萌点："):
                    moe = line[len("萌点：") :].strip()
                elif line.startswith("发色："):
                    hair = line[len("发色：") :].strip()
                elif line.startswith("瞳色："):
                    eye = line[len("瞳色：") :].strip()
    if not (name and moe):
        return None
    look = f"（{hair}·{eye}）" if hair or eye else ""
    return name, f"{name}：{moe}{look}"


_LASTMOD_RE = re.compile(r'"lastmod":"\s*此页面最后编辑于(\d{4})年(\d{1,2})月(\d{1,2})日')


def _page_lastmod(html_path: Path) -> str:
    """从页面缓存的 HTML 提取最后编辑日期（"2026年9月13日" → "20260913"）。

    萌百渲染页把 lastmod 写在内嵌脚本里；解析失败或无缓存返回空串。
    """
    try:
        html = html_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = _LASTMOD_RE.search(html)
    if not m:
        return ""
    return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"


def build_builtin_index(
    txt_path: Path,
    out_path: Path,
    chunk_size: int = 600,
    display_name: str = "",
) -> Dict[str, Any]:
    """单文件构建（委托多文件版本）。"""
    return build_builtin_index_from_files(
        [txt_path], out_path, chunk_size=chunk_size, display_name=display_name
    )


def build_builtin_index_from_files(
    txt_paths: List[Path],
    out_path: Path,
    chunk_size: int = 600,
    display_name: str = "",
    html_dir: Path | None = None,
) -> Dict[str, Any]:
    """开发期工具：把一个或多个 txt 合并构建为随插件分发的内置索引文件。

    内置数据随插件仓库分发，宿主启动时直接加载，运行时不依赖原始 txt。
    display_name 为知识库显示主题（如游戏名），写入索引供 Tool 描述使用。
    html_dir 提供同名 html 缓存时，角色目录按页面最后编辑时间倒序（新角色在前）。
    """
    all_chunks: List[Dict[str, str]] = []
    sources: List[Dict[str, Any]] = []
    directory: List[tuple[str, str, str]] = []  # (角色名, "萌点…（发色·瞳色）", lastmod)
    seen_names: set[str] = set()
    playable_names: set[str] = set()
    main_title = str(display_name or "").strip()
    for txt_path in txt_paths:
        text = _read_text_file(txt_path)
        sections = parse_sections(text)
        if sections:
            all_chunks.extend(chunk_sections(sections, chunk_size=chunk_size))
            # 主词条的【登场角色】节列出的是可控/正式登场角色（hsr 行带属性命途、
            # zzz 名字行紧跟 CV 行），作为名单白名单，滤掉虚狩、NPC 之类的杂项
            if main_title and text.split("\n", 1)[0].strip() == main_title:
                playable_names |= _extract_playable_names(text)
            # 角色页（基本信息含萌点行）→ 收进目录，供主观对比类查询一次取全
            entry = _extract_character_directory(sections)
            if entry and entry[0] not in seen_names:
                seen_names.add(entry[0])
                lastmod = (
                    _page_lastmod(html_dir / f"{txt_path.stem}.html")
                    if html_dir
                    else ""
                )
                directory.append((html.unescape(entry[0]), entry[1], lastmod))
        else:
            all_chunks.extend(chunk_plain_text(text, chunk_size=chunk_size))
        sources.append({**_file_fingerprint(txt_path), "builtin": True})

    # 合成目录片段：主观对比类问题（"哪个角色可爱/你喜欢谁"）一次查询
    # 即可拿到全部角色的萌点与外观标签
    if len(directory) >= 2:
        topic_label = str(display_name or "").strip() or "本作"
        # 有主词条白名单时（可控/正式登场角色），名单与萌点目录同步滤掉
        # 钥匙、虚狩、NPC 等杂项
        if playable_names:
            directory = [
                (name, desc, lastmod)
                for name, desc, lastmod in directory
                if any(
                    name == w or name in w or w in name for w in playable_names
                )
            ]
        # 按页面最后编辑时间倒序：新实装/新编辑的角色排最前（无时间的垫底）
        directory.sort(key=lambda item: item[2], reverse=True)
        roster_names = [name for name, _, _ in directory]
        # 萌点按主题内出现次数排序、每角色只留前十：高频标签（玩家公认的通用
        # 印象，如"天然萌""巨乳"）排前面，长尾描述（注音口癖、一次性设定）被裁掉
        tag_counter: Counter[str] = Counter()
        parsed_dir: list[tuple[str, list[str], str]] = []
        for name, desc, _ in directory:
            _, _, tail = desc.partition("：")
            look_m = re.search(r"（[^（）]*·[^（）]*）$", tail)
            look = look_m.group(0) if look_m else ""
            body = tail[: look_m.start()] if look_m else tail
            tags = [t for t in (part.strip() for part in body.split("、")) if t]
            tag_counter.update(tags)
            parsed_dir.append((name, tags, look))
        moe_lines = [
            f"{name}：{'、'.join(sorted(tags, key=lambda t: -tag_counter[t])[:10])}{look}"
            for name, tags, look in parsed_dir
        ]
        all_chunks.append(
            {
                "title": f"{topic_label} 角色萌点一览",
                "text": (
                    f"以下为{topic_label}最近登场角色的萌点与外观标签（最新10位角色，萌点按常见度排序取前10），可用于对比角色形象：\n"
                    + "\n".join(moe_lines[:10])
                ),
            }
        )
        # 合成"登场角色一览"纯名单片段：名字比萌点行短得多，max_chars 默认值
        # 也能完整返回，"XX 有哪些角色"的列表查询靠它看到全量角色
        all_chunks.append(
            {
                "title": f"{topic_label} 登场角色一览",
                "text": (
                    f"以下为{topic_label}全部登场角色名单（共{len(roster_names)}位）：\n"
                    + "、".join(roster_names)
                ),
            }
        )

    all_chunks = _finalize_chunks(all_chunks)
    index = KnowledgeIndex()
    index.build(all_chunks, sources)
    if index.size == 0:
        raise ValueError("知识库内容为空，未能构建任何索引片段")
    index.display_name = str(display_name or "").strip()
    index.save(out_path)
    return {"files": [path.name for path in txt_paths], "chunks": index.size}
