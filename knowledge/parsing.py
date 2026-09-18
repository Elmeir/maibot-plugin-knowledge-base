"""构建期结构解析与分块：把清洗后的 txt 拆成 [(标题路径, 段落)] 再打包为片段。

抓取脚本产出的 txt 版式固定（首行标题、一级【】、二级〔〕），据此切分小节；
识别不到结构时走滑动窗口兜底。只在离线构建时用，运行时不导入。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .cleaning import (
    _BOILERPLATE_SECTION_RE,
    _BRACKET_HEADER_RE,
    _EDGE_SLASH_RE,
    _INFOBOX_DROP_LINES,
    _INFOBOX_KEYS,
    _MERGED_HEADERS_RE,
    _SUB_HEADER_RE,
    _TERM_HEADER_RE,
    _clean_body_line,
    _fold_infobox_value,
    _has_readable_content,
    _is_chapter_header,
    _is_decoration_line,
)
from .text import _SENTENCE_RE


def parse_sections(text: str) -> List[Tuple[str, List[str]]]:
    """把整篇文本解析为 [(标题路径, 段落列表)]。

    抓取脚本产出的 txt 结构固定：首行为词条标题，一级小节以【标题】标记、
    二级小节以〔标题〕标记、三级小节（正文 <dt> 术语）以〖标题〗标记，
    因此标题路径形如 ``404ERROR / 常驻角色/NPC / 店主``（最多四段）。
    识别不到任何小节时返回空列表，由调用方走无结构兜底分块。

    【基本信息】节里信息框是"键名行 + 若干值行"的扁平结构，这里折叠成
    ``键：值``，避免检索结果出现大量只有键名的断行。
    """
    # 旧抓取管线会把相邻的空小节标题挤在同一行（如"【简介】【角色经历】"），
    # 拆回每行一个，保证小节边界不丢
    text = _MERGED_HEADERS_RE.sub(r"\1\n", text)
    sections: List[Tuple[str, List[str]]] = []
    chapter = ""
    section = ""
    sub = ""
    sub2 = ""
    paragraphs: List[str] = []
    buffer: List[str] = []
    pending_key = ""  # 信息框键名，等待收集其值行
    pending_values: List[str] = []
    first_content_line = True

    def _flush_paragraph() -> None:
        nonlocal pending_key, pending_values
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                paragraphs.append(joined)
            buffer.clear()
        if pending_key:
            paragraphs.append(_fold_infobox_value(pending_key, pending_values))
            pending_key = ""
            pending_values = []

    def _flush_section() -> None:
        _flush_paragraph()
        if paragraphs:
            title = " / ".join(
                part for part in (chapter, section, sub, sub2) if part
            ) or "前言"
            total = sum(len(p) for p in paragraphs)
            if sub2 and total < 60 and sections:
                # 微型 dt 节：单独成块会稀释 BM25 长度统计（avg_len 被拉低），
                # 折进上一节（"术语：内容"），信息不丢
                sections[-1][1].extend(
                    [f"{sub2}：{paragraphs[0]}"] + paragraphs[1:]
                )
            else:
                sections.append([title, list(paragraphs)])
        elif sub2:
            # dt 标签下没有任何内容（下一个标题紧跟而来）：标题不单独保留
            pass
        paragraphs.clear()

    in_infobox = False  # 当前是否处于【基本信息】节
    in_boilerplate = False  # 当前是否处于注释/外链类样板节（整节丢弃）

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            # 信息框值可能跨行，空行只结束普通段落、不打断键值收集
            if not pending_key:
                _flush_paragraph()
            continue
        if _is_decoration_line(line):
            _flush_paragraph()
            continue

        # 首个有内容的行：词条标题（抓取脚本写入），作为全篇章名
        if first_content_line:
            first_content_line = False
            if _is_chapter_header(line):
                chapter = line
                continue

        # 【括号标题】一级小节
        bracket_match = _BRACKET_HEADER_RE.match(line)
        if bracket_match is not None:
            _flush_section()
            section = bracket_match.group(1).strip()
            sub = ""
            sub2 = ""
            in_infobox = section == "基本信息"
            in_boilerplate = bool(_BOILERPLATE_SECTION_RE.search(section))
            continue
        # 〔括号子标题〕二级小节
        sub_match = _SUB_HEADER_RE.match(line)
        if sub_match is not None:
            _flush_section()
            sub = sub_match.group(1).strip()
            sub2 = ""
            in_boilerplate = in_boilerplate or bool(
                _BOILERPLATE_SECTION_RE.search(sub)
            )
            continue
        # 〖括号三级标题〗（正文 <dt> 术语）
        term_match = _TERM_HEADER_RE.match(line)
        if term_match is not None:
            _flush_section()
            sub2 = term_match.group(1).strip()
            in_boilerplate = in_boilerplate or bool(
                _BOILERPLATE_SECTION_RE.search(sub2)
            )
            continue
        if in_boilerplate:
            continue

        # 内容清洗：剔除 wiki 转载残留噪音
        cleaned = _clean_body_line(line)
        if not cleaned:
            continue

        if in_infobox:
            if cleaned in _INFOBOX_DROP_LINES:
                _flush_paragraph()
                continue
            if cleaned in _INFOBOX_KEYS:
                _flush_paragraph()
                pending_key = cleaned
                continue
            if pending_key:
                # 值行不会以句号结尾；遇到完整句子说明信息框已结束（页面导语）。
                # 信息框值允许单字（如"性别：女"），不做可读性过滤
                if "。" in cleaned:
                    _flush_paragraph()
                else:
                    pending_values.append(cleaned)
                    continue

        if not _has_readable_content(cleaned):
            continue

        buffer.append(cleaned)

    _flush_section()
    return sections


def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """把超长段落按句子边界切成若干不超过 chunk_size 的片段。"""
    sentences = _SENTENCE_RE.findall(paragraph)
    pieces: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for sentence in sentences:
        # 单句超长时硬切
        while len(sentence) > chunk_size:
            if buffer:
                pieces.append("".join(buffer))
                buffer, buffer_len = [], 0
            pieces.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if sentence and buffer_len + len(sentence) > chunk_size and buffer:
            pieces.append("".join(buffer))
            buffer, buffer_len = [], 0
        if sentence:
            buffer.append(sentence)
            buffer_len += len(sentence)
    if buffer:
        pieces.append("".join(buffer))
    # 表格行被并成超长行后按句切分，切口处会留下行内单元格分隔符（" / 开拓任务…"）
    return [_EDGE_SLASH_RE.sub("", piece).strip() for piece in pieces if piece.strip()]


def chunk_sections(
    sections: List[Tuple[str, List[str]]], chunk_size: int = 600
) -> List[Dict[str, str]]:
    """按段落边界把各小节贪心打包为 chunk_size 左右的知识片段。

    每个小节末尾过短的碎片（< 60 字符，多为章节边界截断出的半句话）
    并入同节上一片段，避免独立成块污染检索结果。
    """
    chunks: List[Dict[str, str]] = []
    merge_threshold = 60

    for title, paragraphs in sections:
        section_texts: List[str] = []
        current: List[str] = []
        current_len = 0

        def _push(parts: List[str]) -> None:
            text = "\n".join(parts).strip()
            if text:
                section_texts.append(text)

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size * 2:
                # 超长段落先独立切分
                for piece in _split_long_paragraph(paragraph, chunk_size):
                    if current_len + len(piece) > chunk_size and current:
                        _push(current)
                        current, current_len = [], 0
                    current.append(piece)
                    current_len += len(piece) + 1
                continue
            if current_len + len(paragraph) > chunk_size and current:
                _push(current)
                current, current_len = [], 0
            current.append(paragraph)
            current_len += len(paragraph) + 1

        if current:
            _push(current)

        # 尾部碎片合并
        if len(section_texts) >= 2 and len(section_texts[-1]) < merge_threshold:
            merged = section_texts[-2] + "\n" + section_texts[-1]
            if len(merged) <= chunk_size * 2:
                section_texts[-2:] = [merged]

        for text in section_texts:
            chunks.append({"title": title, "text": text})

    return chunks


def chunk_plain_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, str]]:
    """无结构兜底：滑动窗口切块。"""
    normalized = re.sub(r"[ \u3000]+", " ", re.sub(r"\n{2,}", "\n", text)).strip()
    if not normalized:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[Dict[str, str]] = []
    for start in range(0, len(normalized), step):
        piece = normalized[start : start + chunk_size].strip()
        if piece:
            chunks.append({"title": "全文", "text": piece})
        if start + chunk_size >= len(normalized):
            break
    return chunks
