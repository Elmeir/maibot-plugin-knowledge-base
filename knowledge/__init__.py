"""知识库包：萌娘百科离线检索 + bangumi 资料库接口。

按"运行时 / 构建期"分层，职责与依赖单向收敛：

运行时（插件启动即加载）
- `index.KnowledgeIndex`   BM25 索引：加载 / 合并 / 检索 / 打分
- `text`                   分词、查询清洗、简称近似、开窗截断
- `paths`                  插件根 / kb_data 目录 / 索引发现
- `bangumi`                bangumi SQLite FTS5 只读检索接口

构建期（离线生成索引，运行时不导入）
- `cleaning` / `parsing`   txt 清洗、结构解析、分块
- `builder`                txt → 内置索引的构建入口

本 `__init__` 只重导出运行时公共出口；构建期模块需按需 `from knowledge import builder`
显式导入，避免把离线清洗逻辑拖进插件启动路径。
"""

from __future__ import annotations

from . import bangumi, paths, text
from .index import (
    ENGINE_VERSION,
    INDEX_FORMAT_VERSION,
    KnowledgeIndex,
    read_index_file,
    read_index_theme,
)
from .paths import KB_DATA_DIR, PLUGIN_ROOT, TOPICS_DIR, bundled_index_paths
from .text import tokenize

__all__ = [
    "ENGINE_VERSION",
    "INDEX_FORMAT_VERSION",
    "KnowledgeIndex",
    "read_index_file",
    "read_index_theme",
    "tokenize",
    "bangumi",
    "paths",
    "text",
    "PLUGIN_ROOT",
    "KB_DATA_DIR",
    "TOPICS_DIR",
    "bundled_index_paths",
]
