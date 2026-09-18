"""插件内共享路径：包内模块据此定位随插件分发的数据，不受导入方式影响。

`PLUGIN_ROOT` 取包的父目录（插件根），无论 `knowledge` 是作为插件包的一部分
被 Runner 以相对导入加载，还是开发脚本以顶层包导入，路径都稳定指向插件目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# knowledge/ 的父目录即插件根
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# 运行时数据目录（萌百索引 + bangumi 库），随插件分发
KB_DATA_DIR = PLUGIN_ROOT / "kb_data"
# 重建用源数据目录（页面 / txt / bangumi 原始导出），不打包
TOPICS_DIR = PLUGIN_ROOT / "topics"


def bundled_index_paths(data_dir: Path = None) -> List[Path]:
    """索引文件列表：兼容纯 JSON（.index.json）与压缩版（.index.json.gz）。"""
    directory = data_dir if data_dir is not None else KB_DATA_DIR
    return sorted(
        set(directory.glob("*.index.json")) | set(directory.glob("*.index.json.gz"))
    )
