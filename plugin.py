"""知识库插件 — 自带离线知识数据，并提供 LLM 检索工具。

知识库分两部分，均随插件自带、离线可用（实现见 `knowledge/` 包）：
- **萌娘百科**：kb_data/*.index.json（由 knowledge.builder 从 topics/<主题>/txt 构建），
  安装即用，运行时不依赖任何外部 txt；
- **bangumi**：kb_data/bangumi.db（SQLite FTS5，由 build_bangumi_db.py 构建），
  收录动画/漫画/游戏/音乐/三次元作品及角色、人物；
- 注册四个 Tool：search_knowledge（萌娘百科）、search_bangumi（bangumi 检索）、
  get_bangumi_season（季度新番列表）、browse_bangumi（按标签/评分筛选作品），
  均为离线检索（jieba + BM25 / FTS5，零外部调用）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from maibot_sdk import (
    Command,
    CONFIG_RELOAD_SCOPE_SELF,
    Field,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ToolParameterInfo, ToolParamType

try:  # Runner 以包形式加载 plugin.py（submodule_search_locations=插件目录，
    # 见 standards/sdk-usage.md「多文件插件与本地模块导入」），本地模块用相对导入
    from . import knowledge
    from .knowledge import bangumi
except ImportError:  # 仅独立调试（插件目录在 sys.path）时走这里
    import knowledge
    from knowledge import bangumi

SUPPORTED_CONFIG_VERSION = "1.0.0"

# 描述中的主题数量上限：知识库再多也只列前 5 个 + "等"
_THEME_LIMIT = 5
# 检索未命中时向模型展示的收录主题上限
_TOPIC_LIMIT = 12
# 源文件名中的通用词，提取主题时剔除
_GENERIC_NAME_TOKENS = ("知识库", "资料库", "百科", "合集", "大全")

# bangumi 库缺失时统一的工具返回（多处复用，避免重复大段提示文本）：
# 数据不随包分发属正常状态，按"未收录"温和告知，不算工具故障（success=True）
_BANGUMI_UNAVAILABLE = {
    "success": True,
    "content": (
        "bangumi 资料库未随本插件安装（缺少 kb_data/bangumi.db），本次查询不可用；"
        "可用 search_knowledge 检索已收录的萌娘百科主题。"
    ),
}

# 工具结果只进 planner：replyer 构建请求时显式过滤 ToolResultMessage
# （MaiBot maisaka_generator_base._should_keep_replyer_history_message）。
# planner→replyer 的官方信息通道是 reply 工具参数 reply_reference（schema 描述明说
# 收"事实信息"，渲染为回复参考块、优先级最高）。
# 因此工具的成功返回文本末尾都带这段指引（与用户在 WebUI 提示词里的规则保持同一口径），
# 让 planner 主动把要点带过去。
_TOOL_RESULT_CARRY_HINT = (
    "\n\n（系统提示：调用 reply 时，必须把本次回复需要引用的上述工具关键结果"
    "客观摘要写入 reply_reference 参数，否则最终回复无法使用这些资料。）"
)


def _theme_from_sources(data: Dict[str, Any]) -> str:
    """回退主题：从索引 sources 的首个源文件名剔除"知识库"等通用词。"""
    for source in data.get("sources", []):
        name = str(source.get("name", "")) if isinstance(source, dict) else str(source)
        stem = Path(name).stem.strip()
        for token in _GENERIC_NAME_TOKENS:
            stem = stem.replace(token, "")
        return stem.strip(" -_、，")
    return ""


def _index_theme(path: Path) -> str:
    """取单个索引的显示主题：优先读头部 display_name（轻量），缺省再解析整包 sources。"""
    theme = knowledge.read_index_theme(path)
    if theme:
        return theme
    return _theme_from_sources(knowledge.read_index_file(path))


def _build_knowledge_themes() -> str:
    """从内置知识数据提取显示主题（模块导入时执行一次，进 Tool 描述）。

    优先用构建时写入的 display_name（如"崩坏：星穹铁道"）。主题名在索引文件头部，
    只读前缀即可拿到，不必为拼描述而完整解析十几 MB 的词频表（那会拖慢插件导入）。
    """
    try:
        themes: List[str] = []
        for path in knowledge.bundled_index_paths():
            theme = _index_theme(path)
            if theme and theme not in themes:
                themes.append(theme)
        if not themes:
            return "（空）"
        return "、".join(themes[:_THEME_LIMIT]) + ("等" if len(themes) > _THEME_LIMIT else "")
    except Exception:
        return "（未加载）"


_KNOWLEDGE_THEMES = _build_knowledge_themes()


def _build_library_summary() -> str:
    """生成"已加载知识库"界面提示文本（模块导入时检测一次内置数据）。

    用于 WebUI 配置页的只读字段：插件未运行时也能看到随包收录了哪些知识库；
    运行中的加载详情（片段数/构建时间）用 /kb_stats 命令查看。
    """
    try:
        themes: List[str] = []
        for path in knowledge.bundled_index_paths():
            theme = _index_theme(path)
            if theme and theme not in themes:
                themes.append(theme)
        if themes:
            moegirl = "萌娘百科：" + "、".join(themes[:_THEME_LIMIT]) + (
                "等" if len(themes) > _THEME_LIMIT else ""
            )
        else:
            moegirl = "萌娘百科：（未找到内置索引数据）"
        if bangumi.available():
            detail = ""
            try:
                info = bangumi.stats()
                total = sum(info.get("counts", {}).values())
                source_date = str(info.get("source_date") or "").strip()
                parts = []
                if total:
                    parts.append(f"约 {total / 10000:.1f} 万条")
                if source_date:
                    parts.append(f"数据 {source_date}")
                detail = f"（{'，'.join(parts)}）" if parts else ""
            except Exception:
                detail = ""
            bangumi_state = f"bangumi 资料库：已内置{detail}"
        else:
            bangumi_state = "bangumi 资料库：（未内置）"
        return f"{moegirl}；{bangumi_state}"
    except Exception:
        return "（内置数据检测失败）"


_LIBRARY_SUMMARY = _build_library_summary()


def _format_ranked(header: str, results: List[Dict[str, Any]]) -> str:
    """把检索结果格式化为「【N】标题（相关度）+ 正文」的 LLM 可读文本。"""
    lines = [header, ""]
    for order, result in enumerate(results, 1):
        lines.append(f"【{order}】{result['title']}（相关度 {result['score']}）")
        lines.append(result["text"])
        lines.append("")
    return "\n".join(lines).strip()


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本",
        json_schema_extra={"hidden": True},
    )


class KnowledgeConfig(PluginConfigBase):
    """知识库配置。"""

    __ui_label__ = "知识库"
    __ui_icon__ = "library_books"
    __ui_order__ = 1

    top_k: int = Field(
        default=4,
        description="每次检索默认返回的片段数",
        json_schema_extra={
            "label": "默认返回条数",
            "hint": "1-8；模型调用工具时可临时覆盖，此处是未指定时的默认值",
        },
    )
    max_chars: int = Field(
        default=300,
        description="单个片段返回给模型的最大字符数",
        json_schema_extra={
            "label": "片段字符上限",
            "hint": "200-800；越大信息越全、占用的 token 也越多",
        },
    )
    bundled_libraries: str = Field(
        default=_LIBRARY_SUMMARY,
        description="已加载知识库（自动检测插件内置数据，只读提示）",
        json_schema_extra={
            "label": "已加载知识库",
            "disabled": True,
            "hint": "打开配置页时实时检测（含 bangumi 数据日期）；运行详情（片段数/构建时间）可发 /kb_stats 查看",
        },
    )


class KnowledgeBasePluginConfig(PluginConfigBase):
    """知识库插件完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)


class KnowledgeBasePlugin(MaiBotPlugin):
    """本地知识库插件。"""

    config_model = KnowledgeBasePluginConfig

    def __init__(self) -> None:
        super().__init__()
        self._index: Optional[knowledge.KnowledgeIndex] = None
        self._lock = asyncio.Lock()
        self._load_task: Optional[asyncio.Task] = None

    def get_webui_config_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> Dict[str, Any]:
        """配置 Schema：把「已加载知识库」只读字段刷新为实时检测值。

        宿主的字段 default 只在生成 config.toml 时写入一次、之后不再变化；
        这里在每次打开配置页时用当前实测结果覆盖展示值（含 bangumi 数据日期），
        避免换过 kb_data 后页面仍显示旧信息。
        """
        schema = super().get_webui_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        try:
            section = (schema.get("sections") or {}).get("knowledge") or {}
            fields = section.get("fields") or {}
            field = fields.get("bundled_libraries")
            if isinstance(field, dict):
                field["default"] = _build_library_summary()
        except Exception:
            pass
        return schema

    # ── 生命周期 ──

    async def on_load(self) -> None:
        """启动后台任务加载内置知识数据（不阻塞宿主启动流程）。

        索引文件较大（几 MB 到几十 MB，解析需数秒），放在后台任务里加载；
        检索工具执行时若仍在加载会短暂等待（见 _wait_index）。数据缺失/损坏
        一律温和降级（对应工具返回"未收录"提示），不作为错误上报。
        """
        self._start_load("知识库初始化")

    def _start_load(self, reason: str) -> None:
        """起一个后台加载任务（配置热更新重载与首次启动共用）。"""
        self._load_task = asyncio.create_task(self._load_in_background(reason))

    async def _load_in_background(self, reason: str) -> None:
        """后台加载知识库并记录日志（数据缺失属正常状态，不打扰宿主）。"""
        try:
            summary = await self._ensure_index()
        except Exception as e:  # 防御：加载逻辑本身的意外异常也不向上抛
            self.ctx.logger.warning("%s异常: %s", reason, e)
            return
        if not summary.get("files"):
            self.ctx.logger.info(
                "%s：未找到内置知识数据（kb_data/ 为空或缺失），"
                "search_knowledge 将提示未收录，其余功能不受影响",
                reason,
            )
            return
        self.ctx.logger.info("%s完成：%s", reason, self._format_summary(summary))
        broken = getattr(self._index, "broken_files", None) if self._index else None
        for name in broken or []:
            self.ctx.logger.warning("知识库索引文件损坏已跳过: %s", name)
        await self._log_bangumi_status()

    async def _log_bangumi_status(self) -> None:
        """记录 bangumi 资料库状态（未内置属正常状态，info 级说明即可）。"""
        if not bangumi.available():
            self.ctx.logger.info("bangumi 资料库未随插件安装，search_bangumi 等工具将提示未收录")
            return
        try:
            info = await asyncio.to_thread(bangumi.stats)
        except Exception as e:
            self.ctx.logger.warning("bangumi 资料库读取异常: %s", e)
            return
        total = sum(info.get("counts", {}).values())
        self.ctx.logger.info(
            "bangumi 资料库就绪：%d 条（%.1f MB，数据 %s）",
            total,
            info.get("size_mb", 0.0),
            info.get("source_date") or "未知",
        )

    async def on_unload(self) -> None:
        """卸载时取消后台加载并释放索引。"""
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
        self._load_task = None
        self._index = None

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        """配置热更新：检索参数（返回条数 / 字符上限）在检索时实时读取，自动生效。

        索引数据与配置无关，这里不再重新加载（此前每次保存配置都会重跑一次
        数秒的索引加载，属无谓开销）。
        """
        del config_data
        del version
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        self.ctx.logger.info("知识库配置已更新（检索参数即时生效，索引无需重载）")

    # ── 内部：检索参数与守卫 ──

    def _clamp_retrieval(self, top_k_raw: int) -> Tuple[int, int]:
        """把工具入参与配置钳制到合法区间，返回 (top_k, max_chars)。"""
        cfg = self.config.knowledge
        top_k = int(top_k_raw or 0) or int(cfg.top_k)
        return max(1, min(top_k, 8)), max(100, min(int(cfg.max_chars), 1000))

    # ── Tool：LLM 检索入口 ──

    @Tool(
        "search_knowledge",
        brief_description=f"检索萌娘百科离线知识库（收录：{_KNOWLEDGE_THEMES}）的词条片段",
        detailed_description=(
            f"检索本地萌娘百科离线知识库，收录主题：{_KNOWLEDGE_THEMES}。"
            "何时调用：用户询问收录主题的游戏设定、角色资料、技能/命之座机制、剧情、台词、"
            "道具、用语梗等背景事实；聊角色人气或萌战（世萌/B萌、萌王、历年冠军）；"
            "被问「你喜欢哪个角色」「哪个角色可爱」等偏好问题时。"
            "何时不用：查动画/漫画/游戏作品的播出时间、评分、制作阵容、声优（用 search_bangumi）；"
            "找季度新番或按条件筛选作品（用 get_bangumi_season / browse_bangumi）；"
            "与收录主题无关或需要实时信息时不要调用。"
        ),
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="2-4 个空格分隔的关键词（如「三月七 星魂」「世萌 萌王」），优先官方译名/全名；偏好类问题（你喜欢谁/谁可爱）可传原句",
                required=True,
            ),
            ToolParameterInfo(
                name="top_k",
                param_type=ToolParamType.INTEGER,
                description="返回条数 1-8；0 表示用插件配置默认",
                required=False,
                default=0,
            ),
        ],
    )
    async def handle_search_knowledge(self, query: str = "", top_k: int = 0, **kwargs):
        """执行知识库检索并返回 LLM 可读的片段列表。"""
        del kwargs
        query = str(query or "").strip()
        if not query:
            return {"success": False, "content": "检索词为空，请提供要查询的关键词。"}

        index = await self._wait_index()
        if index is None or index.size == 0:
            # 数据缺失属正常状态：温和告知而非报错，避免 LLM 把它当工具故障
            return {
                "success": True,
                "content": (
                    "当前插件未收录知识数据（kb_data/ 目录为空或缺失），"
                    "无法检索。"
                ),
            }

        effective_top_k, max_chars = self._clamp_retrieval(top_k)
        try:
            results = await asyncio.to_thread(index.search, query, effective_top_k, max_chars)
        except Exception as e:
            self.ctx.logger.error("知识库检索失败: %s", e)
            return {"success": False, "content": f"知识库检索出错：{e}"}

        if not results:
            # 未命中时直接告知知识库主题，引导 LLM 换更贴切的 query 再查
            topics = index.topics
            if topics:
                topics_text = "、".join(topics[:_TOPIC_LIMIT]) + (
                    "等" if len(topics) > _TOPIC_LIMIT else ""
                )
                return {
                    "success": True,
                    "content": (
                        f"知识库中没有找到与「{query}」直接相关的内容。"
                        f"知识库当前收录的主题：{topics_text}。"
                        "可参考主题换更贴切的 query 再次检索。"
                    ),
                }
            return {"success": True, "content": f"知识库中没有找到与「{query}」相关的内容。"}

        header = f"知识库检索「{query}」，共 {len(results)} 条相关片段："
        return {"success": True, "content": _format_ranked(header, results) + _TOOL_RESULT_CARRY_HINT}

    # ── Tool：bangumi 资料库检索 ──

    @Tool(
        "search_bangumi",
        brief_description="查作品/角色/人物在 Bangumi 的条目资料（时间/评分/制作阵容/声优/关联）",
        detailed_description=(
            "检索 Bangumi（番组计划）离线资料库：动画/漫画/游戏/音乐/三次元作品条目"
            "及角色、人物，含播出/发行时间、评分、制作人员、声优、登场角色、续作/改编关联。"
            "何时调用：问某部作品的客观资料，或某角色/人物出演、参与过哪些作品"
            "（如「进击的巨人 评分」「牧濑红莉栖 配过什么」）。"
            "何时不用：查收录游戏内的设定/技能/剧情细节（用 search_knowledge）；"
            "找季度新番榜单（用 get_bangumi_season）；按标签/评分筛选作品（用 browse_bangumi）。"
        ),
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="作品官方中文名/常见译名、角色或人物全名，支持多词（如「进击的巨人」「牧濑红莉栖」）",
                required=True,
            ),
            ToolParameterInfo(
                name="top_k",
                param_type=ToolParamType.INTEGER,
                description="返回条数 1-8；0 表示用插件配置默认",
                required=False,
                default=0,
            ),
        ],
    )
    async def handle_search_bangumi(self, query: str = "", top_k: int = 0, **kwargs):
        """执行 bangumi 资料库检索并返回 LLM 可读的条目列表。"""
        del kwargs
        query = str(query or "").strip()
        if not query:
            return {"success": False, "content": "检索词为空，请提供作品名/角色名/人名。"}
        if not bangumi.available():
            return _BANGUMI_UNAVAILABLE

        effective_top_k, max_chars = self._clamp_retrieval(top_k)
        try:
            results = await asyncio.to_thread(
                bangumi.search, query, effective_top_k, max_chars
            )
        except Exception as e:
            self.ctx.logger.error("bangumi 检索失败: %s", e)
            return {"success": False, "content": f"bangumi 检索出错：{e}"}

        if not results:
            return {
                "success": True,
                "content": (
                    f"bangumi 库中没有找到与「{query}」相关的条目。"
                    "可换用作品的官方译名或角色全名重试。"
                ),
            }
        header = f"bangumi 检索「{query}」，共 {len(results)} 条："
        return {"success": True, "content": _format_ranked(header, results) + _TOOL_RESULT_CARRY_HINT}

    # ── Tool：季度新番列表 ──

    @Tool(
        "get_bangumi_season",
        brief_description="列出某季度（1/4/7/10 月番）的新番动画，按追番人数排序",
        detailed_description=(
            "按季度列出 Bangumi 收录的动画新番，按追番人数降序。"
            "何时调用：「这季有什么新番」「2026 年 7 月新番有哪些」「最近有什么新番推荐」"
            "这类按时间看番/追番的问题。"
            "何时不用：查某一部作品的资料（用 search_bangumi）；"
            "按标签/评分筛选作品（用 browse_bangumi）。"
        ),
        parameters=[
            ToolParameterInfo(
                name="year",
                param_type=ToolParamType.INTEGER,
                description="年份；0 表示今年",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="month",
                param_type=ToolParamType.INTEGER,
                description="月份 1-12；0 表示当前季度，季度内任意一月均可（如 9 月即 7 月番季度）",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="limit",
                param_type=ToolParamType.INTEGER,
                description="返回条数 1-40；0 表示默认 20",
                required=False,
                default=0,
            ),
        ],
    )
    async def handle_get_bangumi_season(
        self, year: int = 0, month: int = 0, limit: int = 0, **kwargs
    ):
        """按季度列出新番动画（按追番人数排序）。"""
        del kwargs
        if not bangumi.available():
            return _BANGUMI_UNAVAILABLE
        try:
            # 新番列表默认给 20 条（不跟随 top_k 配置：检索条数与列表长度需求不同）
            data = await asyncio.to_thread(
                bangumi.season,
                int(year or 0),
                int(month or 0),
                int(limit or 0) or 20,
            )
        except Exception as e:
            self.ctx.logger.error("bangumi 季度查询失败: %s", e)
            return {"success": False, "content": f"bangumi 季度查询出错：{e}"}
        if not data["items"]:
            return {
                "success": True,
                "content": f"{data['label']}：资料库中没有收录的动画（可换个年份/季度试试）。",
            }
        lines = [
            f"{data['label']}（收录 {data['total']} 部，按追番人数列前 {len(data['items'])} 部）：",
            "",
        ]
        for order, item in enumerate(data["items"], 1):
            meta = " | ".join(
                part
                for part in (
                    f"开播 {item['date']}" if item["date"] else "",
                    f"评分 {item['score']}" if item["score"] else "",
                    f"追番 {item['favorite']}" if item["favorite"] else "",
                )
                if part
            )
            lines.append(f"【{order}】{item['name']}（{meta}）")
            if item["extra"]:
                lines.append(item["extra"])
            lines.append("")
        return {"success": True, "content": "\n".join(lines).strip() + _TOOL_RESULT_CARRY_HINT}

    # ── Tool：条件筛选作品 ──

    @Tool(
        "browse_bangumi",
        brief_description="按标签/类型/年份/评分筛选 Bangumi 作品（发现/推荐类查询）",
        detailed_description=(
            "按条件筛选作品列表，用于推荐与发现：「2025 年的高分科幻动画」"
            "「评分 8 以上的恋爱漫画」「冷门但评价好的日常番」。"
            "tag 与 keyword 至少提供一个。"
            "何时调用：用户给出标签、类型、年份、评分等组合条件找一批作品时。"
            "何时不用：查某一部作品/角色资料（用 search_bangumi）；"
            "查某季度新番榜单（用 get_bangumi_season）。"
        ),
        parameters=[
            ToolParameterInfo(
                name="tag",
                param_type=ToolParamType.STRING,
                description="Bangumi 常用标签（科幻/日常/催泪/搞笑/恋爱…）；与 keyword 至少给一个",
                required=False,
            ),
            ToolParameterInfo(
                name="keyword",
                param_type=ToolParamType.STRING,
                description="附加关键词；与 tag 至少给一个",
                required=False,
            ),
            ToolParameterInfo(
                name="subject_type",
                param_type=ToolParamType.STRING,
                description="作品类型",
                required=False,
                enum_values=["动画", "漫画", "游戏", "音乐", "三次元"],
                default="动画",
            ),
            ToolParameterInfo(
                name="year_from",
                param_type=ToolParamType.INTEGER,
                description="起始年份（如 2025）；0 表示不限",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="min_score",
                param_type=ToolParamType.INTEGER,
                description="最低评分（如 8）；0 表示不限",
                required=False,
                default=0,
            ),
            ToolParameterInfo(
                name="sort",
                param_type=ToolParamType.STRING,
                description="排序字段",
                required=False,
                enum_values=["score", "favorite", "date"],
                default="score",
            ),
            ToolParameterInfo(
                name="limit",
                param_type=ToolParamType.INTEGER,
                description="返回条数 1-40；0 表示默认 15",
                required=False,
                default=0,
            ),
        ],
    )
    async def handle_browse_bangumi(
        self,
        tag: str = "",
        keyword: str = "",
        subject_type: str = "动画",
        year_from: int = 0,
        min_score: int = 0,
        sort: str = "score",
        limit: int = 0,
        **kwargs,
    ):
        """按标签/类型/年份/评分筛选作品（发现型查询）。"""
        del kwargs
        if not bangumi.available():
            return _BANGUMI_UNAVAILABLE
        tag = str(tag or "").strip()
        keyword = str(keyword or "").strip()
        if not tag and not keyword:
            return {"success": False, "content": "请给出要筛选的标签或关键词（如 tag=喜剧）。"}
        sort_key = str(sort or "score").strip().lower()
        sort_label = {"score": "评分", "favorite": "追番人数", "date": "开播时间"}.get(
            sort_key, "评分"
        )
        try:
            results = await asyncio.to_thread(
                bangumi.browse,
                tag,
                keyword,
                bangumi.subject_type_of(subject_type),
                int(year_from or 0),
                0,
                float(min_score or 0),
                sort_key,
                int(limit or 0) or 15,
            )
        except Exception as e:
            self.ctx.logger.error("bangumi 筛选失败: %s", e)
            return {"success": False, "content": f"bangumi 筛选出错：{e}"}
        conditions = "、".join(
            part
            for part in (
                f"标签「{tag}」" if tag else "",
                f"关键词「{keyword}」" if keyword else "",
                str(subject_type or "动画"),
                f"{int(year_from)} 年起" if year_from else "",
                f"评分 ≥{min_score}" if min_score else "",
            )
            if part
        )
        if not results:
            return {
                "success": True,
                "content": f"没有符合条件的作品（{conditions}）。可放宽年份或评分再试。",
            }
        header = f"按{sort_label}排序，符合条件（{conditions}）的作品共 {len(results)} 条："
        lines = [header, ""]
        for order, item in enumerate(results, 1):
            lines.append(f"【{order}】{item['title']}")
            lines.append(item["text"])
            lines.append("")
        return {"success": True, "content": "\n".join(lines).strip() + _TOOL_RESULT_CARRY_HINT}

    # ── Command：管理命令 ──

    @Command(
        "kb_stats",
        description="查看知识库状态",
        pattern=r"^/kb_stats$",
        permission="operator",
    )
    async def handle_kb_stats(self, stream_id: str = "", **kwargs):
        """查看萌娘百科与 bangumi 两个知识库的状态。"""
        del kwargs
        index = await self._wait_index()
        if index is None or index.size == 0:
            await self.ctx.send.text("知识库状态：未收录知识数据（kb_data/ 为空或缺失）。", stream_id)
            return True, "知识库未收录数据", 2
        message = (
            f"知识库状态：萌娘百科 {index.size} 个片段；"
            f"索引构建时间 {index.built_at}。"
        )
        broken = getattr(index, "broken_files", None)
        if broken:
            message += f" 有 {len(broken)} 个索引文件损坏被跳过（{'、'.join(broken)}）。"
        if bangumi.available():
            try:
                info = await asyncio.to_thread(bangumi.stats)
                total = sum(info.get("counts", {}).values())
                source_date = str(info.get("source_date") or "").strip()
                date_part = f"，数据 {source_date}" if source_date else ""
                message += (
                    f" bangumi 资料库：{total} 条"
                    f"（{info.get('size_mb', 0.0)} MB{date_part}）。"
                )
            except Exception as e:
                message += f" bangumi 资料库读取失败：{e}。"
        else:
            message += " bangumi 资料库未构建。"
        await self.ctx.send.text(message, stream_id)
        return True, "已发送知识库状态", 2

    # ── 内部方法 ──

    def _bundled_data_dir(self) -> Path:
        """插件目录内自带的内置知识数据目录（随插件分发，只读）。"""
        return Path(__file__).resolve().parent / "kb_data"

    async def _wait_index(self) -> Optional[knowledge.KnowledgeIndex]:
        """等待后台加载任务完成（若仍在进行），返回当前索引。

        加载通常几秒内完成：检索请求先等它一下，而不是直接报"未就绪"。
        超时（含 shield 保护）只放弃等待、不取消加载任务本身。
        """
        task = self._load_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15)
            except Exception:
                pass  # 超时或任务失败：按当前状态返回（失败详情见 _load_error）
        return self._index

    async def _ensure_index(self) -> Dict[str, Any]:
        """确保索引可用：加载 kb_data/ 下的全部索引文件，返回加载摘要。

        容错设计：目录为空/缺失返回空摘要（不抛错）；单个文件损坏由
        KnowledgeIndex.load_combined 跳过并记入 broken_files。
        """
        async with self._lock:

            def _task() -> Dict[str, Any]:
                index_paths = knowledge.bundled_index_paths(self._bundled_data_dir())
                if not index_paths:
                    self._index = None
                    return {"chunks": 0, "files": 0}
                self._index = knowledge.KnowledgeIndex.load_combined(index_paths)
                return {"chunks": self._index.size, "files": len(index_paths)}

            return await asyncio.to_thread(_task)

    @staticmethod
    def _format_summary(summary: Dict[str, Any]) -> str:
        """把加载摘要格式化为日志文本。"""
        return f"{summary.get('files', 0)} 个索引文件，共 {summary.get('chunks', 0)} 个片段"


def create_plugin() -> KnowledgeBasePlugin:
    """创建知识库插件实例。"""
    return KnowledgeBasePlugin()
