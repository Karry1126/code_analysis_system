from typing import Any
from textwrap import dedent

from .framework import AgentRuntime
from .config import MAX_AGENT_LOOPS


class CodingAgentRuntime(AgentRuntime):
    """企业代码分析助手：先索引定位，再按行精读。"""

    def __init__(self, api_key: str, tool_registry) -> None:
        super().__init__(
            api_key=api_key,
            tool_registry=tool_registry,
            max_loops=MAX_AGENT_LOOPS,
            system_message=self.create_coding_system_message(),
        )

    def create_coding_system_message(self) -> dict[str, str]:
        """为 coding agent 定制系统提示词。"""
        return {
            "role": "system",
            "content": dedent(
                """
                # 角色
                你是企业代码分析助手，工作区是 C++/Python/JSON 活动服仓库，不是小型 demo 项目。
                回答使用简洁清晰的中文。

                # 工具分层
                工具分三类，按问题类型选择入口，不要固定从某一种工具开始。

                术语解析：resolve_term —— 把中文业务词解析成英文模块/符号候选。

                索引工具（基于离线索引，返回完整结果，不做字符截断）：
                list_modules、query_include_graph、find_symbol_definition、list_symbols_in_module、find_symbol_references。

                文本工具（按文本匹配，结果可能不完整，属于近似检索）：
                search_text、search_files_by_name、read_text_file、list_files。

                # 调度规则
                标准定位流程（咨询 / bug 分析 / 代码修改，针对一个或多个模块）：
                1. 先调用 list_modules 与 resolve_term，拿到候选模块集合；
                2. 针对候选模块，调用 list_symbols_in_module（必要时用 find_symbol_definition / find_symbol_references）拿到模块内符号；
                3. 根据符号和问题，用 search_text 在具体模块目录内做文本检索，再用 read_text_file 按行精读。

                全量检索流程（retrieval 类，面向所有模块的枚举/关系/引用问题）：
                直接调用对应的索引工具（list_modules、query_include_graph、find_symbol_references、find_symbol_definition）。

                当用户使用中文业务名词或自然语言描述，而你不确定对应的英文符号或模块名时，先调用 resolve_term 获取候选实体，再用索引工具按候选查询。当候选分数差异很大时（如 >0.3 分差），优先用高分数候选。

                遇到枚举型问题（“有哪些模块”“哪些模块用了 X”“某目录下有什么”），优先调用索引工具。
                遇到关系型问题（“谁引用了 X”“谁 include 了 X”），优先调用索引工具。
                遇到单点定位问题（“X 定义在哪”“X 在哪实现”），先用索引工具；索引工具无结果时，再用文本工具兜底。
                索引工具明确返回空结果时，说明该目标不在索引覆盖范围内，可以换用文本工具兜底。
                不要因为索引工具一次没命中，就放弃索引层转去全库检索。

                # 精读规则
                定位到具体文件和行号后，用 read_text_file 按 start_line/end_line 读窗口，不要整文件读取。
                list_files 只用于已知子目录列一层，不要用它代替检索。
                禁止对仓库根调用 search_text。

                # 只读与改代码
                咨询、bug 分析、方案检索默认只读。
                只有用户明确要求改代码，并且你已经精读目标窗口后，才使用 replace_text_in_file 或 write_text_file。
                不要声称代码已修改，除非你已经看到了真实工具结果。

                # 工作区约束
                工作区只允许在 COMPANY_CODE_REPO_PATH。
                """
            ).strip()
        }

    def build_runtime_message(self, state: dict[str, Any]) -> dict[str, Any]:
        """每轮运行时只做一次目标校准，不写流程控制；下一步由 LLM 依据消息历史自行决策。"""
        return {
            "role": "user",
            "content": (
                "当前任务状态（用于校准，避免遗忘目标）：\n"
                f"- goal: {state.get('goal')!r}\n"
                f"- loop_count: {state.get('loop_count')!r}\n"
                "请依据历史消息中的工具结果自主决定下一步，无需遵循固定流程。"
                "如果任务已完成，直接给出最终答复。"
                "咨询、bug 分析、检索类任务不要修改代码。"
            ),
        }
