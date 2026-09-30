import json
from typing import Any
from textwrap import dedent

from .framework import AgentRuntime
from .config import MAX_AGENT_LOOPS, COMPANY_CODE_REPO_PATH


class CodingAgentRuntime(AgentRuntime):
    """企业代码分析助手：先宽度定位，再按行精读。"""

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
                工具分两层，按问题类型选择，不要固定从某一种工具开始。
                索引工具：list_modules、query_include_graph。它们基于离线索引，返回完整结果，不做字符截断。
                文本工具：retrieve_code、search_text、search_files_by_name、read_text_file、list_files。
                它们按文本匹配，结果可能不完整，属于近似检索。

                # 调度规则
                当用户使用中文业务名词或自然语言描述，而你不确定对应的英文符号或模块名时，先调用 resolve_term 获取候选实体，再用索引工具按候选查询。当候选分数差异很大时（如 >0.3 分差），优先用高分数候选。
                遇到枚举型问题（“有哪些模块”“哪些模块用了 X”“某目录下有什么”），优先调用索引工具。
                遇到关系型问题（“谁引用了 X”“谁 include 了 X”），优先调用索引工具。
                遇到单点定位问题（“X 定义在哪”“X 在哪实现”），先用索引工具；索引工具无结果时，再用文本工具。
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

    def create_state(self, goal: str) -> dict[str, Any]:
        state = super().create_state(goal)
        state["phase"] = "locating"
        state["last_retrieve"] = []
        state["shared_context"] = {
            "COMPANY_CODE_REPO_PATH": str(COMPANY_CODE_REPO_PATH),
        }
        return state

    def update_state_from_tool_result(
        self,
        state: dict[str, Any],
        tool_name: str,
        tool_result: dict[str, Any],
    ) -> None:
        super().update_state_from_tool_result(state, tool_name, tool_result)

        last_retrieve = state.get("shared_context", {}).get("last_retrieve")
        if isinstance(last_retrieve, list):
            state["last_retrieve"] = last_retrieve

        if tool_name == "read_text_file" and tool_result.get("ok") and not tool_result.get("preview"):
            state["phase"] = "reading"
        elif state.get("last_retrieve"):
            state["phase"] = "has_candidates"
        else:
            state["phase"] = "locating"

    def build_runtime_message(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                "你正在分析企业活动服代码仓库。按问题类型选择工具层，不要固定从 retrieve_code 开始。\n"
                f"- goal: {state.get('goal')!r}\n"
                f"- phase: {state.get('phase')!r}\n"
                f"- last_retrieve: {json.dumps(state.get('last_retrieve') or [], ensure_ascii=False)}\n"
                f"- last_tool_name: {state.get('last_tool_name')!r}\n"
                f"- loop_count: {state.get('loop_count')!r}\n"
                "phase=locating 时按问题类型选层："
                "枚举型、关系型、单点定位优先用索引工具 list_modules、query_include_graph；"
                "索引工具明确无结果，或问题需要文本匹配时，再用 retrieve_code、search_text、search_files_by_name。"
                "索引工具已给出完整结果时直接作答，不必再调用 retrieve_code。"
                "phase=has_candidates 时按 last_retrieve 的行号 read_text_file；"
                "phase=reading 时基于已读窗口作答或扩窗。"
                "咨询类任务不要修改代码。"
                "如果任务已完成，直接给出最终答复。"
            ),
        }
