import json
from typing import Any

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
            "content": (
                "你是企业代码分析助手，工作区是 C++/Python/JSON 活动服仓库，不是小型 demo 项目。"
                "查询流程必须先宽后深："
                "1. 先调用 retrieve_code 做宽度定位，根据返回的 path/行号/符号判断候选；"
                "2. 再用 read_text_file 按 start_line/end_line 精读窗口，不要整文件读取；"
                "3. 仅当已经知道英文符号（如 RankNode、sid_name）时，才在具体子目录里用 search_text。"
                "禁止对仓库根调用 search_text。"
                "禁止用 list_files('.') 代替检索；list_files 只用于已知子目录列一层。"
                "咨询、bug 分析、方案检索默认只读。"
                "只有用户明确要求改代码，并且你已经精读目标窗口后，才使用 replace_text_in_file 或 write_text_file。"
                "不要声称代码已修改，除非你已经看到了真实工具结果。"
                "工作区只允许在 COMPANY_CODE_REPO_PATH。"
                "回答使用简洁清晰的中文。"
            ),
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
                "你正在分析企业活动服代码仓库。按「先 retrieve_code 定位，再按行精读」继续。\n"
                f"- goal: {state.get('goal')!r}\n"
                f"- phase: {state.get('phase')!r}\n"
                f"- last_retrieve: {json.dumps(state.get('last_retrieve') or [], ensure_ascii=False)}\n"
                f"- last_tool_name: {state.get('last_tool_name')!r}\n"
                f"- loop_count: {state.get('loop_count')!r}\n"
                "phase=locating 时先 retrieve_code；"
                "phase=has_candidates 时按 last_retrieve 的行号 read_text_file；"
                "phase=reading 时基于已读窗口作答或扩窗。"
                "咨询类任务不要修改代码。"
                "如果任务已完成，直接给出最终答复。"
            ),
        }
