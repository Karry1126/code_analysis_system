"""Agent 可编程入口：供 CLI 与评测 harness 复用。

把 coding_agent.main() 里的「建 registry + 建 runtime」抽出成 build_agent()，
并提供 run_agent()，让评测 harness 可以多次、结构化地驱动同一套 Agent。
"""

from __future__ import annotations

from typing import Any

from .coding_runtime import CodingAgentRuntime
from .config import MAX_HISTORY_INPUT_TOKENS, get_api_key
from .framework import MessageStore, ToolRegistry
from .tools import register_coding_tools
from .tools_index import register_index_tools
from .tools_references import register_reference_tools
from .tools_rag import register_rag_tools
from .tools_symbol import register_symbol_tools


def build_registry() -> ToolRegistry:
    """构造并注册全部工具的 ToolRegistry。"""
    registry = ToolRegistry()
    register_coding_tools(registry)
    register_index_tools(registry)
    register_symbol_tools(registry)
    register_reference_tools(registry)
    register_rag_tools(registry)
    return registry


def build_agent(
    api_key: str | None = None,
    tool_registry: ToolRegistry | None = None,
) -> CodingAgentRuntime:
    """构造 CodingAgentRuntime。

    api_key 缺省时从环境变量读取；tool_registry 缺省时调用 build_registry()。
    """
    if api_key is None:
        api_key = get_api_key()
    if tool_registry is None:
        tool_registry = build_registry()
    return CodingAgentRuntime(api_key=api_key, tool_registry=tool_registry)


def run_agent(
    question: str,
    *,
    api_key: str | None = None,
    n_runs: int = 1,
) -> list[dict[str, Any]]:
    """对同一个问题跑 n_runs 次，返回结构化结果列表。

    每次运行使用全新的 MessageStore，互不污染。
    结果字段：run_id / question / answer / status / error / steps / metrics。
    """
    agent = build_agent(api_key=api_key)
    results: list[dict[str, Any]] = []
    for i in range(1, n_runs + 1):
        message_store = MessageStore(max_input_tokens=MAX_HISTORY_INPUT_TOKENS)
        message_store.append({"role": "user", "content": question})
        outcome = agent.run_traced(goal=question, message_store=message_store)
        results.append(
            {
                "run_id": i,
                "question": question,
                "answer": outcome.answer,
                "status": outcome.status,
                "error": outcome.error,
                "steps": outcome.steps,
                "metrics": outcome.metrics,
            }
        )
    return results
