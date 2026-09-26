
from .framework import MessageStore, ToolRegistry
from .coding_runtime import CodingAgentRuntime
from .config import MAX_HISTORY_INPUT_TOKENS, COMPANY_CODE_RELATIVE_PATH, get_api_key
from .tools import register_coding_tools
from .tools_index import register_index_tools
from .tools_symbol import register_symbol_tools
import traceback


def main() -> None:
    api_key = get_api_key()
    registry = ToolRegistry()
    register_coding_tools(registry)
    register_index_tools(registry)
    register_symbol_tools(registry)

    # 这里直接复用 .framework 里的 ToolRegistry 和 MessageStore。
    runtime = CodingAgentRuntime(api_key=api_key, tool_registry=registry)
    message_store = MessageStore(max_input_tokens=MAX_HISTORY_INPUT_TOKENS)

    print("Coding Agent Demo 已启动。输入 exit 或 quit 结束。")
    print(f"工作区目录：{COMPANY_CODE_RELATIVE_PATH}")
    print(f"当前会保留约 {MAX_HISTORY_INPUT_TOKENS} tokens 的会话记忆。")

    while True:
        user_goal = input("\n你：").strip()

        if not user_goal:
            print("请输入任务目标。")
            continue

        if user_goal.lower() in {"exit", "quit"}:
            print("对话结束。")
            break

        # coding agent 同样使用 message history。
        # 这意味着多轮任务约束、上文要求、前一次修改结论都能被保留下来。
        message_store.append({"role": "user", "content": user_goal})

        try:
            final_answer = runtime.run(goal=user_goal, message_store=message_store)
        except Exception as exc:
            traceback.print_exc()
            print(f"\n执行失败：{exc}")
            if message_store.messages and message_store.messages[-1]["role"] == "user":
                message_store.messages.pop()
            continue

        print(f"\n助手：{final_answer}")


if __name__ == "__main__":
    main()
