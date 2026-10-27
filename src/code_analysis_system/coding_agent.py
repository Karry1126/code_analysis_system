import traceback
import uuid

from .agent_api import build_langgraph_agent
from .config import COMPANY_CODE_RELATIVE_PATH, MAX_HISTORY_INPUT_TOKENS, get_api_key


def main() -> None:
    api_key = get_api_key()
    runtime = build_langgraph_agent(api_key=api_key)
    # 进程级 session_id 作为 thread_id，维持多轮记忆（替代 MessageStore）。
    session_id = uuid.uuid4().hex

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

        # LangGraph 的 InMemorySaver 以 thread_id 分区维护多轮记忆：
        # 多轮任务约束、上文要求、前一次修改结论都被保留下来。
        try:
            outcome = runtime.run(goal=user_goal, thread_id=session_id)
        except Exception as exc:
            traceback.print_exc()
            print(f"\n执行失败：{exc}")
            continue

        if outcome.status != "completed":
            print(f"\n执行失败：{outcome.error or outcome.status}")
            continue

        print(f"\n助手：{outcome.answer}")


if __name__ == "__main__":
    main()
