import os
from pathlib import Path
from openai import OpenAI
from textwrap import dedent
from typing import Any
import pprint
import json

COMPANY_CODE_REPO_PATH = Path(__file__).parent.parent.parent.parent / "company_project/online"
MAX_AGENT_STEPS = 5
MAX_TOKENS = 10000

def build_system_prompt() -> dict[str, str]:
    """
    构建系统提示词
    """
    return {
        "role": "system",
        "content": dedent(
            """
            # 角色
            你是代码分析专家，负责分析代码并回答用户问题

            # 任务
            你将利用工具浏览公司里的代码库，并回答用户问题

            # 结果
            除非用户要求详细说明代码内容，否则你优先使用简洁、清晰的中文回答
            如果你不理解用户的问题，请直接告诉用户你无法回答
            如果你使用工具后，无法解决用户的问题，请直接告诉用户你无法回答
            """
        ).strip()
    }

def build_tools() -> list[dict[str, Any]]:
    """
    定义可供模型调用的工具
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_code_file",
                "description": (
                    "读取指定路径的代码文件，并返回代码内容"
                    "调用时机：当用户的问题需要依赖某个代码文件时"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "位于公司代码仓库的相对路径"
                        }
                    },
                "required": ["relative_path"]
                }
            }
        }
    ]

def read_code_file(relative_path: str) -> str:
    """ 
    将相对路径解析成绝对路径，返回代码内容和文件名
    """
    absolute_path = COMPANY_CODE_REPO_PATH / relative_path
    if not absolute_path.exists():
        raise RuntimeError(f"文件 {absolute_path} 不存在")
    with open(absolute_path, "r", encoding="utf-8") as f:
        return f.read()

def call_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]], api_key: str) -> OpenAI.ChatCompletionMessageParam:
    """
    调用LLM模型，返回模型响应
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=MAX_TOKENS,
        temperature=0.1,
    )
    return response.choices[0].message

def run_task_agent(messages: list[dict[str, Any]], tools: list[dict[str, Any]], api_key: str) -> str:
    """
    运行任务代理，返回模型响应
    """
    for _ in range(MAX_AGENT_STEPS):
        response = call_llm(messages, tools, api_key)
        pprint.pprint(response)
        if response.tool_calls:
            # message协议要求：工具执行结果前需要将调用工具的信息填充到message中
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            tool_results = []
            for tool_call in response.tool_calls:
                tool_result = {
                    "ok": False,
                    "error": "unknown tool call",
                }
                if tool_call.function.name == "read_code_file":
                    relative_path = json.loads(tool_call.function.arguments)["relative_path"]
                    try:
                        code = read_code_file(relative_path)
                        tool_result["ok"] = True
                        tool_result["code"] = code
                        tool_result.pop("error", None)
                    except RuntimeError as e:
                        tool_result["error"] = str(e)
                print(f"tool_result: {tool_result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })
            continue
        # 如果模型没有调用工具，则将模型响应添加到消息列表中
        messages.append({
            "role": "assistant",
            "content": response.content,
        })
        if response.content.strip():
            return response.content.strip()
    return "任务执行失败"

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    messages = [build_system_prompt()]
    tools = build_tools()

    while True:
        user_input = input("请输入问题: ")
        if not user_input.strip():
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        messages.append({"role": "user", "content": user_input})
        response = run_task_agent(messages, tools, api_key)
        print(f"\n[AI 回答]: {response}")

if __name__ == "__main__":
    main()