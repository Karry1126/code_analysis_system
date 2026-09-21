import json
from typing import Any


def _has_tool_calls(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _split_atomic_blocks(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    把消息切成不可拆开的块。

    assistant(tool_calls) 和紧随其后的 tool 结果必须待在同一块里，
    否则裁剪后会出现孤立的 tool 消息，LLM API 会直接 400。
    """
    blocks: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _has_tool_calls(message):
            block = [message]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                block.append(messages[index])
                index += 1
            blocks.append(block)
            continue

        blocks.append([message])
        index += 1
    return blocks


def _flatten_blocks(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block in blocks:
        messages.extend(block)
    return messages


def _tool_calls_text(tool_calls: Any) -> str:
    if not tool_calls:
        return ""

    parts: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict):
            parts.append(json.dumps(call, ensure_ascii=False, default=str))
            continue

        function = getattr(call, "function", None)
        if function is None:
            parts.append(str(call))
            continue

        parts.append(str(getattr(function, "name", "") or ""))
        parts.append(str(getattr(function, "arguments", "") or ""))
    return "\n".join(parts)


def _estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    粗估单条消息占用的 token。

    不引入 tokenizer 依赖。对中文和代码按“约 2 字符 = 1 token”
    做偏保守估计，另加每条消息的角色开销。
    """
    content = message.get("content")
    if content is None:
        content_text = ""
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = json.dumps(content, ensure_ascii=False, default=str)

    serialized = "\n".join(
        [
            str(message.get("role", "")),
            content_text,
            _tool_calls_text(message.get("tool_calls")),
            str(message.get("tool_call_id", "") or ""),
            str(message.get("name", "") or ""),
        ]
    )
    return max(1, (len(serialized) + 1) // 2) + 4


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


class MessageStore:
    """
    管理会话消息和历史裁剪。
    """

    def __init__(self, max_input_tokens: int) -> None:
        self.max_input_tokens = max_input_tokens
        self.messages: list[dict[str, Any]] = []

    def append(self, message: dict[str, Any]) -> None:
        # 所有消息最终都通过这里进入存储，
        # 这样裁剪逻辑就不会散落在各个调用方里。
        self.messages.append(message)
        self.trim()

    def extend(self, new_messages: list[dict[str, Any]]) -> None:
        # 主要给“批量回写消息”的场景留接口，
        # 虽然当前 demo 用得不多，但作为框架层抽象更完整。
        self.messages.extend(new_messages)
        self.trim()

    def trim(self) -> None:
        """
        按 token 预算裁剪历史，并保证工具调用链完整。

        先把消息切成原子块：assistant(tool_calls) 和紧随其后的
        tool 结果视为同一块。从最旧的块开始丢弃，直到剩余消息
        估测 token 不超过预算。最新一块即使超预算也保留，
        避免把当前未写完的工具调用链拆开。
        """
        if not self.messages:
            return

        blocks = _split_atomic_blocks(self.messages)
        start = 0
        while start < len(blocks) - 1:
            remaining = _flatten_blocks(blocks[start:])
            if _estimate_tokens(remaining) <= self.max_input_tokens:
                break
            start += 1

        self.messages = _flatten_blocks(blocks[start:])

    def snapshot(self) -> list[dict[str, Any]]:
        """返回当前消息快照。"""
        return list(self.messages)
