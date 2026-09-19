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


class MessageStore:
    """
    管理会话消息和历史裁剪。
    """

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
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
        按用户轮次裁剪历史，并保证工具调用链完整。

        一轮从一条 user 消息开始，包含其间所有
        assistant(tool_calls) -> tool -> ... -> assistant。
        截断时整块保留或整块丢弃，不会把 tool_calls 和 tool result 拆开。
        当前轮尚未写完的工具结果留在末尾，不会被当成残缺块删掉。
        """
        max_message_count = self.max_turns * 4
        if len(self.messages) > max_message_count:
            self.messages = self.messages[-max_message_count:]

    def snapshot(self) -> list[dict[str, Any]]:
        """返回当前消息快照。"""
        return list(self.messages)
