"""LangGraph 1.x 聚合 Runtime。

用 ``create_agent`` + ``AgentMiddleware`` + ``InMemorySaver`` 替代
``framework/runtime.py`` 里的原生 ReAct 循环，同时保持以下契约不变：

- ``RunOutcome`` 六字段（answer/status/steps/metrics/error/exception）与
  ``on_event`` 回调时机与原生 ``run_traced`` 一致；
- ``ToolDefinition`` 仍是 schema 唯一真源，模型看到的 tools 与
  ``ToolRegistry.build_tools_payload()`` 逐字节等价（初始化时用断言兜底）；
- cache 指标（cache_hit_tokens / cache_miss_tokens）固定为 0：langchain-openai
  不把 DeepSeek 的 prompt_cache_* 字段透传到 usage_metadata，且该指标无工程意义。

多轮记忆由 ``InMemorySaver`` 的 ``thread_id`` 提供（替代 ``MessageStore``）；
除 ``messages`` 之外的自定义状态（goal/loop_count/loop_exceeded）每轮由
input 显式重置，与原生 ``run_traced`` 每次 ``create_state`` 的语义一致。
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from typing import Any, Callable, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .coding_runtime import CodingAgentRuntime
from .config import (
    MAX_AGENT_LOOPS,
    MAX_HISTORY_INPUT_TOKENS,
)
from .framework.llm import (
    DEFAULT_API_URL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL_NAME,
)
from .framework.runtime import RunOutcome, compact_tool_result
from .framework.tool_registry import ToolRegistry


class CodingAgentState(TypedDict, total=False):
    """create_agent 的自定义状态，叠加在内置 AgentState 的 messages 之上。

    只保留 goal / loop_count / loop_exceeded 三个非 messages 字段，均用默认
    LastValue（覆盖）语义，且都不是「同一 super-step 内被多个工具并发写」的
    字段：goal 只在 input 时写一次，loop_count / loop_exceeded 只在
    loop_guard（before_model）每步写一次。

    工具执行结果只写回 messages（Annotated[list, add_messages]），天然支持
    并发；不再维护 phase / last_retrieve / last_tool_name / shared_context 等
    顺序依赖的流程控制字段，从而允许模型原生并行调用工具。
    """

    goal: str
    loop_count: int
    loop_exceeded: bool


_probe: CodingAgentRuntime | None = None


def _coding_probe() -> CodingAgentRuntime:
    """惰性取得一个 CodingAgentRuntime 探针，仅用于复用 prompt 文本逻辑。

    create_coding_system_message / build_runtime_message 都是纯文本方法，不依赖
    实例状态；这里传空的 api_key / tool_registry 不会触发环境变量读取或工具注册。
    """
    global _probe
    if _probe is None:
        _probe = CodingAgentRuntime(api_key="", tool_registry=None)
    return _probe


def _runtime_message_content(state: dict[str, Any]) -> str:
    """复用 CodingAgentRuntime.build_runtime_message 的文本，保证 prompt 逐字节一致。"""
    return _coding_probe().build_runtime_message(state)["content"]


def _estimate_message_tokens(msg: Any) -> int:
    """复刻 MessageStore._estimate_message_tokens 的粗估逻辑，作用于 BaseMessage。

    与原生一致用「约 2 字符 ≈ 1 token」粗估，仅用于历史裁剪，不追求绝对精确。
    """
    role = type(msg).__name__
    content = getattr(msg, "content", None)
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)

    tool_calls_text = ""
    for tc in getattr(msg, "tool_calls", None) or []:
        tool_calls_text += json.dumps(tc, ensure_ascii=False, default=str)

    tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
    name = str(getattr(msg, "name", "") or "")
    serialized = "\n".join([role, text, tool_calls_text, tool_call_id, name])
    return max(1, (len(serialized) + 1) // 2) + 4


def _trim_messages(messages: list, max_tokens: int) -> list:
    """按 token 预算裁剪历史，复刻 MessageStore.trim 的原子块语义。

    assistant(tool_calls) 与其紧邻的 ToolMessage 不可拆，从最旧的块开始丢弃，
    直到剩余消息估测 token 数不超过 max_tokens。
    """
    blocks: list[list] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            block = [msg]
            i += 1
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                block.append(messages[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1

    start = 0
    while start < len(blocks) - 1:
        remaining = [m for block in blocks[start:] for m in block]
        if sum(_estimate_message_tokens(m) for m in remaining) <= max_tokens:
            break
        start += 1

    return [m for block in blocks[start:] for m in block]


def to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """把 ToolRegistry 里的 ToolDefinition 转成 StructuredTool 列表。

    关键点：``args_schema=td.parameters`` 直接传原生 JSON schema，``infer_schema=False``
    禁止 LangChain 从函数签名重新推断，从而保证 ``convert_to_openai_tool`` 的输出与
    ``registry.build_tools_payload()`` 逐字节等价。handler 内部完成
    ``compact_tool_result`` 并返回 ``(json_str, compacted_dict)`` 二元组：
    - ToolMessage.content = json.dumps(compacted)（与原生 message_store 写入一致）
    - ToolMessage.artifact = compacted dict（供 steps 事件与 state 更新复用）
    """
    tools: list[StructuredTool] = []
    for td in registry._tools.values():  # noqa: SLF001 - 同项目内唯一真源
        name = td.name
        description = td.description
        # deepcopy 防 convert_to_openai_tool 内部的原地 pop(title/description) 污染真源
        parameters = copy.deepcopy(td.parameters)
        handler = td.handler

        def make_handler(
            func: Callable[..., dict[str, Any]],
            bound_name: str,
        ) -> Callable[..., tuple]:
            def _run(**kwargs: Any) -> tuple[str, dict[str, Any]]:
                try:
                    result = func(**kwargs)
                except Exception as exc:  # noqa: BLE001 - 与原生 execute_tool_call 一致，转成错误 dict
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                compacted = compact_tool_result(result, tool_name=bound_name)
                return (json.dumps(compacted, ensure_ascii=False), compacted)

            return _run

        tools.append(
            StructuredTool.from_function(
                func=make_handler(handler, name),
                name=name,
                description=description,
                args_schema=parameters,
                infer_schema=False,
                response_format="content_and_artifact",
            )
        )
    return tools


def assert_schema_equivalence(registry: ToolRegistry) -> None:
    """初始化时的兜底断言：LangChain tools 与原生 payload 逐字节等价。"""
    native_payload = registry.build_tools_payload()
    langchain_payload = [convert_to_openai_tool(t) for t in to_langchain_tools(registry)]
    if langchain_payload != native_payload:
        raise AssertionError(
            "tools schema 漂移：convert_to_openai_tool 输出与 build_tools_payload 不一致。"
        )


class _Collector:
    """收集 steps 事件与 metrics，并实时回调 on_event（与原生 run_traced 对齐）。"""

    def __init__(self, on_event: Callable[[dict[str, Any]], None] | None) -> None:
        self.steps: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {
            "llm_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "llm_latency_ms": 0.0,
            "tool_latency_ms": 0.0,
        }
        self.on_event = on_event

    def emit(self, event: dict[str, Any]) -> None:
        self.steps.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def on_llm_call(
        self, state: dict[str, Any], response: Any, latency_ms: float
    ) -> None:
        """从 wrap_model_call 的 ModelResponse 里采集 llm_call 事件与 token 指标。"""
        ai_msg: AIMessage | None = response.result[0] if response.result else None
        usage = getattr(ai_msg, "usage_metadata", None) if ai_msg is not None else None
        input_tokens = (usage or {}).get("input_tokens")
        output_tokens = (usage or {}).get("output_tokens")
        tool_calls = (
            [
                {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}
                for tc in (ai_msg.tool_calls or [])
            ]
            if ai_msg is not None
            else []
        )

        self.metrics["llm_calls"] += 1
        self.metrics["llm_latency_ms"] += latency_ms
        if input_tokens is not None:
            self.metrics["input_tokens"] += input_tokens
        if output_tokens is not None:
            self.metrics["output_tokens"] += output_tokens

        self.emit(
            {
                "type": "llm_call",
                "loop": state.get("loop_count"),
                "model": DEFAULT_MODEL_NAME,
                # langchain-openai 把 finish_reason 放进 generation_info 而非 message，
                # response_metadata 里通常没有，故尽力读取，拿不到为 None。
                "finish_reason": (ai_msg.response_metadata or {}).get("finish_reason")
                if ai_msg is not None
                else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
                "latency_ms": latency_ms,
                "tool_calls": tool_calls,
            }
        )

    def on_tool_call(
        self,
        state: dict[str, Any],
        name: str,
        arguments: str,
        compacted: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """采集 tool_call 事件与工具延迟指标。"""
        self.metrics["tool_calls"] += 1
        self.metrics["tool_latency_ms"] += latency_ms
        self.emit(
            {
                "type": "tool_call",
                "loop": state.get("loop_count"),
                "name": name,
                "arguments": arguments,
                "ok": compacted.get("ok"),
                "truncated": compacted.get("truncated", False),
                "error": compacted.get("error"),
                "latency_ms": latency_ms,
                "result": compacted,
            }
        )


def _build_loop_guard_middleware():
    """复刻 MAX_AGENT_LOOPS 语义：第 10 轮工具照常执行，第 11 次进模型前跳 end。"""

    def loop_guard(state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        count = state.get("loop_count", 0) + 1
        if count > MAX_AGENT_LOOPS:
            return {"loop_count": count, "loop_exceeded": True, "jump_to": "end"}
        return {"loop_count": count}

    return before_model(loop_guard, can_jump_to=["end"])


def _build_inject_runtime_middleware(collector: _Collector):
    """把运行时调度消息注入当次模型调用，不污染持久 history。"""

    def inject_runtime(request: Any, handler: Callable) -> Any:
        started = time.perf_counter()
        # 先按 token 预算裁剪历史，再注入当次运行时调度消息（不污染持久 history）。
        trimmed = _trim_messages(request.messages, MAX_HISTORY_INPUT_TOKENS)
        messages = [
            *trimmed,
            HumanMessage(content=_runtime_message_content(request.state)),
        ]
        response = handler(request.override(messages=messages))
        latency_ms = (time.perf_counter() - started) * 1000

        # 复刻原生 call_llm 的 finish_reason 截断校验：langchain-openai 通常把
        # finish_reason 放进 generation_info 而非 message，这里尽力读取，拿不到则跳过。
        ai_msg = response.result[0] if response.result else None
        finish_reason = (
            (ai_msg.response_metadata or {}).get("finish_reason")
            if ai_msg is not None
            else None
        )
        if finish_reason == "length":
            raise RuntimeError(
                "模型输出被 max_tokens 截断（finish_reason=length），请拆分任务或减小输出规模。"
            )
        if finish_reason == "content_filter":
            raise RuntimeError(
                "模型输出被内容安全策略拦截（finish_reason=content_filter）。"
            )

        collector.on_llm_call(request.state, response, latency_ms)
        return response

    return wrap_model_call(inject_runtime)


def _build_track_tool_middleware(collector: _Collector):
    """执行工具后记录事件，并把 ToolMessage 追加进 messages（不写标量 state）。"""

    def track_tool(request: Any, handler: Callable) -> Command:
        started = time.perf_counter()
        result = handler(request)  # ToolMessage，artifact = compacted dict
        latency_ms = (time.perf_counter() - started) * 1000

        name = request.tool_call["name"]
        arguments = json.dumps(request.tool_call["args"], ensure_ascii=False)
        compacted = getattr(result, "artifact", None)
        if not isinstance(compacted, dict):
            compacted = {"ok": False, "error": str(getattr(result, "content", ""))}

        collector.on_tool_call(request.state, name, arguments, compacted, latency_ms)

        # 只把 ToolMessage 追加进 messages（add_messages 天然并发安全），
        # 不写任何标量 state，从而支持模型并行调用多个工具。
        return Command(update={"messages": [result]})

    return wrap_tool_call(track_tool)


class LangGraphAgentRuntime:
    """create_agent 聚合 Runtime，暴露 run() 返回 RunOutcome，契约与原生一致。"""

    def __init__(
        self,
        api_key: str,
        tool_registry: ToolRegistry,
        *,
        disabled_middleware: set[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.tool_registry = tool_registry
        self.disabled_middleware = disabled_middleware or set()
        # checkpointer 是实例级的：CLI 复用同一实例 + 同一 thread_id 维持多轮记忆；
        # harness 每次 run_agent 新建实例（全新 checkpointer）等价原生每次全新 MessageStore。
        self._checkpointer = InMemorySaver()

        # 初始化即校验 schema 等价，把「约定」变成运行时保证。
        assert_schema_equivalence(self.tool_registry)

    def _initial_state(self, goal: str) -> dict[str, Any]:
        """每次 run 的 input：messages 追加，自定义字段覆盖重置（等价 create_state）。"""
        return {
            "messages": [HumanMessage(content=goal)],
            "goal": goal,
            "loop_count": 0,
            "loop_exceeded": False,
        }

    def _build_graph(self, collector: _Collector):
        model = ChatOpenAI(
            model=DEFAULT_MODEL_NAME,
            api_key=self.api_key,
            base_url=DEFAULT_API_URL,
            max_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
            temperature=0.1,
            extra_body={"thinking": {"type": "disabled"}},
        )

        middlewares: list = []
        if "loop_guard" not in self.disabled_middleware:
            middlewares.append(_build_loop_guard_middleware())
        if "inject_runtime" not in self.disabled_middleware:
            middlewares.append(_build_inject_runtime_middleware(collector))
        if "track_tool" not in self.disabled_middleware:
            middlewares.append(_build_track_tool_middleware(collector))

        return create_agent(
            model=model,
            tools=to_langchain_tools(self.tool_registry),
            system_prompt=_coding_probe().create_coding_system_message()["content"],
            middleware=middlewares,
            state_schema=CodingAgentState,
            checkpointer=self._checkpointer,
        )

    def run(
        self,
        goal: str,
        thread_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunOutcome:
        """执行一轮任务，返回 RunOutcome（字段与原生 run_traced 一致）。"""
        if thread_id is None:
            thread_id = uuid.uuid4().hex

        collector = _Collector(on_event=on_event)
        graph = self._build_graph(collector)

        try:
            final_state = graph.invoke(
                self._initial_state(goal),
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:  # noqa: BLE001 - 与原生一致，异常归因到 status=error
            return RunOutcome(
                answer="",
                status="error",
                steps=collector.steps,
                metrics=collector.metrics,
                error=f"{type(exc).__name__}: {exc}",
                exception=exc,
            )

        if final_state.get("loop_exceeded"):
            return RunOutcome(
                answer="",
                status="max_loops_exceeded",
                steps=collector.steps,
                metrics=collector.metrics,
                error="超过最大循环次数，任务仍未完成。可以把任务描述得更具体一点再重试。",
            )

        last_msg = final_state.get("messages")[-1] if final_state.get("messages") else None
        answer = last_msg.content if last_msg is not None else ""
        answer = answer or "任务已处理完成。"
        collector.emit(
            {"type": "final_answer", "loop": final_state.get("loop_count"), "content": answer}
        )
        return RunOutcome(
            answer=answer,
            status="completed",
            steps=collector.steps,
            metrics=collector.metrics,
        )
