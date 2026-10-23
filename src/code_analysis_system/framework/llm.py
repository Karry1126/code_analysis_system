import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage

DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-flash"
DEFAULT_MAX_COMPLETION_TOKENS = 3000


@dataclass
class LLMUsage:
    """单次模型调用的 token 用量。"""

    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.prompt_cache_hit_tokens + self.prompt_cache_miss_tokens


@dataclass
class LLMCall:
    """单次模型调用的完整结果：消息 + 用量 + 延迟 + 元信息。"""

    message: ChatCompletionMessage
    usage: LLMUsage | None
    latency_ms: float
    model: str
    finish_reason: str


def call_llm(
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_url: str = DEFAULT_API_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> LLMCall:
    """
    调用 DeepSeek Chat API，

    这一层只做“模型通信”，不做业务判断。
    这样 runtime、tool registry、tool handlers 都能保持边界清楚。

    返回值从 ChatCompletionMessage 升级为 LLMCall，
    附带 token 用量与延迟，供评测 harness 采集成本/延迟指标。
    """
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_completion_tokens,
            temperature=0.1,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except APITimeoutError as exc:
        raise RuntimeError("调用模型超时") from exc
    except APIConnectionError as exc:
        raise RuntimeError("无法连接到模型服务") from exc
    except RateLimitError as exc:
        raise RuntimeError("模型请求触发限流") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"模型接口返回错误：status={exc.status_code}, body={exc.body}"
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000

    if not response.choices:
        raise RuntimeError("模型没有返回 choices")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("模型输出被 max_tokens 截断")
    if choice.finish_reason == "content_filter":
        raise RuntimeError("模型输出被内容安全策略拦截")
    if choice.message is None:
        raise RuntimeError("模型返回的 message 为空")

    usage = None
    if response.usage:
        usage = LLMUsage(
            prompt_cache_hit_tokens=getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0,
            prompt_cache_miss_tokens=getattr(response.usage, "prompt_cache_miss_tokens", 0) or 0,
            completion_tokens=response.usage.completion_tokens or 0,
            total_tokens=response.usage.total_tokens or 0,
        )

    return LLMCall(
        message=choice.message,
        usage=usage,
        latency_ms=latency_ms,
        model=model_name,
        finish_reason=choice.finish_reason or "",
    )
