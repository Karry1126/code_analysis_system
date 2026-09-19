from typing import Any
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from icecream import ic

DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-flash"
DEFAULT_MAX_COMPLETION_TOKENS = 3000


def call_llm(
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_url: str = DEFAULT_API_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> OpenAI.ChatCompletionMessageParam:
    """
    调用 DeepSeek Chat API，

    这一层只做“模型通信”，不做业务判断。
    这样 runtime、tool registry、tool handlers 都能保持边界清楚。
    """
    # ic()
    # ic(messages)
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )
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

    if not response.choices:
        raise RuntimeError("模型没有返回 choices")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("模型输出被 max_tokens 截断")
    if choice.finish_reason == "content_filter":
        raise RuntimeError("模型输出被内容安全策略拦截")
    if choice.message is None:
        raise RuntimeError("模型返回的 message 为空")

    usage = response.usage
    if usage:
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        print(
            f"[tokens] 输入命中={cache_hit} "
            f"输入未命中={cache_miss} "
            f"输出={usage.completion_tokens}"
        )

    return choice.message