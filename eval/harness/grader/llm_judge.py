"""LLM-as-a-judge：对开放答案的 task 层做语义判分。

judge 使用与被测模型不同的模型（默认 deepseek-chat，被测为 deepseek-flash），
避免自评乐观偏差。输出结构化 JSON，score 0-5 归一化到 0-1。

judge_fn 签名与 grader.__init__ 约定一致：
    judge_fn(case, result, layer) -> {"score": float, "raw": dict, "judge_text": str}
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from ..schema import Case, GradingLayer

DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEFAULT_API_URL = "https://api.deepseek.com"
MAX_RETRIES = 2

JUDGE_PROMPT = """你是代码分析助手的评测官。请依据【题目】、【评分要点】和【作答】打分。

只输出一个 JSON 对象，不要输出任何其他文字或代码块：
{{"score": <0-5 的整数>, "正确性": "<对|部分对|错>", "证据充分": "<是|否>", "幻觉": "<是|否>", "理由": "<一句话>"}}

- score 0-5：5 完全正确且证据充分，3 大致正确但有遗漏或小错，1 基本错误，0 完全错误或未作答。
- 幻觉：作答是否编造了不存在的模块、符号、文件、字段或结论。

【题目】
{question}

【评分要点】
{rubric}

【作答】
{answer}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """从 judge 输出中提取 JSON 对象（容忍 ```json``` 包裹或前后噪声）。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def judge_answer(
    question: str,
    rubric: str,
    answer: str,
    api_key: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """对单次作答调用 judge，返回 {score, raw, judge_text, error}。"""
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric or "（未提供评分要点，按是否准确、有据、无幻觉综合判断）", answer=answer or "（空）")
    client = OpenAI(api_key=api_key, base_url=api_url)

    last_error: str | None = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
            finish = getattr(resp.choices[0], "finish_reason", None)
            raw = _extract_json(content)
            if "score" not in raw:
                print(f"[judge raw content] {content!r}")
                print(f"[judge raw reasoning_content] {reasoning[:500]!r}")
                print(f"[judge finish_reason] {finish!r}")
                last_error = (
                    f"judge 未输出合法 score：content={content!r} "
                    f"reasoning_content={reasoning[:300]!r} finish_reason={finish!r}"
                )
                continue
            score = raw.get("score")
            try:
                score = float(score)
            except (TypeError, ValueError):
                last_error = f"judge score 非法：{score}"
                continue
            return {
                "score": max(0.0, min(5.0, score)) / 5.0,
                "raw": raw,
                "judge_text": content,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "score": None,
        "raw": {},
        "judge_text": "",
        "error": f"judge 调用失败（重试 {max_retries} 次后放弃）：{last_error}",
    }


def build_judge_fn(
    api_key: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_url: str = DEFAULT_API_URL,
):
    """构造 judge_fn，供 grader.grade_case 使用。"""

    def judge_fn(case: Case, result: dict[str, Any], layer: GradingLayer) -> dict[str, Any]:
        rubric = layer.rubric or case.expected.rubric or ""
        answer = result.get("answer") or ""
        out = judge_answer(
            case.question,
            rubric,
            answer,
            api_key,
            model=model,
            api_url=api_url,
        )
        return {"score": out["score"], "raw": out["raw"], "judge_text": out["judge_text"], "error": out["error"]}

    return judge_fn
