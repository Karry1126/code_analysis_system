"""指标聚合：分层分数 + 工程指标（延迟/成本/稳定率）。

输入：
- case_grades：grader.grade_case 的结果列表（每 case 一个）
- results_by_case：runner.run_all 的结果，{case_id: [run dict]}

输出：
- 分数：每 case 均值、分场景均值、总体均值
- 工程：P50/P95 延迟、token 均值、成本、稳定率（completed 占比）
"""

from __future__ import annotations

from typing import Any


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """线性插值百分位。"""
    if not sorted_values:
        return None
    values = sorted(sorted_values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def aggregate(
    case_grades: list[dict[str, Any]],
    results_by_case: dict[str, list[dict[str, Any]]],
    *,
    price_per_1k_input: float = 0.0,
    price_per_1k_output: float = 0.0,
) -> dict[str, Any]:
    """聚合分数与工程指标。"""
    # ---- 分数聚合 ----
    scenario_scores: dict[str, list[float]] = {}
    overall_scores: list[float] = []
    for g in case_grades:
        score = g.get("case_score_mean")
        if score is None:
            continue
        overall_scores.append(score)
        scenario_scores.setdefault(g.get("scenario", "unknown"), []).append(score)

    scenario_means = {
        s: sum(v) / len(v) for s, v in scenario_scores.items() if v
    }
    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else None

    # ---- 工程指标 ----
    durations: list[float] = []
    llm_latencies: list[float] = []
    input_tokens_total = 0
    output_tokens_total = 0
    llm_calls_total = 0
    tool_calls_total = 0
    stable_runs = 0
    total_runs = 0

    for runs in results_by_case.values():
        for r in runs:
            total_runs += 1
            if r.get("status") == "completed":
                stable_runs += 1
            d = r.get("duration_ms")
            if isinstance(d, (int, float)):
                durations.append(float(d))
            m = r.get("metrics") or {}
            input_tokens_total += int(m.get("input_tokens", 0) or 0)
            output_tokens_total += int(m.get("output_tokens", 0) or 0)
            llm_calls_total += int(m.get("llm_calls", 0) or 0)
            tool_calls_total += int(m.get("tool_calls", 0) or 0)
            llm_latencies.append(float(m.get("llm_latency_ms", 0.0) or 0.0))

    input_cost = input_tokens_total / 1000 * price_per_1k_input
    output_cost = output_tokens_total / 1000 * price_per_1k_output

    engineering = {
        "total_runs": total_runs,
        "stable_runs": stable_runs,
        "stability_rate": (stable_runs / total_runs) if total_runs else None,
        "duration_p50_ms": _percentile(durations, 0.50),
        "duration_p95_ms": _percentile(durations, 0.95),
        "llm_latency_p50_ms": _percentile(llm_latencies, 0.50),
        "llm_latency_p95_ms": _percentile(llm_latencies, 0.95),
        "input_tokens_total": input_tokens_total,
        "output_tokens_total": output_tokens_total,
        "llm_calls_total": llm_calls_total,
        "tool_calls_total": tool_calls_total,
        "avg_llm_calls_per_run": (llm_calls_total / total_runs) if total_runs else None,
        "avg_tool_calls_per_run": (tool_calls_total / total_runs) if total_runs else None,
        "estimated_cost_cny": input_cost + output_cost,
    }

    return {
        "n_cases": len(case_grades),
        "overall_mean": overall_mean,
        "scenario_means": scenario_means,
        "engineering": engineering,
    }
