"""分层判分组合器：task / evidence / trace 三层加权成 case_score。

- 规则层：调用 rules.layer_score 做确定性判分。
- LLM 层：调用 judge_fn（S5 实现）做语义判分；judge_fn 未提供时标记 pending。
- 多层多次运行：对每次 run 分别判分，再聚合 mean / std。
"""

from __future__ import annotations

from typing import Any, Callable

from ..schema import Case, GradingLayer
from . import rules

# judge_fn(case, result, layer) -> {"score": float, "raw": dict}
JudgeFn = Callable[[Case, dict[str, Any], GradingLayer], dict[str, Any]]


def grade_result(
    case: Case,
    result: dict[str, Any],
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """对单次 run 判分，返回 {case_score, layers, status}。"""
    grading = case.grading
    layers: dict[str, Any] = {}

    for layer_name in ("task", "evidence", "trace"):
        layer = getattr(grading, layer_name)
        if layer.type == "rules":
            layers[layer_name] = rules.layer_score(layer.checks, case, result)
            layers[layer_name]["type"] = "rules"
        elif layer.type == "llm_judge":
            if judge_fn is None:
                layers[layer_name] = {"score": None, "type": "llm_judge", "pending": True}
            else:
                layers[layer_name] = judge_fn(case, result, layer)
                layers[layer_name]["type"] = "llm_judge"
                if layers[layer_name].get("score") is None:
                    print(f"[judge] {case.id} {layer_name} 层判分失败：{layers[layer_name].get('error')}")

    total_weight = 0.0
    weighted = 0.0
    for layer_name in ("task", "evidence", "trace"):
        layer = getattr(grading, layer_name)
        score = layers[layer_name].get("score")
        if score is not None:
            weighted += float(score) * layer.weight
            total_weight += layer.weight

    case_score = weighted / total_weight if total_weight else None
    return {
        "case_score": case_score,
        "layers": layers,
        "status": result.get("status"),
    }


def grade_case(
    case: Case,
    results: list[dict[str, Any]],
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """对单题多次运行判分并聚合。

    返回 {case_id, n_runs, case_score_mean, case_score_std, per_run, layer_means}。
    """
    per_run = [grade_result(case, r, judge_fn=judge_fn) for r in results]
    scores = [r["case_score"] for r in per_run if r["case_score"] is not None]

    # 分层聚合：task / evidence / trace 各自均值
    layer_means: dict[str, Any] = {}
    for layer_name in ("task", "evidence", "trace"):
        values = [
            r["layers"][layer_name]["score"]
            for r in per_run
            if r["layers"][layer_name].get("score") is not None
        ]
        layer_means[layer_name] = sum(values) / len(values) if values else None

    mean = sum(scores) / len(scores) if scores else None
    std = None
    if scores and len(scores) > 1:
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5

    return {
        "case_id": case.id,
        "scenario": case.scenario,
        "n_runs": len(results),
        "case_score_mean": mean,
        "case_score_std": std,
        "layer_means": layer_means,
        "per_run": per_run,
    }
