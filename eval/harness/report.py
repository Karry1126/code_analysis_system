"""评测报告生成：report.json + report.md + 失败归因。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 状态 → 归因标签
_ATTRIBUTION = {
    "max_loops_exceeded": "循环超限：模型在最大循环数内未收敛（可能检索未收敛、工具结果截断或任务描述不清）",
    "error": "运行异常：执行中抛出异常",
}


def _collect_failures(results_by_case: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for case_id, runs in results_by_case.items():
        for r in runs:
            status = r.get("status")
            if status == "completed":
                continue
            failures.append(
                {
                    "case_id": case_id,
                    "run_id": r.get("run_id"),
                    "status": status,
                    "error": r.get("error"),
                    "attribution": _ATTRIBUTION.get(status, "未知状态"),
                }
            )
    return failures


def build_report(
    *,
    run_id: str,
    runtime_impl: str,
    case_grades: list[dict[str, Any]],
    results_by_case: dict[str, list[dict[str, Any]]],
    aggregate_result: dict[str, Any],
) -> dict[str, Any]:
    """构建报告 dict。"""
    per_case = [
        {
            "case_id": g.get("case_id"),
            "scenario": g.get("scenario"),
            "score_mean": g.get("case_score_mean"),
            "score_std": g.get("case_score_std"),
            "layer_means": g.get("layer_means"),
            "statuses": [r.get("status") for r in results_by_case.get(g.get("case_id"), [])],
        }
        for g in case_grades
    ]

    return {
        "run_id": run_id,
        "runtime_impl": runtime_impl,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_mean": aggregate_result.get("overall_mean"),
        "scenario_means": aggregate_result.get("scenario_means"),
        "engineering": aggregate_result.get("engineering"),
        "per_case": per_case,
        "failures": _collect_failures(results_by_case),
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(c) for c in row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """把报告 dict 渲染成 markdown 文本。"""
    lines: list[str] = []
    lines.append(f"# 评测报告 · {report['run_id']}")
    lines.append("")
    lines.append(f"- 被测实现：`{report['runtime_impl']}`")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append("")

    lines.append("## 总体")
    lines.append("")
    eng = report.get("engineering") or {}
    lines.append(_md_table(
        ["总体均分", "稳定率", "端到端 P50(ms)", "端到端 P95(ms)", "LLM 延迟 P95(ms)", "估算成本(元)"],
        [[
            report.get("overall_mean"),
            eng.get("stability_rate"),
            eng.get("duration_p50_ms"),
            eng.get("duration_p95_ms"),
            eng.get("llm_latency_p95_ms"),
            eng.get("estimated_cost_cny"),
        ]],
    ))
    lines.append("")

    lines.append("## 分场景")
    lines.append("")
    scenario_means = report.get("scenario_means") or {}
    lines.append(_md_table(
        ["场景", "均分"],
        [[s, v] for s, v in scenario_means.items()],
    ))
    lines.append("")

    lines.append("## 逐题")
    lines.append("")
    lines.append(_md_table(
        ["用例", "场景", "均分", "标准差", "状态"],
        [
            [c["case_id"], c["scenario"], c["score_mean"], c["score_std"], ", ".join(c["statuses"])]
            for c in report.get("per_case", [])
        ],
    ))
    lines.append("")

    failures = report.get("failures") or []
    lines.append("## 失败样本")
    lines.append("")
    if not failures:
        lines.append("无失败样本。")
    else:
        lines.append(_md_table(
            ["用例", "状态", "归因", "错误信息"],
            [[f["case_id"], f["status"], f["attribution"], f["error"]] for f in failures],
        ))
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """写 report.json 与 report.md，返回两个文件路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "md": md_path}
