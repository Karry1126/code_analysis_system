"""LLM judge 校准流程入口：抽样生成标注样本 → 人工标注 → judge 打分 → 算 κ。

校准是「一次性、离线、有人工参与」的动作，不放进 run_eval.py 主流程。

两个子命令：

1. generate —— 从一次评测的轨迹里，按场景抽样作答，生成待人工标注文件。
   每行 {case_id, scenario, question, answer, rubric, human: null, judge: null}。
   human / judge 先留空，避免人工看到 judge 结果被锚定。

   python scripts/run_calibration.py generate \
       --dataset eval/datasets/v1/train.jsonl \
       --traces-dir eval/baselines/baseline_native_v0/traces \
       --out eval/artifacts/calibration_labels.jsonl \
       --per-scenario 10

2. evaluate —— 读回人工填好 human 的文件，对 judge 为空的条目调用 judge 打分，
   计算 Cohen's κ 并输出报告。

   python scripts/run_calibration.py evaluate \
       --labels eval/artifacts/calibration_labels.jsonl \
       --out eval/artifacts/calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from eval.harness.dataset import load_dataset  # noqa: E402
from eval.harness.grader.calibration import load_labels, run_calibration  # noqa: E402
from eval.harness.grader.llm_judge import judge_answer  # noqa: E402


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _case_by_id(cases: list) -> dict[str, Any]:
    return {c.id: c for c in cases}


def _load_answer(traces_dir: str | Path, case_id: str) -> str | None:
    """从 traces/<case_id>/run_*.json 里取一条 completed 的 answer。"""
    case_dir = Path(traces_dir) / case_id
    if not case_dir.is_dir():
        return None
    fallback: str | None = None
    for f in sorted(case_dir.glob("run_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        answer = data.get("answer") or ""
        if fallback is None:
            fallback = answer
        if data.get("status") == "completed" and answer:
            return answer
    return fallback


def cmd_generate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    cases = load_dataset(args.dataset)
    by_id = _case_by_id(cases)

    # 按场景分组
    scenario_cases: dict[str, list[Any]] = defaultdict(list)
    for c in cases:
        scenario_cases[c.scenario].append(c)

    labels: list[dict[str, Any]] = []
    per_scenario = args.per_scenario
    for scenario in sorted(scenario_cases):
        for c in scenario_cases[scenario][:per_scenario]:
            answer = _load_answer(args.traces_dir, c.id)
            if not answer:
                print(f"[calibration] 警告：{c.id} 无可用作答，跳过（请先跑 run_eval 产出轨迹）")
                continue
            labels.append(
                {
                    "case_id": c.id,
                    "scenario": scenario,
                    "question": c.question,
                    "answer": answer,
                    "rubric": c.expected.rubric,
                    "human": None,
                    "judge": None,
                }
            )

    if not labels:
        print("[calibration] 没有生成任何样本，请确认 --traces-dir 已存在且包含轨迹。")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[calibration] 已生成 {len(labels)} 条待标注样本 -> {out}")
    print("[calibration] 下一步：人工填写每行的 human 字段（对 / 部分对 / 错），不要修改 judge 字段。")
    print("[calibration] 标注时建议覆盖 对 / 错 / 边界 三类作答，不要只挑好答的。")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    model_cfg = config.get("model", {})
    judge_model = model_cfg.get("judge_model", "deepseek-v4-pro")
    api_url = model_cfg.get("api_url", "https://api.deepseek.com")

    from code_analysis_system.config import get_api_key

    api_key = get_api_key()

    labels = load_labels(args.labels)
    print(f"[calibration] 读取 {len(labels)} 条标注样本")

    # 对 judge 为空的条目用 judge 打分，避免人工被 judge 锚定
    filled = 0
    for row in labels:
        if row.get("judge") not in (None, ""):
            continue
        out = judge_answer(
            row.get("question", ""),
            row.get("rubric") or "",
            row.get("answer", ""),
            api_key,
            model=judge_model,
            api_url=api_url,
        )
        # judge 输出的"正确性"是三分类标签，直接作为 judge 标签参与 κ
        judge_label = (out.get("raw") or {}).get("正确性") if not out.get("error") else None
        row["judge"] = judge_label
        row["judge_score"] = out.get("score")
        row["judge_text"] = out.get("judge_text")
        row["judge_error"] = out.get("error")
        if judge_label:
            filled += 1
        else:
            print(f"[calibration] 警告：{row.get('case_id')} judge 未产出标签：{out.get('error')}")

    print(f"[calibration] judge 补全 {filled} 条标签")

    # 回写 labels（保留 judge 结果，便于追溯）
    labels_path = Path(args.labels)
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = run_calibration(labels, threshold=args.kappa_threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[calibration] κ={report['kappa']:.3f}，一致率={report['accuracy']:.3f}，"
          f"可用={'是' if report['usable'] else '否'}")
    print(f"[calibration] 分场景 κ：{json.dumps(report['per_scenario'], ensure_ascii=False)}")
    if report["disagreements"]:
        print(f"[calibration] 分歧样例 {len(report['disagreements'])} 条（见 {out}）")
    print(f"[calibration] 报告已写入 {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM judge 校准流程")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "eval" / "config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="抽样生成待人工标注样本")
    p_gen.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "datasets" / "v1" / "train.jsonl"))
    p_gen.add_argument("--traces-dir", required=True)
    p_gen.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "artifacts" / "calibration_labels.jsonl"))
    p_gen.add_argument("--per-scenario", type=int, default=10)
    p_gen.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate", help="judge 打分并算 κ")
    p_eval.add_argument("--labels", required=True)
    p_eval.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "artifacts" / "calibration_report.json"))
    p_eval.add_argument("--kappa-threshold", type=float, default=0.6)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
