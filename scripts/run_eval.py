"""评测主入口：串联 用例加载 → 快照校验 → 驱动执行 → 判分 → 聚合 → 报告 → A/B 对比。

用法示例：
    # 首次：创建快照并跑全量，产出基线
    python scripts/run_eval.py --run-id baseline_native_v0 --create-snapshot

    # 后续：校验快照一致性 + 对比已有基线
    python scripts/run_eval.py --run-id langchain_v0 --runtime-impl langchain \
        --snapshot eval/baselines/baseline_native_v0/snapshot.json \
        --compare eval/baselines/baseline_native_v0/report.json

    # 调试：只跑前 3 题、跳过 judge
    python scripts/run_eval.py --limit 3 --no-judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from eval.harness.dataset import load_dataset, validate  # noqa: E402
from eval.harness.grader import grade_case  # noqa: E402
from eval.harness.grader.llm_judge import build_judge_fn  # noqa: E402
from eval.harness.metrics import aggregate  # noqa: E402
from eval.harness.report import build_report, write_report  # noqa: E402
from eval.harness.runner import run_case  # noqa: E402
from eval.harness.sandbox import Sandbox, run_modify_isolated  # noqa: E402
from eval.harness.schema import SCENARIO_MODIFY  # noqa: E402
from eval.harness.trace import Trace  # noqa: E402
from eval.harness.snapshot import capture_snapshot, verify_snapshot  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(project_root: Path, config: dict, key: str, default: str) -> Path:
    raw = config.get("paths", {}).get(key, default)
    return (project_root / raw).resolve()


def compare_baseline(report: dict, baseline_path: str | Path) -> dict:
    """对比当前报告与基线报告，返回 A/B diff。"""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return {"error": f"基线报告不存在：{baseline_path}"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    diff: dict[str, Any] = {}
    cur_mean = report.get("overall_mean")
    base_mean = baseline.get("overall_mean")
    if cur_mean is not None and base_mean is not None:
        diff["overall_mean"] = {
            "baseline": base_mean,
            "current": cur_mean,
            "delta": cur_mean - base_mean,
        }

    scenario_diff: dict[str, Any] = {}
    base_sc = baseline.get("scenario_means") or {}
    cur_sc = report.get("scenario_means") or {}
    for s in sorted(set(base_sc) | set(cur_sc)):
        b = base_sc.get(s)
        c = cur_sc.get(s)
        scenario_diff[s] = {
            "baseline": b,
            "current": c,
            "delta": (c - b) if (b is not None and c is not None) else None,
        }
    diff["scenario_means"] = scenario_diff

    eng_diff: dict[str, Any] = {}
    base_eng = baseline.get("engineering") or {}
    cur_eng = report.get("engineering") or {}
    for k in ("stability_rate", "duration_p95_ms", "estimated_cost_cny"):
        b = base_eng.get(k)
        c = cur_eng.get(k)
        eng_diff[k] = {"baseline": b, "current": c}
    diff["engineering"] = eng_diff

    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 harness 主入口")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "eval" / "config.yaml"))
    parser.add_argument("--run-id", default="baseline_native_v0")
    parser.add_argument("--runtime-impl", default=None)
    parser.add_argument("--dataset", default="train.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--create-snapshot", action="store_true")
    parser.add_argument("--snapshot", default=None, help="已有快照路径（校验模式）")
    parser.add_argument("--compare", default=None, help="基线 report.json 路径（A/B 对比）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM judge")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime_impl = args.runtime_impl or config.get("runtime_impl", "native")
    project_root = PROJECT_ROOT

    dataset_dir = _resolve(project_root, config, "dataset_dir", "eval/datasets/v1")
    data_dir = _resolve(project_root, config, "data_dir", "data")
    harness_dir = _resolve(project_root, config, "harness_dir", "eval/harness")
    baselines_dir = _resolve(project_root, config, "baselines_dir", "eval/baselines")
    out_dir = Path(args.out_dir) if args.out_dir else (baselines_dir / args.run_id)

    dataset_path = dataset_dir / args.dataset
    holdout_path = dataset_dir / "holdout.jsonl"

    # 被测仓库路径：直接用 code_analysis_system.config 解析结果
    from code_analysis_system.config import COMPANY_CODE_REPO_PATH, get_api_key

    print(f"[eval] 数据集：{dataset_path}")
    print(f"[eval] 被测仓库：{COMPANY_CODE_REPO_PATH}")
    print(f"[eval] runtime_impl：{runtime_impl}")

    # 1. 加载并校验数据集
    cases = load_dataset(dataset_path)
    validate(cases)
    if args.limit:
        cases = cases[: args.limit]
    print(f"[eval] 用例数：{len(cases)}")

    # 2. 快照
    dataset_paths = {"train": dataset_path, "holdout": holdout_path}
    snapshot_path = out_dir / "snapshot.json"
    if args.snapshot:
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
        issues = verify_snapshot(
            snapshot,
            dataset_paths=dataset_paths,
            company_repo_path=COMPANY_CODE_REPO_PATH,
            data_dir=data_dir,
            harness_dir=harness_dir,
        )
        if issues:
            print("[eval] 快照校验不一致，终止：")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("[eval] 快照校验通过")
    elif args.create_snapshot:
        snapshot = capture_snapshot(
            args.run_id,
            runtime_impl,
            dataset_paths=dataset_paths,
            company_repo_path=COMPANY_CODE_REPO_PATH,
            data_dir=data_dir,
            harness_dir=harness_dir,
            extra_config={
                "model": config.get("model"),
                "run": config.get("run"),
            },
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] 快照已写入 {snapshot_path}")

    # 3. 驱动执行
    traces_dir = out_dir / "traces"
    results_by_case: dict[str, list[dict]] = {}
    api_key = get_api_key()
    n_runs = config.get("run", {}).get("n_runs")
    for case in cases:
        if case.scenario == SCENARIO_MODIFY:
            with Sandbox(COMPANY_CODE_REPO_PATH) as sandbox:
                sandbox.prepare()
                started = time.perf_counter()
                result = run_modify_isolated(case.question, sandbox.repo_copy)
                duration_ms = (time.perf_counter() - started) * 1000
                result["duration_ms"] = duration_ms
                result["diff"] = sandbox.compute_diff(case.expected.target_file or "")
                Trace.from_agent_result(case.id, result, duration_ms).save(
                    traces_dir / case.id / "run_1.json"
                )
                results_by_case[case.id] = [result]
                sandbox.teardown()
        else:
            results_by_case[case.id] = run_case(case, n_runs=n_runs, traces_dir=traces_dir)
        statuses = [r.get("status") for r in results_by_case[case.id]]
        print(f"[eval] {case.id} ({case.scenario}) -> {statuses}")

    # 4. 判分
    judge_fn = None if args.no_judge else build_judge_fn(api_key, model=config["model"]["judge_model"])
    case_grades = [
        grade_case(case, results_by_case[case.id], judge_fn=judge_fn)
        for case in cases
    ]

    # 5. 聚合
    model_cfg = config.get("model", {})
    agg = aggregate(
        case_grades,
        results_by_case,
        price_per_1k_input=float(model_cfg.get("price_per_1k_input", 0.0)),
        price_per_1k_output=float(model_cfg.get("price_per_1k_output", 0.0)),
    )

    # 6. 报告
    report = build_report(
        run_id=args.run_id,
        runtime_impl=runtime_impl,
        case_grades=case_grades,
        results_by_case=results_by_case,
        aggregate_result=agg,
    )
    write_report(report, out_dir)
    print(f"[eval] 报告已写入 {out_dir}")

    # 7. A/B 对比
    if args.compare:
        diff = compare_baseline(report, args.compare)
        diff_path = out_dir / "diff.json"
        diff_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] A/B diff 已写入 {diff_path}")
        print(json.dumps(diff, ensure_ascii=False, indent=2))

    print(f"[eval] 总体均分：{agg['overall_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
