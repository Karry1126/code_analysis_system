"""评测 Runner：驱动 Agent 多次运行、落盘轨迹、按状态分类。

对单题跑 n 次取统计（非确定性 LLM Agent 必须多次运行），每次运行互不污染
（agent_api.run_agent 内部每次使用全新 MessageStore）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from code_analysis_system.agent_api import run_agent

from .schema import Case, SCENARIO_MODIFY
from .trace import Trace

DEFAULT_N_RUNS = 3


def resolve_n_runs(case: Case, default: int = DEFAULT_N_RUNS) -> int:
    """确定单题运行次数：modify 场景强制 1（有文件副作用），否则用 case.n_runs 或默认。"""
    if case.scenario == SCENARIO_MODIFY:
        return 1
    if case.n_runs is not None:
        return case.n_runs
    return default


def run_case(
    case: Case,
    *,
    n_runs: int | None = None,
    traces_dir: str | Path | None = None,
    sandbox=None,
    runtime_impl: str = "native",
) -> list[dict[str, Any]]:
    """对单题跑 n 次，返回结构化结果列表，并可选落盘轨迹。

    结果结构与 agent_api.run_agent 单次一致：
    run_id / question / answer / status / error / steps / metrics。

    modify 场景在 sandbox 就绪前拒绝执行（S7 实现后传入 sandbox）。
    """
    if case.scenario == SCENARIO_MODIFY and sandbox is None:
        raise NotImplementedError(
            f"[{case.id}] modify 场景需要工作区沙箱隔离（sandbox），在 S7 实现后传入。"
        )

    n = n_runs or resolve_n_runs(case)
    results: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        started = time.perf_counter()
        single = run_agent(case.question, n_runs=1, runtime_impl=runtime_impl)[0]
        duration_ms = (time.perf_counter() - started) * 1000
        single["duration_ms"] = duration_ms
        results.append(single)
        if traces_dir is not None:
            Trace.from_agent_result(case.id, single, duration_ms).save(
                Path(traces_dir) / case.id / f"run_{i}.json"
            )
    return results


def run_all(
    cases: list[Case],
    *,
    n_runs: int | None = None,
    traces_dir: str | Path | None = None,
    sandbox=None,
    runtime_impl: str = "native",
) -> dict[str, list[dict[str, Any]]]:
    """遍历全部用例，返回 {case_id: [run results]}。"""
    out: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        out[case.id] = run_case(
            case,
            n_runs=n_runs,
            traces_dir=traces_dir,
            sandbox=sandbox,
            runtime_impl=runtime_impl,
        )
    return out
