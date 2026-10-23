"""评测数据集加载、校验与拆分。

- load_dataset(path)：读取 JSONL，逐条 parse 成 Case，缺省 grading 用场景默认值。
- validate(cases)：场景级必填校验，非法即 raise（fail-fast，避免坏数据进基线）。
- split_by_scenario(cases)：按场景分组，返回 dict[str, list[Case]]。
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import (
    Case,
    Expected,
    Grading,
    GradingLayer,
    SCENARIOS,
    SCENARIO_BUG,
    SCENARIO_CONSULT,
    SCENARIO_MODIFY,
    SCENARIO_RETRIEVAL,
)

# 每个场景的默认分层判分：task / evidence / trace 的权重和为 1。
DEFAULT_GRADING: dict[str, Grading] = {
    SCENARIO_CONSULT: Grading(
        task=GradingLayer(type="llm_judge", weight=0.5),
        evidence=GradingLayer(
            type="rules",
            checks=["modules_covered", "symbols_covered", "no_hallucination"],
            weight=0.3,
        ),
        trace=GradingLayer(
            type="rules",
            checks=["used_index_tools", "no_illegal_search", "stable"],
            weight=0.2,
        ),
    ),
    SCENARIO_RETRIEVAL: Grading(
        task=GradingLayer(
            type="rules",
            checks=["recall", "precision"],
            weight=0.5,
        ),
        evidence=GradingLayer(
            type="rules",
            checks=["no_hallucination"],
            weight=0.2,
        ),
        trace=GradingLayer(
            type="rules",
            checks=["used_index_tools", "no_illegal_search", "stable"],
            weight=0.3,
        ),
    ),
    SCENARIO_BUG: Grading(
        task=GradingLayer(type="llm_judge", weight=0.5),
        evidence=GradingLayer(
            type="rules",
            checks=["root_cause_located", "no_hallucination"],
            weight=0.3,
        ),
        trace=GradingLayer(
            type="rules",
            checks=["read_target_file", "no_illegal_search", "stable"],
            weight=0.2,
        ),
    ),
    SCENARIO_MODIFY: Grading(
        task=GradingLayer(type="llm_judge", weight=0.5),
        evidence=GradingLayer(
            type="rules",
            checks=["key_change_present", "only_target_file_modified"],
            weight=0.3,
        ),
        trace=GradingLayer(
            type="rules",
            checks=["read_before_write", "no_claimed_without_result", "stable"],
            weight=0.2,
        ),
    ),
}


def _parse_expected(raw: dict | None) -> Expected:
    raw = raw or {}
    return Expected(**{k: v for k, v in raw.items() if k in Expected.__dataclass_fields__})


def _parse_grading_layer(raw: dict | None) -> GradingLayer:
    raw = raw or {}
    return GradingLayer(
        type=raw.get("type", "rules"),
        checks=raw.get("checks"),
        rubric=raw.get("rubric"),
        weight=float(raw.get("weight", 1.0)),
    )


def _parse_grading(raw: dict | None) -> Grading | None:
    if not raw:
        return None
    return Grading(
        task=_parse_grading_layer(raw.get("task")),
        evidence=_parse_grading_layer(raw.get("evidence")),
        trace=_parse_grading_layer(raw.get("trace")),
    )


def parse_case(raw: dict) -> Case:
    """把一条 JSON 对象解析成 Case。"""
    return Case(
        id=str(raw["id"]),
        scenario=str(raw["scenario"]),
        question=str(raw["question"]),
        tags=list(raw.get("tags") or []),
        allow_modify=bool(raw.get("allow_modify", False)),
        n_runs=raw.get("n_runs"),
        expected=_parse_expected(raw.get("expected")),
        grading=_parse_grading(raw.get("grading")),
    )


def load_dataset(path: str | Path) -> list[Case]:
    """读取 JSONL 数据集，逐条解析，缺省 grading 用场景默认值。"""
    path = Path(path)
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} 不是合法 JSON：{exc}") from exc
            case = parse_case(raw)
            if case.grading is None:
                case.grading = DEFAULT_GRADING.get(case.scenario)
            cases.append(case)
    return cases


def validate(cases: list[Case]) -> None:
    """场景级必填校验，非法即 raise。

    - id 必须唯一
    - scenario 必须在四类内
    - retrieval 必须有 gold_list
    - bug 必须有 root_cause_file 或 root_cause_symbol
    - modify 必须有 target_file 与 key_change_fields
    """
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"用例 id 重复：{case.id}")
        seen.add(case.id)

        if case.scenario not in SCENARIOS:
            raise ValueError(f"[{case.id}] 未知场景：{case.scenario}，应为 {sorted(SCENARIOS)}")

        if case.scenario == SCENARIO_RETRIEVAL and not case.expected.gold_list:
            raise ValueError(f"[{case.id}] retrieval 场景必须提供 expected.gold_list")

        if case.scenario == SCENARIO_BUG and not (
            case.expected.root_cause_file or case.expected.root_cause_symbol
        ):
            raise ValueError(f"[{case.id}] bug 场景必须提供 root_cause_file 或 root_cause_symbol")

        if case.scenario == SCENARIO_MODIFY:
            if not case.expected.target_file:
                raise ValueError(f"[{case.id}] modify 场景必须提供 expected.target_file")
            if not case.expected.key_change_fields:
                raise ValueError(f"[{case.id}] modify 场景必须提供 expected.key_change_fields")


def split_by_scenario(cases: list[Case]) -> dict[str, list[Case]]:
    """按场景分组。"""
    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(case.scenario, []).append(case)
    return groups
