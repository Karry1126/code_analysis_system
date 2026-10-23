"""LLM judge 与人工标注的一致性校准。

校准目的：回应「不承诺全自动判分」。judge 上线前必须与人工标注对齐，
计算 Cohen's κ；κ < 0.6 说明 rubric 不可用，需改 rubric 后重校。

校准流程（实际在 S7 全量跑出作答后执行）：
1. 每场景抽样 ≥10 条作答，覆盖对/错/边界；
2. 2 名人工按同一 rubric 标注（artifacts/calibration_labels.jsonl）；
3. 用 judge 对同一批作答打分，与人工比对；
4. 计算 κ / 一致率 / 分歧样例，κ < 0.6 则改 rubric 迭代；
5. 冻结 rubric 版本写入 snapshot。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# 人工与 judge 都映射到三分类标签，便于计算 κ。
_LABELS = ("对", "部分对", "错")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """两个离散标签列表的 Cohen's kappa。"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    labels = set(a) | set(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _normalize(label: str) -> str:
    label = (label or "").strip()
    if label in _LABELS:
        return label
    if label in {"正确", "对"}:
        return "对"
    if label in {"部分", "部分正确"}:
        return "部分对"
    if label in {"错误", "错"}:
        return "错"
    return label


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    """加载校准标注 JSONL。每行 {case_id, question, answer, human, judge}。"""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def run_calibration(
    labels: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """计算 judge 与人工的 κ、一致率、分歧样例。"""
    human = [_normalize(str(r.get("human", ""))) for r in labels]
    judge = [_normalize(str(r.get("judge", ""))) for r in labels]

    kappa = cohen_kappa(human, judge)
    n = len(labels)
    agreements = sum(1 for h, j in zip(human, judge) if h == j)
    accuracy = agreements / n if n else 0.0

    disagreements = [
        {
            "case_id": r.get("case_id"),
            "question": r.get("question"),
            "human": h,
            "judge": j,
        }
        for r, h, j in zip(labels, human, judge)
        if h != j
    ]

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r, h, j in zip(labels, human, judge):
        by_scenario.setdefault(str(r.get("scenario", "unknown")), []).append((h, j))
    per_scenario = {
        s: {
            "n": len(pairs),
            "kappa": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs]),
        }
        for s, pairs in by_scenario.items()
    }

    return {
        "n": n,
        "kappa": kappa,
        "accuracy": accuracy,
        "usable": kappa >= threshold,
        "threshold": threshold,
        "per_scenario": per_scenario,
        "disagreements": disagreements,
    }
