"""执行轨迹的数据结构与落盘。

轨迹 schema 保持框架中立（llm_call / tool_call / final_answer 三个通用事件），
这样后续 LangGraph 通过 callback handler 映射成同一 schema，即可用同一套 grader
对原生实现与迁移实现做同集 A/B。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Trace:
    """一次运行的完整轨迹。"""

    run_id: str
    case_id: str
    status: str  # completed | max_loops_exceeded | error
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str = field(default_factory=_now_iso)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
            "steps": self.steps,
            "aggregates": self.metrics,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_agent_result(cls, case_id: str, result: dict[str, Any], duration_ms: float | None) -> "Trace":
        """从 agent_api.run_agent 的单次结果构造 Trace。"""
        return cls(
            run_id=f"{case_id}__r{result.get('run_id')}",
            case_id=case_id,
            status=result.get("status", "error"),
            answer=result.get("answer", ""),
            steps=result.get("steps", []),
            metrics=result.get("metrics", {}),
            error=result.get("error"),
            duration_ms=duration_ms,
        )
