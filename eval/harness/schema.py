"""评测用例的 schema 定义（单一事实源）。

用 dataclass 而非 pydantic，与项目现有 framework/agent_types.py 风格一致，
零新增依赖。字段全部可选，按 scenario 取用，由 dataset.validate() 做场景级必填校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 四类场景
SCENARIO_CONSULT = "consult"
SCENARIO_RETRIEVAL = "retrieval"
SCENARIO_BUG = "bug"
SCENARIO_MODIFY = "modify"
SCENARIOS = {SCENARIO_CONSULT, SCENARIO_RETRIEVAL, SCENARIO_BUG, SCENARIO_MODIFY}


@dataclass
class Expected:
    """一题的标准答案要点，按场景取用不同字段。"""

    # consult / bug 共用：必须覆盖到的模块与符号
    must_contain_modules: Optional[list[str]] = None
    must_contain_symbols: Optional[list[str]] = None
    # consult / bug：答案正文必须出现的要点（由 mentions_covered 检查，忽略空白）。
    # 一项里用 | 分隔等价说法，命中其中之一即可，例如「回收飞船|领奖」。
    must_mention: Optional[list[str]] = None
    # 轨迹必须调用的工具名（由 required_tools_called 检查）
    required_tools: Optional[list[str]] = None
    # 全部：不允许出现的幻觉内容（不存在的模块/符号/结论）
    must_not_contain: Optional[list[str]] = None
    # retrieval：完整答案集合（模块或符号全集，用于 recall/precision）
    gold_list: Optional[list[str]] = None
    # bug：根因定位（文件 / 行号 / 符号）
    root_cause_file: Optional[str] = None
    root_cause_line: Optional[int] = None
    root_cause_symbol: Optional[str] = None
    # modify：目标文件、关键改动要素、禁止改动的文件。
    # any_filename 为真时不限定新建文件的路径和文件名，只要写出文件即可。
    any_filename: bool = False
    target_file: Optional[str] = None
    # 需全部写到的文件；设置后 only_target_file_modified 要求写全，且不得写集合外的文件
    target_files: Optional[list[str]] = None
    key_change_fields: Optional[list[str]] = None
    forbidden_files: Optional[list[str]] = None
    # llm_judge 评分要点
    rubric: Optional[str] = None


@dataclass
class GradingLayer:
    """一层判分：规则判分（rules）或 LLM 判分（llm_judge）。"""

    type: str  # rules | llm_judge
    checks: Optional[list[str]] = None  # rules 时的检查项名列表
    rubric: Optional[str] = None  # llm_judge 时的评分要点，可覆盖 Expected.rubric
    weight: float = 1.0


@dataclass
class Grading:
    """分层判分：task（答案）/ evidence（证据）/ trace（轨迹）。"""

    task: GradingLayer
    evidence: GradingLayer
    trace: GradingLayer


@dataclass
class Case:
    """一条评测用例。"""

    id: str
    scenario: str
    question: str
    tags: list[str] = field(default_factory=list)
    allow_modify: bool = False
    n_runs: Optional[int] = None  # 缺省时用 config 默认（modify 场景强制 1）
    expected: Expected = field(default_factory=Expected)
    grading: Optional[Grading] = None  # 缺省时用场景默认 grading
