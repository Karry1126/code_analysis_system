"""代码修改类场景的工作区隔离与回滚。

modify 场景有文件副作用，不能直接跑在真实 COMPANY_CODE_REPO_PATH 上。
Sandbox 把真实仓库复制到临时副本，改代码动作全部落在副本上，
执行完统一删除回滚。改代码场景必须在子进程执行：config.py 在 import 时
解析 COMPANY_CODE_RELATIVE_PATH 为模块级常量，只有新进程才能在 import 前
注入指向副本的环境变量。
"""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class Sandbox:
    """modify 场景的临时工作区。"""

    def __init__(self, company_repo_path: str | Path) -> None:
        self.company_repo_path = Path(company_repo_path).resolve()
        self.tmp_dir: Path | None = None
        self.repo_copy: Path | None = None

    def prepare(self) -> Path:
        """复制真实仓库到临时副本，返回副本路径。"""
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="eval_modify_sandbox_"))
        self.repo_copy = self.tmp_dir / "repo"
        shutil.copytree(
            self.company_repo_path,
            self.repo_copy,
            ignore=shutil.ignore_patterns(".git"),
        )
        return self.repo_copy

    def teardown(self) -> None:
        """无论成败删除副本，回滚所有改动。"""
        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_dir = None
            self.repo_copy = None

    def compute_diff(self, target_file: str) -> str:
        """对比真实仓库与副本的 target_file，返回 unified diff 文本。"""
        original_path = self.company_repo_path / target_file
        modified_path = self.repo_copy / target_file  # type: ignore[operator]

        def read(p: Path) -> str:
            try:
                return p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return ""

        original = read(original_path)
        modified = read(modified_path)
        return "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                modified.splitlines(),
                fromfile=f"a/{target_file}",
                tofile=f"b/{target_file}",
                lineterm="",
            )
        )

    def __enter__(self) -> "Sandbox":
        self.prepare()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.teardown()


def run_modify_isolated(
    question: str,
    repo_copy: Path,
    *,
    n_runs: int = 1,
    runtime_impl: str = "native",
) -> dict[str, Any]:
    """在子进程中把工作区指向副本，执行一次 Agent，返回结构化结果。

    子进程先设 COMPANY_CODE_RELATIVE_PATH=副本绝对路径，再 import config，
    使 resolve_safe_path 等工具全部落在副本上，不触碰真实仓库。
    """
    script = (
        "import os, sys, json\n"
        f"os.environ['COMPANY_CODE_RELATIVE_PATH'] = {str(repo_copy)!r}\n"
        "from code_analysis_system.agent_api import run_agent\n"
        f"result = run_agent({question!r}, n_runs={n_runs}, runtime_impl={runtime_impl!r})[0]\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return {"status": "error", "error": f"modify 子进程超时：{exc}", "answer": "", "steps": [], "metrics": {}}

    if proc.returncode != 0:
        return {
            "status": "error",
            "error": f"modify 子进程退出码 {proc.returncode}: {proc.stderr[-1000:]}",
            "answer": "",
            "steps": [],
            "metrics": {},
        }

    lines = proc.stdout.strip().splitlines()
    # 子进程 stdout 里，最后一行是结果 JSON，其余是 Agent 的工具调用日志。
    # 把这些日志透传到父进程终端，便于观察 modify 题的执行过程。
    if len(lines) > 1:
        print("\n".join(lines[:-1]))
    try:
        return json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        return {
            "status": "error",
            "error": f"modify 子进程输出无法解析：{proc.stdout[-500:]}",
            "answer": "",
            "steps": [],
            "metrics": {},
        }
