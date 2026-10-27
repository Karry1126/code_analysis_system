"""基线快照：固定环境版本，保证同集 A/B 可归因。

快照记录：
- 被测仓库 git commit（若可用）
- 评测集哈希（train/holdout）
- prompt 哈希（system message）
- 工具 schema 哈希（build_tools_payload）
- 索引文件哈希（data/*.json 与 data/rag/*，因 data/ 不入 git，须靠哈希锁版本）
- 关键 config 参数
- runtime_impl（native | langchain | langgraph）

每次评测前 verify_snapshot() 校验当前环境与快照一致，不一致则拒绝跑分。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_deps() -> dict[str, str]:
    """读取关键依赖的已安装版本，用于评测结果可归因。"""
    names = [
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "openai",
        "pydantic",
    ]
    deps: dict[str, str] = {}
    for name in names:
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = "NOT_INSTALLED"
    return deps


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_head_commit(repo_path: str | Path) -> str | None:
    """获取 git HEAD commit，失败返回 None。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def hash_tree(root: str | Path) -> dict[str, str]:
    """递归哈希目录下所有文件，返回 {相对路径: sha256}。

    排除 __pycache__ 目录与 .pyc/.pyo 缓存文件，避免纯缓存重建造成假不一致。
    """
    root = Path(root)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        out[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return out


def capture_snapshot(
    run_id: str,
    runtime_impl: str,
    *,
    dataset_paths: dict[str, str | Path],
    company_repo_path: str | Path,
    data_dir: str | Path,
    harness_dir: str | Path,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """采集当前环境快照。

    dataset_paths：{name: path}，如 {"train": "eval/datasets/v1/train.jsonl", ...}
    """
    from code_analysis_system.agent_api import build_agent

    agent = build_agent()
    system_message = agent.get_system_message()
    tools_payload = agent.tool_registry.build_tools_payload()
    tool_names = [t["function"]["name"] for t in tools_payload]

    snapshot: dict[str, Any] = {
        "run_id": run_id,
        "runtime_impl": runtime_impl,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "company_repo_commit": git_head_commit(company_repo_path),
        },
        "dataset": {name: sha256_file(path) for name, path in dataset_paths.items()},
        "prompt": {
            "system_prompt_sha256": sha256_text(json.dumps(system_message, ensure_ascii=False)),
        },
        "tools": {
            "schema_sha256": sha256_text(json.dumps(tools_payload, ensure_ascii=False)),
            "names": tool_names,
        },
        "indexes": hash_tree(data_dir),
        "harness": hash_tree(harness_dir),
        "deps": collect_deps(),
        "config": extra_config or {},
    }
    return snapshot


def verify_snapshot(
    snapshot: dict[str, Any],
    *,
    dataset_paths: dict[str, str | Path],
    company_repo_path: str | Path,
    data_dir: str | Path,
    harness_dir: str | Path,
) -> list[str]:
    """校验当前环境是否与快照一致，返回不一致项描述列表（空 = 一致）。"""
    issues: list[str] = []

    commit = git_head_commit(company_repo_path)
    if snapshot.get("git", {}).get("company_repo_commit") != commit:
        issues.append(
            f"被测仓库 commit 不一致：快照={snapshot.get('git', {}).get('company_repo_commit')}，当前={commit}"
        )

    for name, path in dataset_paths.items():
        expected = snapshot.get("dataset", {}).get(name)
        current = sha256_file(path)
        if expected != current:
            issues.append(f"评测集 {name} 哈希不一致：快照={expected}，当前={current}")

    for relpath, expected in snapshot.get("indexes", {}).items():
        current_path = Path(data_dir) / relpath
        if not current_path.exists():
            issues.append(f"索引文件缺失：{relpath}")
            continue
        current = sha256_file(current_path)
        if expected != current:
            issues.append(f"索引文件 {relpath} 哈希不一致")

    # system prompt + 工具 schema：与 capture_snapshot 同源，重新 build_agent 取当前值比对。
    from code_analysis_system.agent_api import build_agent

    agent = build_agent()
    system_message = agent.get_system_message()
    current_prompt_hash = sha256_text(json.dumps(system_message, ensure_ascii=False))
    expected_prompt = snapshot.get("prompt", {}).get("system_prompt_sha256")
    if expected_prompt != current_prompt_hash:
        issues.append(f"system prompt 哈希不一致：快照={expected_prompt}，当前={current_prompt_hash}")

    tools_payload = agent.tool_registry.build_tools_payload()
    current_tools_hash = sha256_text(json.dumps(tools_payload, ensure_ascii=False))
    expected_tools = snapshot.get("tools", {}).get("schema_sha256")
    if expected_tools != current_tools_hash:
        issues.append(f"工具 schema 哈希不一致：快照={expected_tools}，当前={current_tools_hash}")

    # 依赖版本：langchain/langgraph 等版本漂移会改变聚合 runtime 行为，须锁定。
    current_deps = collect_deps()
    for name, expected in snapshot.get("deps", {}).items():
        current = current_deps.get(name)
        if current != expected:
            issues.append(f"依赖 {name} 版本不一致：快照={expected}，当前={current}")

    # harness：判分/驱动逻辑，改动会影响判分口径，须锁定。
    for relpath, expected in snapshot.get("harness", {}).items():
        current_path = Path(harness_dir) / relpath
        if not current_path.exists():
            issues.append(f"harness 文件缺失：{relpath}")
            continue
        current = sha256_file(current_path)
        if expected != current:
            issues.append(f"harness 文件 {relpath} 哈希不一致")

    return issues
