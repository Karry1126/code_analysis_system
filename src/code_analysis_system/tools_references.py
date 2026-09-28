"""符号引用查询工具：find_symbol_references。

find_symbol_references 读 reference_index.json 的 references，
按符号名精确查找文本引用。可选 scope 按 module_path 精确限定模块。
返回引用总数、模块分布，以及最多 20 条样例，不返回全部明细。

索引路径在 config.py 的 REFERENCE_INDEX_PATH。
本模块导入时一次性读入内存。文件缺失或 JSON 非法时工具返回 ok=false，不抛异常。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import REFERENCE_INDEX_PATH
from .framework import ToolRegistry, tool


SAMPLE_REF_LIMIT = 20


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_references(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    references = payload.get("references")
    if not isinstance(references, dict):
        return None
    return references


_REFERENCES = _load_references(REFERENCE_INDEX_PATH)


def _index_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "index not available"}


def _module_key(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


@tool(
    description=(
        "Look up text references to a C/C++ symbol by exact name. "
        "Returns the reference count, per-module counts, and at most 20 sample hits. "
        "Optional scope limits results to one exact module_path. "
        "Does not fuzzy-match. An empty sample_refs list with ref_count 0 means nothing matched."
    ),
    parameter_descriptions={
        "symbol_name": "Exact symbol name, e.g. RankMgr.",
        "scope": "Optional exact module_path, e.g. plt/kvk_alliance_serv. Omit it to search the whole repo.",
    },
)
def find_symbol_references(symbol_name: str, scope: str | None = None) -> dict[str, Any]:
    """按符号名精确查询引用。scope 可选，且只按 module_path 精确匹配。"""
    if _REFERENCES is None:
        return _index_unavailable()

    rows = _REFERENCES.get(symbol_name)
    hits: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            module_path = _module_key(row.get("module_path"))
            if scope is not None and module_path != scope:
                continue
            hits.append(row)

    counts: dict[str, int] = {}
    for row in hits:
        module_path = _module_key(row.get("module_path"))
        if module_path is None:
            continue
        counts[module_path] = counts.get(module_path, 0) + 1

    modules = sorted(counts)
    by_module = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    sample_refs = [
        {
            "module_path": row.get("module_path"),
            "file": row.get("file"),
            "line": row.get("line"),
            "context": row.get("context"),
        }
        for row in hits[:SAMPLE_REF_LIMIT]
    ]
    return {
        "ok": True,
        "symbol": symbol_name,
        "scope": scope,
        "ref_count": len(hits),
        "module_count": len(modules),
        "modules": modules,
        "by_module": by_module,
        "sample_refs": sample_refs,
    }


def register_reference_tools(registry: ToolRegistry) -> None:
    """注册符号引用查询工具。"""
    registry.register_many(find_symbol_references)
