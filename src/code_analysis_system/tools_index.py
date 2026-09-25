"""离线索引查询工具：query_include_graph、list_modules。

query_include_graph 读 include_index.json 的 header_index，
按 header basename 原样查找，返回引用模块和 include 写法计数。
list_modules 读 module_cards.json，返回全部模块的
module_path、module_name、module_type。

两条索引的路径在 config.py：INCLUDE_INDEX_PATH、MODULE_CARDS_PATH。
本模块导入时一次性读入内存。文件缺失或 JSON 非法时工具返回 ok=false，不抛异常。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import INCLUDE_INDEX_PATH, MODULE_CARDS_PATH
from .framework import ToolRegistry, tool


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_header_index(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    header_index = payload.get("header_index")
    if not isinstance(header_index, dict):
        return None
    return header_index


def _load_module_cards(path: Path) -> list[Any] | None:
    payload = _load_json(path)
    if not isinstance(payload, list):
        return None
    return payload


_HEADER_INDEX = _load_header_index(INCLUDE_INDEX_PATH)
_MODULE_CARDS = _load_module_cards(MODULE_CARDS_PATH)


def _index_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "index not available"}


@tool(
    description=(
        "Look up which modules include a C/C++ header, by exact basename. "
        "Returns deduplicated module paths and include-writing counts, not file or line details. "
        "Does not guess aliases. An empty modules list means the header is absent from the index."
    ),
    parameter_descriptions={
        "header": "Header basename to look up exactly, e.g. rank_mgr.h.",
    },
)
def query_include_graph(header: str) -> dict[str, Any]:
    """按 header basename 查询 include 倒排索引，只返回模块列表和写法计数。"""
    if _HEADER_INDEX is None:
        return _index_unavailable()

    entry = _HEADER_INDEX.get(header)
    if not isinstance(entry, dict):
        return {
            "ok": True,
            "header": header,
            "modules": [],
            "module_count": 0,
            "by_writing": {},
            "total_refs": 0,
        }

    module_paths: set[str] = set()
    refs = entry.get("refs")
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            module_path = ref.get("module_path")
            if isinstance(module_path, str) and module_path:
                module_paths.add(module_path)

    by_writing: dict[str, int] = {}
    by_full_path = entry.get("by_full_path")
    if isinstance(by_full_path, dict):
        for target in sorted(key for key in by_full_path if isinstance(key, str)):
            rows = by_full_path[target]
            by_writing[target] = len(rows) if isinstance(rows, list) else 0

    modules = sorted(module_paths)
    return {
        "ok": True,
        "header": header,
        "modules": modules,
        "module_count": len(modules),
        "by_writing": by_writing,
        "total_refs": sum(by_writing.values()),
    }


@tool(
    description=(
        "List all modules in the codebase, with their module_path, "
        "module_name, and module_type. "
        "Returns the full list with no filtering or truncation."
    ),
)
def list_modules() -> dict[str, Any]:
    """返回全部模块的 module_path、module_name、module_type。"""
    if _MODULE_CARDS is None:
        return _index_unavailable()

    modules: list[dict[str, Any]] = []
    for card in _MODULE_CARDS:
        if not isinstance(card, dict):
            continue
        modules.append(
            {
                "module_path": card.get("module_path"),
                "module_name": card.get("module_name"),
                "module_type": card.get("module_type"),
            }
        )

    return {
        "ok": True,
        "modules": modules,
        "module_count": len(modules),
    }


def register_index_tools(registry: ToolRegistry) -> None:
    """注册离线索引查询工具。"""
    registry.register_many(
        query_include_graph,
        list_modules,
    )
