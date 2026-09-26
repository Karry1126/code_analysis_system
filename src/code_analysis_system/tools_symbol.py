"""符号索引查询工具：find_symbol_definition、list_symbols_in_module。

find_symbol_definition 读 symbol_index.json 的 symbols，
按符号名精确查找定义位置和所属模块。
list_symbols_in_module 读同一文件的 module_symbols，
按 module_path 精确列出符号，可选按 kind 精确过滤。

索引路径在 config.py 的 SYMBOL_INDEX_PATH。
本模块导入时一次性读入内存。文件缺失或 JSON 非法时工具返回 ok=false，不抛异常。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import SYMBOL_INDEX_PATH
from .framework import ToolRegistry, tool


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_symbol_tables(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None, None
    symbols = payload.get("symbols")
    module_symbols = payload.get("module_symbols")
    if not isinstance(symbols, dict):
        symbols = None
    if not isinstance(module_symbols, dict):
        module_symbols = None
    return symbols, module_symbols


_SYMBOLS, _MODULE_SYMBOLS = _load_symbol_tables(SYMBOL_INDEX_PATH)


def _index_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "index not available"}


@tool(
    description=(
        "Look up C/C++ symbol definitions by exact name. "
        "Returns file, line, kind, signature, and the modules that own those definitions. "
        "Does not fuzzy-match or resolve aliases. An empty definitions list means the name is absent."
    ),
    parameter_descriptions={
        "symbol_name": "Exact symbol name, e.g. RankMgr.",
    },
)
def find_symbol_definition(symbol_name: str) -> dict[str, Any]:
    """按符号名精确查询定义位置和所属模块。"""
    if _SYMBOLS is None:
        return _index_unavailable()

    rows = _SYMBOLS.get(symbol_name)
    definitions: list[dict[str, Any]] = []
    module_paths: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            module_path = row.get("module_path")
            definitions.append(
                {
                    "module_path": module_path,
                    "file": row.get("file"),
                    "line": row.get("line"),
                    "kind": row.get("kind"),
                    "signature": row.get("signature"),
                }
            )
            if isinstance(module_path, str) and module_path:
                module_paths.add(module_path)

    modules = sorted(module_paths)
    return {
        "ok": True,
        "symbol": symbol_name,
        "definitions": definitions,
        "definition_count": len(definitions),
        "modules": modules,
        "module_count": len(modules),
    }


@tool(
    description=(
        "List symbols defined in one module, by exact module_path. "
        "Optionally filter by exact kind, such as class, function, member, macro, enum, or namespace. "
        "Returns the full list with no truncation. An empty symbols list means the module is absent or nothing matched."
    ),
    parameter_descriptions={
        "module_path": "Exact module path, e.g. plt/rank_event_center.",
        "kind": "Optional exact kind filter. Omit it to return every symbol in the module.",
    },
)
def list_symbols_in_module(module_path: str, kind: str | None = None) -> dict[str, Any]:
    """按 module_path 精确列出模块内符号，kind 可选且只做精确匹配。"""
    if _MODULE_SYMBOLS is None:
        return _index_unavailable()

    rows = _MODULE_SYMBOLS.get(module_path)
    symbols: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_kind = row.get("kind")
            if kind is not None and row_kind != kind:
                continue
            symbols.append(
                {
                    "name": row.get("name"),
                    "kind": row_kind,
                    "file": row.get("file"),
                    "line": row.get("line"),
                }
            )

    return {
        "ok": True,
        "module_path": module_path,
        "filter": {"kind": kind},
        "symbols": symbols,
        "symbol_count": len(symbols),
    }


def register_symbol_tools(registry: ToolRegistry) -> None:
    """注册符号索引查询工具。"""
    registry.register_many(
        find_symbol_definition,
        list_symbols_in_module,
    )
