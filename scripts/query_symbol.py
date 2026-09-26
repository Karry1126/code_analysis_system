"""用符号名查询符号索引，把定义位置和所属模块打到 stdout。

用法:
    python query_symbol.py <symbol_name> <index.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _display(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询某个符号的定义位置")
    parser.add_argument("symbol", help="符号名，与索引 symbols 的 key 一致")
    parser.add_argument("index", help="build_symbol_index.py 产出的 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    index_path = Path(args.index)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"无法读取索引: {exc}", file=sys.stderr)
        return 1

    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    rows = symbols.get(args.symbol) if isinstance(symbols, dict) else None
    if not isinstance(rows, list):
        print(f"未找到符号: {args.symbol}")
        return 1

    print(f"symbol: {args.symbol}")
    print(f"定义总数: {len(rows)}")
    print("定义位置:")
    modules: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        module_path = _display(row.get("module_path"))
        modules.add(module_path)
        print(
            f"  module_path={module_path}"
            f"  file={_display(row.get('file'))}"
            f"  line={_display(row.get('line'))}"
            f"  kind={_display(row.get('kind'))}"
            f"  signature={_display(row.get('signature'))}"
        )

    print("出现的 module:")
    for module_path in sorted(modules):
        print(f"  {module_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
