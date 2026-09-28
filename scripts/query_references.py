"""用符号名查询引用索引，把引用次数、模块分布和前若干条明细打到 stdout。

读 build_reference_index.py 产出的 JSON。引用为空的符号不会出现在索引里。

用法:
    python query_references.py <symbol_name> <reference_index.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 命令行核对时只展开前这么多条引用明细。
PREVIEW_LIMIT = 10


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


def _module_key(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询某个符号的文本引用")
    parser.add_argument("symbol", help="符号名，与引用索引 references 的 key 一致")
    parser.add_argument("index", help="build_reference_index.py 产出的 JSON")
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

    references = payload.get("references") if isinstance(payload, dict) else None
    rows = references.get(args.symbol) if isinstance(references, dict) else None
    if not isinstance(rows, list):
        print(f"未找到符号: {args.symbol}")
        return 1

    hits = [row for row in rows if isinstance(row, dict)]
    counts: dict[str | None, int] = {}
    for row in hits:
        module_path = _module_key(row.get("module_path"))
        counts[module_path] = counts.get(module_path, 0) + 1

    modules = sorted(counts, key=lambda module_path: (module_path is None, module_path or ""))

    print(f"symbol: {args.symbol}")
    print(f"引用总数: {len(hits)}")
    print("引用模块:")
    for module_path in modules:
        print(f"  {_display(module_path)}")

    print("每个模块的引用次数:")
    for module_path in modules:
        print(f"  {_display(module_path)}  {counts[module_path]}")

    print(f"前 {PREVIEW_LIMIT} 条引用:")
    for row in hits[:PREVIEW_LIMIT]:
        print(
            f"  module_path={_display(row.get('module_path'))}"
            f"  file={_display(row.get('file'))}"
            f"  line={_display(row.get('line'))}"
            f"  context={_display(row.get('context'))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
