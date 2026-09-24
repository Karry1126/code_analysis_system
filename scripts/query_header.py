"""用 header 文件名查询 include 倒排索引，把模块和写法分布打到 stdout。

用法:
    python query_header.py <header_basename> <index.json>
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询某个 header 被哪些模块引用")
    parser.add_argument("header", help="header basename，例如 rank_mgr.h")
    parser.add_argument("index", help="build_include_index.py 产出的 JSON")
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

    header_index = payload.get("header_index") if isinstance(payload, dict) else None
    entry = header_index.get(args.header) if isinstance(header_index, dict) else None
    if not isinstance(entry, dict):
        print(f"未找到 header: {args.header}")
        return 1

    counts: dict[str, int] = {}
    for ref in entry.get("refs") or []:
        if not isinstance(ref, dict):
            continue
        module_path = ref.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            continue
        counts[module_path] = counts.get(module_path, 0) + 1

    print(f"header: {args.header}")
    print("引用模块:")
    for module_path in sorted(counts):
        print(f"  {module_path}  {counts[module_path]}")

    print("写法分布:")
    by_full_path = entry.get("by_full_path") if isinstance(entry.get("by_full_path"), dict) else {}
    for target in sorted(by_full_path):
        rows = by_full_path[target]
        count = len(rows) if isinstance(rows, list) else 0
        print(f"  {target}  {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
