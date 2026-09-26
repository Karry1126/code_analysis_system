"""调用 Universal Ctags 抽取 C/C++ 符号定义，生成符号索引 JSON。

依赖本机 Universal Ctags。ctags 参数、版本标记和模块归属策略在下方 CONFIG 区。
启动时跑 ctags --version，输出不含 Universal Ctags 则退出。
ctags 只扫描 module_cards.json 里 module_path 对应的绝对目录。
符号按 name 聚合进 symbols；再按 module_path 聚合进 module_symbols。
匹配不到模块的定义仍保留，module_path 为 null。

用法:
    python build_symbol_index.py <仓库根> --module-cards <cards.json> --ctags <ctags> -o <输出.json>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# =============================================================================
# 1. CONFIG 区
# =============================================================================

CONFIG = {
    # 版本输出里必须出现这段文字。没有就退出，不改走别的 ctags。
    "version_marker": "Universal Ctags",
    # 框架会在这些参数后面追加: -L <绝对目录列表> -o <临时文件>
    # 列表来自 module_cards.json 的 module_path，不扫整个仓库。
    # 本机 Universal Ctags 6.2.0 没有 --files-from，用 -L 读路径列表。
    "ctags_args": [
        "-R",
        "--languages=C,C++",
        "--c++-kinds=+p",
        "--fields=+n",
        "--output-format=json",
    ],
    # JSON Lines 里一条记录的类型字段。不是这个类型的（例如 ptag）不算符号定义。
    "record_type_field": "_type",
    "record_type": "tag",
    # ctags 字段名。signature 可以缺失，缺失时写成 None。
    "tag_fields": {
        "name": "name",
        "path": "path",
        "line": "line",
        "kind": "kind",
        "signature": "signature",
    },
    # 文件相对路径和 module_path 做最长前缀匹配，且必须落在路径边界上。
    "module_attribution": {
        "strategy": "longest_prefix",
        "separator": "/",
    },
    "module_card_path_field": "module_path",
    # 运行结束时按这个顺序打印 kind 计数。计数对象是定义条目，不是符号名。
    "kind_stats": ["class", "function", "member", "macro", "enum", "namespace"],
}


# =============================================================================
# 2. 工具函数区
# =============================================================================

def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_module_paths(cards_path: Path) -> list[str] | None:
    """读取模块路径。文件打不开或 JSON 非法时返回 None。"""
    try:
        data = json.loads(cards_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None

    field = CONFIG["module_card_path_field"]
    separator = CONFIG["module_attribution"]["separator"]
    found: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        value = item.get(field)
        if not isinstance(value, str):
            continue
        module_path = value.replace("\\", separator).strip().strip(separator)
        if not module_path or module_path in seen:
            continue
        seen.add(module_path)
        found.append(module_path)
    found.sort(key=len, reverse=True)
    return found


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def resolve_scan_dirs(repo_root: Path, module_paths: list[str]) -> list[Path]:
    """把 module_path 转成绝对目录。不存在的跳过。

    已被另一个扫描目录包含的路径不再单独列出，避免 ctags -R 扫两遍。
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    for module_path in module_paths:
        absolute = (repo_root / module_path).resolve()
        key = os.path.normcase(os.fspath(absolute))
        if key in seen:
            continue
        seen.add(key)
        if not absolute.is_dir():
            print(f"模块目录不存在，已跳过: {absolute}", file=sys.stderr)
            continue
        candidates.append(absolute)

    scan_dirs = [
        path
        for path in candidates
        if not any(_is_inside(path, other) for other in candidates)
    ]
    scan_dirs.sort(key=lambda path: os.path.normcase(os.fspath(path)))
    return scan_dirs


def attribute_module(relative_path: str, module_paths: list[str]) -> str | None:
    rule = CONFIG["module_attribution"]
    if rule.get("strategy") != "longest_prefix":
        return None
    separator = str(rule.get("separator") or "/")
    for module_path in module_paths:
        if relative_path == module_path or relative_path.startswith(module_path + separator):
            return module_path
    return None


def repo_relative_file(path_value: str, repo_root: Path) -> str:
    root_key = os.fspath(repo_root)
    cache = getattr(repo_relative_file, "_cache", None)
    if cache is None:
        cache = {}
        repo_relative_file._cache = cache
    cached = cache.get((path_value, root_key))
    if cached is not None:
        return cached

    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    else:
        # Windows 上 ctags 可能把绝对路径写成 8.3 短名，先展开再取相对路径。
        try:
            candidate = candidate.resolve()
        except OSError:
            pass
    relative = Path(os.path.relpath(os.fspath(candidate), root_key)).as_posix()
    cache[(path_value, root_key)] = relative
    return relative


def check_universal_ctags(ctags_bin: str) -> tuple[str | None, str | None]:
    """返回 (版本第一行, 错误信息)。不是 Universal Ctags 时不继续调用。"""
    try:
        proc = subprocess.run(
            [ctags_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return None, f"无法执行 ctags: {exc}"

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    marker = str(CONFIG["version_marker"])
    if marker not in stdout and marker not in stderr:
        shown = (stdout or stderr).strip() or "(无输出)"
        return None, f"ctags 不是 Universal Ctags，已退出。\n{shown}"

    for line in stdout.splitlines():
        text = line.strip()
        if text:
            return text, None
    return "", None


def read_ctags_jsonl(path: Path) -> list | None:
    records: list = []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    print(f"ctags 输出第 {line_no} 行不是合法 JSON: {exc}", file=sys.stderr)
                    return None
    except (OSError, UnicodeDecodeError) as exc:
        print(f"无法读取 ctags 输出: {exc}", file=sys.stderr)
        return None
    return records


def _write_files_from(path: Path, scan_dirs: list[Path]) -> None:
    # Windows 上的 ctags 按 ANSI 代码页读这份列表。
    encoding = "mbcs" if os.name == "nt" else "utf-8"
    text = "".join(f"{directory}\n" for directory in scan_dirs)
    path.write_bytes(text.encode(encoding))


def run_ctags(ctags_bin: str, repo_root: Path, scan_dirs: list[Path]) -> list | None:
    fd, temp_name = tempfile.mkstemp(prefix="ctags-", suffix=".json")
    os.close(fd)
    temp_path = Path(temp_name)
    list_fd, list_name = tempfile.mkstemp(prefix="ctags-files-", suffix=".txt")
    os.close(list_fd)
    list_path = Path(list_name)
    try:
        _write_files_from(list_path, scan_dirs)
        command = [
            ctags_bin,
            *CONFIG["ctags_args"],
            "-L",
            str(list_path),
            "-o",
            str(temp_path),
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            print(f"无法执行 ctags: {exc}", file=sys.stderr)
            return None
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            print(f"ctags 退出码 {proc.returncode}", file=sys.stderr)
            if detail:
                print(detail, file=sys.stderr)
            return None
        return read_ctags_jsonl(temp_path)
    finally:
        for path in (temp_path, list_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _definition_from_record(record: dict, repo_root: Path, module_paths: list[str]) -> dict | None:
    type_field = CONFIG["record_type_field"]
    expected_type = CONFIG["record_type"]
    if expected_type is not None and record.get(type_field) != expected_type:
        return None

    fields = CONFIG["tag_fields"]
    name = record.get(fields["name"])
    path_value = record.get(fields["path"])
    line = record.get(fields["line"])
    kind = record.get(fields["kind"])
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(path_value, str) or not path_value:
        return None
    if isinstance(line, bool) or not isinstance(line, int):
        return None
    if not isinstance(kind, str) or not kind:
        return None

    signature = record.get(fields["signature"])
    if not isinstance(signature, str):
        signature = None

    relative_file = repo_relative_file(path_value, repo_root)
    return {
        "name": name,
        "module_path": attribute_module(relative_file, module_paths),
        "file": relative_file,
        "line": line,
        "kind": kind,
        "signature": signature,
    }


# =============================================================================
# 3. 主流程区
# =============================================================================

def build_indexes(records: list, repo_root: Path, module_paths: list[str]) -> tuple[dict, dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        item = _definition_from_record(record, repo_root, module_paths)
        if item is None:
            continue
        grouped.setdefault(item["name"], []).append(item)

    symbols: dict[str, list[dict]] = {}
    module_symbols: dict[str, list[dict]] = {}
    for name in sorted(grouped):
        rows = grouped[name]
        rows.sort(key=lambda row: (row["file"], row["line"], row["kind"], row["signature"] or ""))
        symbols[name] = [
            {
                "module_path": row["module_path"],
                "file": row["file"],
                "line": row["line"],
                "kind": row["kind"],
                "signature": row["signature"],
            }
            for row in rows
        ]
        for row in rows:
            module_path = row["module_path"]
            if not isinstance(module_path, str):
                continue
            module_symbols.setdefault(module_path, []).append({
                "name": name,
                "kind": row["kind"],
                "file": row["file"],
                "line": row["line"],
            })

    ordered_modules: dict[str, list[dict]] = {}
    for module_path in sorted(module_symbols):
        entries = module_symbols[module_path]
        entries.sort(key=lambda row: (row["file"], row["line"], row["name"], row["kind"]))
        ordered_modules[module_path] = entries
    return symbols, ordered_modules


# =============================================================================
# 4. 输出与统计区
# =============================================================================

def write_index(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_stats(payload: dict, raw_count: int) -> None:
    symbols = payload["symbols"]
    kind_counts = {kind: 0 for kind in CONFIG["kind_stats"]}
    null_modules = 0
    for rows in symbols.values():
        for row in rows:
            kind = row.get("kind")
            if kind in kind_counts:
                kind_counts[kind] += 1
            if row.get("module_path") is None:
                null_modules += 1

    print(f"ctags 原始条目数: {raw_count}")
    print(f"去重后的符号名数: {payload['meta']['total_symbols']}")
    print("kind 分布:")
    for kind in CONFIG["kind_stats"]:
        print(f"  {kind}: {kind_counts[kind]}")
    print(f"module_path 为 null 的条目数: {null_modules}")


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 Universal Ctags 生成 C/C++ 符号索引")
    parser.add_argument("repo_root", help="仓库根目录路径")
    parser.add_argument("--module-cards", required=True, help="module_cards.json 路径")
    parser.add_argument("--ctags", required=True, help="Universal Ctags 可执行文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    cards_path = Path(args.module_cards).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    if not repo_root.is_dir():
        print(f"仓库根目录不存在或不是目录: {repo_root}", file=sys.stderr)
        return 1

    version, error = check_universal_ctags(args.ctags)
    if error is not None or version is None:
        print(error or "ctags 不是 Universal Ctags，已退出。", file=sys.stderr)
        return 1

    module_paths = load_module_paths(cards_path)
    if module_paths is None:
        print(f"无法读取模块卡片，或顶层不是 list: {cards_path}", file=sys.stderr)
        return 1

    scan_dirs = resolve_scan_dirs(repo_root, module_paths)
    if not scan_dirs:
        print(f"没有可扫描的模块目录: {cards_path}", file=sys.stderr)
        return 1

    records = run_ctags(args.ctags, repo_root, scan_dirs)
    if records is None:
        return 1

    symbols, module_symbols = build_indexes(records, repo_root, module_paths)
    total_definitions = sum(len(rows) for rows in symbols.values())
    payload = {
        "meta": {
            "repo_root": str(repo_root),
            "ctags_version": version,
            "total_symbols": len(symbols),
            "total_definitions": total_definitions,
        },
        "symbols": symbols,
        "module_symbols": module_symbols,
    }
    write_index(payload, output_path)
    print_stats(payload, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
