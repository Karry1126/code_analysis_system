"""扫描 C/C++ 仓库中的 #include，按 header 文件名生成倒排索引 JSON。

模块归属来自 module_cards.json 的 module_path，按最长前缀匹配。
源文件后缀、include 正则、注释标记和归属策略都在下方 CONFIG 区。
要识别新的 include 写法或注释形式，只改 CONFIG，不必改扫描框架。
不属于任何模块的文件、没有 include 的文件直接跳过，不报错。
被注释掉的 include 会保留，并用 commented 标记。

用法:
    python build_include_index.py <仓库根> --module-cards <cards.json> -o <输出.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# =============================================================================
# 1. CONFIG 区
# =============================================================================

CONFIG = {
    # 参与扫描的源文件后缀。比较时忽略大小写。
    "source_suffixes": [".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx"],
    # 对每一行 finditer。两组命名捕获二选一，内容是引号或尖括号里面的原文。
    # 不锚定行首，这样 "// #include ..." 里的指令也能命中。
    "include_pattern": r'#\s*include\s*(?:<(?P<angle>[^>\r\n]+)>|"(?P<quote>[^"\r\n]+)")',
    "include_target_groups": ["angle", "quote"],
    # 注释标记。框架只认这些字符串，不在逻辑里写死 // 或 /* */。
    # 判定：#include 的起始位置落在注释区间内，则 commented=true。
    # // 会注释掉该标记之后的内容，所以 "// #include" 算注释，
    # "#include ... // note" 里的指令本身不算注释。
    # /* */ 可跨行；块注释未闭合时，后续行继续算注释。
    "comment_rules": {
        "line_marker": "//",
        "block_open": "/*",
        "block_close": "*/",
    },
    # 文件相对路径用 separator 拼接后，和 module_path 做最长前缀匹配。
    # 必须落在路径边界上：plt/foo 不能匹配 plt/foobar/a.cpp。
    "module_attribution": {
        "strategy": "longest_prefix",
        "separator": "/",
    },
    # module_cards.json 里取模块路径的字段名。
    "module_card_path_field": "module_path",
    # header_index 的 key：include 目标按这些分隔符取最后一段。
    "basename_separators": ["/", "\\"],
    "file_encodings": ["utf-8-sig", "utf-8", "gbk", "latin-1"],
    "skip_dir_names": [".git", ".svn", "__pycache__", "node_modules", ".venv"],
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


def _skip_names() -> set[str]:
    return set(CONFIG["skip_dir_names"])


def _source_suffixes() -> set[str]:
    return {suffix.lower() for suffix in CONFIG["source_suffixes"]}


def _read_text(path: Path) -> str | None:
    for encoding in CONFIG["file_encodings"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


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


def attribute_module(relative_path: str, module_paths: list[str]) -> str | None:
    rule = CONFIG["module_attribution"]
    if rule.get("strategy") != "longest_prefix":
        return None
    separator = str(rule.get("separator") or "/")
    for module_path in module_paths:
        if relative_path == module_path or relative_path.startswith(module_path + separator):
            return module_path
    return None


def header_basename(include_target: str) -> str:
    separators = [str(item) for item in CONFIG["basename_separators"] if item]
    if not separators:
        return include_target.strip()
    canonical = separators[0]
    name = include_target
    for separator in separators[1:]:
        name = name.replace(separator, canonical)
    name = name.rstrip(canonical)
    if not name:
        return ""
    return name.rsplit(canonical, 1)[-1]


def _marker_at(line: str, marker: str, start: int) -> int:
    if not marker:
        return -1
    return line.find(marker, start)


def comment_spans(line: str, in_block: bool) -> tuple[list[tuple[int, int]], bool]:
    """返回本行的注释半开区间，以及行结束后是否仍在块注释内。"""
    rules = CONFIG["comment_rules"]
    line_marker = str(rules.get("line_marker") or "")
    block_open = str(rules.get("block_open") or "")
    block_close = str(rules.get("block_close") or "")
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(line)

    while index < length:
        if in_block:
            close_at = _marker_at(line, block_close, index)
            if close_at < 0:
                spans.append((index, length))
                return spans, True
            end = close_at + len(block_close)
            spans.append((index, end))
            if end <= index:
                return spans, True
            index = end
            in_block = False
            continue

        line_at = _marker_at(line, line_marker, index)
        open_at = _marker_at(line, block_open, index)
        positions = [(pos, kind) for pos, kind in ((line_at, "line"), (open_at, "block")) if pos >= 0]
        if not positions:
            break
        pos, kind = min(positions)
        if kind == "line":
            spans.append((pos, length))
            return spans, False
        close_at = _marker_at(line, block_close, pos + len(block_open))
        if close_at < 0:
            spans.append((pos, length))
            return spans, True
        end = close_at + len(block_close)
        spans.append((pos, end))
        if end <= pos:
            return spans, True
        index = end
    return spans, False


def _in_spans(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _include_target(match: re.Match[str]) -> str:
    groups = match.groupdict()
    for name in CONFIG["include_target_groups"]:
        value = groups.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_includes(text: str, pattern: re.Pattern[str]) -> list[dict]:
    hits: list[dict] = []
    in_block = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        spans, in_block = comment_spans(line, in_block)
        for match in pattern.finditer(line):
            target = _include_target(match)
            if not target:
                continue
            hits.append({
                "line": line_no,
                "raw": match.group(0).strip(),
                "include_target": target,
                "commented": _in_spans(match.start(), spans),
            })
    return hits


def iter_source_files(repo_root: Path):
    suffixes = _source_suffixes()
    skip = _skip_names()

    def _onerror(_err: OSError) -> None:
        return None

    for dirpath, dirnames, filenames in os.walk(repo_root, followlinks=False, onerror=_onerror):
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in suffixes:
                yield path


# =============================================================================
# 3. 主流程区
# =============================================================================

def _empty_bucket() -> dict:
    return {"by_full_path": {}, "refs": []}


def _append_hit(
    index: dict,
    seen: dict[str, set[tuple]],
    basename: str,
    module_path: str,
    relative_file: str,
    hit: dict,
) -> None:
    identity = (
        module_path,
        relative_file,
        hit["line"],
        hit["include_target"],
        hit["raw"],
        hit["commented"],
    )
    bucket_seen = seen.setdefault(basename, set())
    if identity in bucket_seen:
        return
    bucket_seen.add(identity)

    bucket = index.setdefault(basename, _empty_bucket())
    path_hit = {
        "module_path": module_path,
        "file": relative_file,
        "line": hit["line"],
        "raw": hit["raw"],
        "commented": hit["commented"],
    }
    by_full_path = bucket["by_full_path"]
    by_full_path.setdefault(hit["include_target"], []).append(path_hit)
    bucket["refs"].append({
        "module_path": module_path,
        "file": relative_file,
        "line": hit["line"],
        "raw": hit["raw"],
        "include_target": hit["include_target"],
        "commented": hit["commented"],
    })


def _sort_index(index: dict) -> dict:
    ordered: dict = {}
    for basename in sorted(index):
        bucket = index[basename]
        by_full_path = {}
        for target in sorted(bucket["by_full_path"]):
            rows = bucket["by_full_path"][target]
            rows.sort(key=lambda row: (row["module_path"], row["file"], row["line"], row["raw"]))
            by_full_path[target] = rows
        refs = bucket["refs"]
        refs.sort(key=lambda row: (
            row["module_path"],
            row["file"],
            row["line"],
            row["include_target"],
            row["raw"],
        ))
        ordered[basename] = {"by_full_path": by_full_path, "refs": refs}
    return ordered


def build_index(repo_root: Path, module_paths: list[str], pattern: re.Pattern[str]) -> tuple[dict, int]:
    index: dict = {}
    seen: dict[str, set[tuple]] = {}
    scanned_files = 0

    for path in iter_source_files(repo_root):
        try:
            relative_file = _repo_relative(path, repo_root)
        except ValueError:
            continue
        module_path = attribute_module(relative_file, module_paths)
        if not module_path:
            continue
        text = _read_text(path)
        if text is None:
            continue
        scanned_files += 1
        try:
            hits = extract_includes(text, pattern)
        except re.error:
            continue
        if not hits:
            continue
        for hit in hits:
            basename = header_basename(hit["include_target"])
            if not basename:
                continue
            _append_hit(index, seen, basename, module_path, relative_file, hit)

    return _sort_index(index), scanned_files


# =============================================================================
# 4. 输出与统计区
# =============================================================================

def _occurrence_rows(index: dict) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for basename, bucket in index.items():
        refs = bucket.get("refs") or []
        commented = sum(1 for ref in refs if ref.get("commented"))
        rows.append((basename, len(refs), commented))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def write_index(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_stats(payload: dict) -> None:
    meta = payload["meta"]
    rows = _occurrence_rows(payload["header_index"])
    commented_total = sum(row[2] for row in rows)
    print(f"扫描到的源文件数: {meta['scanned_files']}")
    print(f"识别到的模块数: {meta['scanned_modules']}")
    print(f"索引里的 header 总数: {len(rows)}")
    print("出现次数前 10 的 header basename:")
    for basename, count, _commented in rows[:10]:
        print(f"  {basename}  {count}")
    print(f"被注释掉的 include 总数: {commented_total}")


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 C/C++ 仓库并生成 #include 倒排索引")
    parser.add_argument("repo_root", help="仓库根目录路径")
    parser.add_argument("--module-cards", required=True, help="module_cards.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    cards_path = Path(args.module_cards).expanduser().resolve()
    if not repo_root.is_dir():
        print(f"仓库根目录不存在或不是目录: {repo_root}", file=sys.stderr)
        return 1

    module_paths = load_module_paths(cards_path)
    if module_paths is None:
        print(f"无法读取模块卡片，或顶层不是 list: {cards_path}", file=sys.stderr)
        return 1

    try:
        pattern = re.compile(CONFIG["include_pattern"])
    except re.error as exc:
        print(f"include_pattern 无法编译: {exc}", file=sys.stderr)
        return 1

    header_index, scanned_files = build_index(repo_root, module_paths, pattern)
    total_includes = sum(len(bucket["refs"]) for bucket in header_index.values())
    payload = {
        "meta": {
            "repo_root": str(repo_root),
            "scanned_files": scanned_files,
            "scanned_modules": len(module_paths),
            "total_includes": total_includes,
        },
        "header_index": header_index,
    }
    write_index(payload, Path(args.output))
    print_stats(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
