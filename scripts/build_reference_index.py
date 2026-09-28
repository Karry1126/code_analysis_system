"""扫描 symbol_index.json 里出现过的源文件，按完整词抽取符号引用。

依赖 symbol_index.json 的 symbols（符号名和定义位置），以及 module_cards.json 的 module_path。
文件后缀、词边界正则、context 截断长度在下方 CONFIG 区；匹配框架只读这些配置。
不排除注释和字符串，也不区分调用、类型或传参，命中一律记为文本引用。
定义所在的 (file, line) 不记为引用。没有任何引用的符号不写入输出。

输出 JSON:
    meta.repo_root / meta.total_symbols_with_refs / meta.total_refs
    references[symbol] = [{module_path, file, line, context}]
    symbol_ref_summary[symbol] = {ref_count, module_count, modules}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# =============================================================================
# 1. CONFIG 区
# =============================================================================

CONFIG = {
    # 参与扫描的源文件后缀。比较时忽略大小写。
    "source_suffixes": [".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx"],
    # 完整词：命中片段左右相邻的字符都不能落在这个类里。大小写敏感，不加 re.I。
    "word_boundary_class": "A-Za-z0-9_",
    # 符号名本身全部由边界类字符组成时，用这条正则切出候选词，再和符号名精确比对。
    # {boundary} 会换成 word_boundary_class。
    # 候选词是左右都不是边界字符的最长一段，所以和“前后都不是 [A-Za-z0-9_]”是同一条规则。
    "word_char_pattern": "[{boundary}]+",
    # 符号名含有边界类以外的字符时（例如 ~Foo、operator =），每个名字单独套这条模板。
    # {boundary} 换成 word_boundary_class，{name} 换成 re.escape(symbol)。
    # 名字之间互不占用位置：较短的名字只要自己满足边界，就会保留。
    "literal_symbol_pattern": "(?<![{boundary}]){name}(?![{boundary}])",
    # context 先 strip，再截到这个长度。
    "context_max_chars": 200,
    # 文件相对路径和 module_path 做最长前缀匹配，且必须落在路径边界上。
    "module_attribution": {
        "strategy": "longest_prefix",
        "separator": "/",
    },
    "module_card_path_field": "module_path",
    # symbol_index.json 里取符号表、定义文件、定义行号的字段名。
    "symbol_index_fields": {
        "symbols": "symbols",
        "file": "file",
        "line": "line",
    },
    "file_encodings": ["utf-8-sig", "utf-8", "gbk", "latin-1"],
    # 运行结束时打印引用数最高的前 N 个符号名。
    "stats_top_n": 10,
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


def _source_suffixes() -> set[str]:
    return {suffix.lower() for suffix in CONFIG["source_suffixes"]}


def _word_char_pattern() -> str:
    return str(CONFIG["word_char_pattern"]).format(boundary=CONFIG["word_boundary_class"])


def _normalize_rel(path_value: str) -> str:
    return path_value.replace("\\", "/")


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def load_module_paths(cards_path: Path) -> list[str] | None:
    """读取模块路径。文件打不开、JSON 非法或顶层不是 list 时返回 None。"""
    data = load_json(cards_path)
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


def load_symbols(index_path: Path) -> dict | None:
    """读取 symbols。文件打不开、JSON 非法或 symbols 不是 dict 时返回 None。"""
    data = load_json(index_path)
    if not isinstance(data, dict):
        return None
    field = CONFIG["symbol_index_fields"]["symbols"]
    symbols = data.get(field)
    if not isinstance(symbols, dict):
        return None
    return symbols


def attribute_module(relative_path: str, module_paths: list[str]) -> str | None:
    rule = CONFIG["module_attribution"]
    if rule.get("strategy") != "longest_prefix":
        return None
    separator = str(rule.get("separator") or "/")
    for module_path in module_paths:
        if relative_path == module_path or relative_path.startswith(module_path + separator):
            return module_path
    return None


def collect_definitions(symbols: dict) -> dict[str, set[tuple[str, int]]]:
    """符号名 -> 定义位置 (file, line)。这些位置上的命中不记为引用。"""
    file_field = CONFIG["symbol_index_fields"]["file"]
    line_field = CONFIG["symbol_index_fields"]["line"]
    definitions: dict[str, set[tuple[str, int]]] = {}
    for name, rows in symbols.items():
        if not isinstance(name, str) or not name:
            continue
        locations: set[tuple[str, int]] = set()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                file_value = row.get(file_field)
                line = row.get(line_field)
                if not isinstance(file_value, str) or not file_value:
                    continue
                if isinstance(line, bool) or not isinstance(line, int):
                    continue
                locations.add((_normalize_rel(file_value), line))
        definitions[name] = locations
    return definitions


def collect_source_files(definitions: dict[str, set[tuple[str, int]]]) -> list[str]:
    """只收集符号索引里出现过、且后缀落在 CONFIG 里的文件。"""
    suffixes = _source_suffixes()
    found: set[str] = set()
    for locations in definitions.values():
        for relative_file, _line in locations:
            if Path(relative_file).suffix.lower() in suffixes:
                found.add(relative_file)
    return sorted(found)


def compile_patterns(
    symbol_names: list[str],
) -> tuple[re.Pattern[str], set[str], list[tuple[str, re.Pattern[str]]]] | None:
    """纯词符号走切词集合；其余符号各自编译字面正则。"""
    try:
        word_re = re.compile(_word_char_pattern())
    except re.error as exc:
        print(f"word_char_pattern 无法编译: {exc}", file=sys.stderr)
        return None

    word_names: set[str] = set()
    specials: list[tuple[str, re.Pattern[str]]] = []
    template = str(CONFIG["literal_symbol_pattern"])
    boundary = str(CONFIG["word_boundary_class"])
    for name in symbol_names:
        if word_re.fullmatch(name):
            word_names.add(name)
            continue
        try:
            pattern = re.compile(template.format(boundary=boundary, name=re.escape(name)))
        except re.error as exc:
            print(f"字面符号正则无法编译: {name}: {exc}", file=sys.stderr)
            return None
        specials.append((name, pattern))
    return word_re, word_names, specials


def iter_symbol_hits(
    line: str,
    word_re: re.Pattern[str],
    word_names: set[str],
    specials: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    """返回这一行里按出现位置排列的符号名。同一行同一个名字可以出现多次。"""
    found: list[tuple[int, str]] = []
    for match in word_re.finditer(line):
        name = match.group(0)
        if name in word_names:
            found.append((match.start(), name))
    for name, pattern in specials:
        for match in pattern.finditer(line):
            found.append((match.start(), name))
    found.sort(key=lambda item: (item[0], item[1]))
    return [name for _start, name in found]


def _context(line: str) -> str:
    return line.strip()[: int(CONFIG["context_max_chars"])]


def _source_path(repo_root: Path, relative_file: str) -> Path:
    candidate = Path(relative_file)
    if candidate.is_absolute():
        return candidate
    return repo_root / relative_file


def _pick_encoding(path: Path) -> str | None:
    """先把文件过一遍，确认编码能读完，再逐行扫描。不把整文件收成一个字符串。"""
    for encoding in CONFIG["file_encodings"]:
        try:
            with path.open(encoding=encoding) as handle:
                for _line in handle:
                    pass
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
        return encoding
    return None


def _iter_lines(path: Path, encoding: str):
    with path.open(encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            yield line_no, line


# =============================================================================
# 3. 主流程区
# =============================================================================

def scan_references(
    repo_root: Path,
    source_files: list[str],
    module_paths: list[str],
    definitions: dict[str, set[tuple[str, int]]],
    word_re: re.Pattern[str],
    word_names: set[str],
    specials: list[tuple[str, re.Pattern[str]]],
) -> tuple[dict[str, list[dict]], int]:
    references: dict[str, list[dict]] = {}
    scanned_files = 0

    for relative_file in source_files:
        path = _source_path(repo_root, relative_file)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if not is_file:
            print(f"源文件不存在，已跳过: {path}", file=sys.stderr)
            continue

        encoding = _pick_encoding(path)
        if encoding is None:
            print(f"无法读取源文件，已跳过: {path}", file=sys.stderr)
            continue

        module_path = attribute_module(relative_file, module_paths)
        try:
            for line_no, line in _iter_lines(path, encoding):
                for name in iter_symbol_hits(line, word_re, word_names, specials):
                    if (relative_file, line_no) in definitions.get(name, ()):
                        continue
                    references.setdefault(name, []).append({
                        "module_path": module_path,
                        "file": relative_file,
                        "line": line_no,
                        "context": _context(line),
                    })
        except (OSError, UnicodeDecodeError) as exc:
            print(f"读取源文件失败，已跳过: {path}: {exc}", file=sys.stderr)
            continue
        scanned_files += 1

    return references, scanned_files


def finalize_references(references: dict[str, list[dict]]) -> dict[str, list[dict]]:
    ordered: dict[str, list[dict]] = {}
    for name in sorted(references):
        rows = references[name]
        if not rows:
            continue
        rows.sort(key=lambda row: (row["file"], row["line"]))
        ordered[name] = rows
    return ordered


def build_summary(references: dict[str, list[dict]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for name, rows in references.items():
        modules = sorted({
            row["module_path"]
            for row in rows
            if isinstance(row.get("module_path"), str) and row["module_path"]
        })
        summary[name] = {
            "ref_count": len(rows),
            "module_count": len(modules),
            "modules": modules,
        }
    return summary


# =============================================================================
# 4. 输出与统计区
# =============================================================================

def write_index(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_stats(payload: dict, scanned_files: int, total_symbols: int) -> None:
    meta = payload["meta"]
    summary = payload["symbol_ref_summary"]
    ranked = sorted(summary.items(), key=lambda item: (-item[1]["ref_count"], item[0]))
    top_n = int(CONFIG["stats_top_n"])
    unreferenced = total_symbols - int(meta["total_symbols_with_refs"])

    print(f"扫描到的文件数: {scanned_files}")
    print(f"总引用数: {meta['total_refs']}")
    print(f"引用数 top {top_n} 的符号名:")
    for name, info in ranked[:top_n]:
        print(f"  {name}  {info['ref_count']}")
    print(f"未被任何地方引用的符号数: {unreferenced}")


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据符号索引扫描源文件并生成引用索引")
    parser.add_argument("repo_root", help="仓库根目录路径")
    parser.add_argument("--symbol-index", required=True, help="symbol_index.json 路径")
    parser.add_argument("--module-cards", required=True, help="module_cards.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    symbol_index_path = Path(args.symbol_index).expanduser().resolve()
    cards_path = Path(args.module_cards).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    if not repo_root.is_dir():
        print(f"仓库根目录不存在或不是目录: {repo_root}", file=sys.stderr)
        return 1

    symbols = load_symbols(symbol_index_path)
    if symbols is None:
        print(f"无法读取符号索引，或 symbols 不是 object: {symbol_index_path}", file=sys.stderr)
        return 1

    module_paths = load_module_paths(cards_path)
    if module_paths is None:
        print(f"无法读取模块卡片，或顶层不是 list: {cards_path}", file=sys.stderr)
        return 1

    definitions = collect_definitions(symbols)
    compiled = compile_patterns(list(definitions))
    if compiled is None:
        return 1
    word_re, word_names, specials = compiled

    source_files = collect_source_files(definitions)
    references, scanned_files = scan_references(
        repo_root,
        source_files,
        module_paths,
        definitions,
        word_re,
        word_names,
        specials,
    )
    references = finalize_references(references)
    summary = build_summary(references)
    total_refs = sum(len(rows) for rows in references.values())
    payload = {
        "meta": {
            "repo_root": str(repo_root),
            "total_symbols_with_refs": len(references),
            "total_refs": total_refs,
        },
        "references": references,
        "symbol_ref_summary": summary,
    }
    try:
        write_index(payload, output_path)
    except OSError as exc:
        print(f"无法写入引用索引: {exc}", file=sys.stderr)
        return 1

    print_stats(payload, scanned_files, len(definitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
