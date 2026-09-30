"""为仓库构建 RAG 检索语料，输出 JSON。

只读 module_cards.json、symbol_index.json 和源码文本，不做 AST / ctags。
正则、阈值、截断长度、注释规则都在下方 CONFIG 区；抽不到的字段留空，不报错。
输出 docs 里每条是一个实体：module 或 symbol。

用法:
    python build_rag_corpus.py <仓库根> --module-cards <cards.json> --symbol-index <index.json> -o <输出.json>
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
    "skip_dir_names": [".git", ".svn", "__pycache__", "node_modules", ".venv"],
    "file_encodings": ["utf-8-sig", "utf-8", "gbk", "latin-1"],
    "src_relpath": "src",
    "source_suffixes": [".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".py"],
    "md_suffixes": [".md"],
    "md_search_recursive": True,
    # 卡片字段名。框架按这些 key 取值，不在逻辑里写死业务文件名。
    "card_fields": {
        "module_path": "module_path",
        "module_name": "module_name",
        "module_type": "module_type",
        "config_files": "config_files",
        "doc_files": "doc_files",
        "entry_files": "entry_files",
    },
    "index_fields": {
        "symbols": "symbols",
        "name": "name",
        "kind": "kind",
        "file": "file",
        "line": "line",
        "signature": "signature",
        "module_path": "module_path",
    },
    # 出现在模块内任意深度、且文件名命中时，并入 key_files。
    "extra_key_file_names": ["global_serv.h", "event_process.h"],
    # 分词：先按这些分隔符切开，再对每段套 token_pattern。
    "name_separators": ["_", "-", ".", "/"],
    "name_token_pattern": r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+",
    "chinese_pattern": r"[\u4e00-\u9fff]+",
    "min_chinese_len": 2,
    "module_comment_max_chars": 5000,
    "symbol_comment_max_chars": 1000,
    "symbol_comment_window": 30,
    # 含这些字符的源码行不抽中文，避免落到字符串字面量里。
    "quote_chars": ['"', "'"],
    "comment_rules": {
        "c_like": {
            "suffixes": [".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx"],
            "line_marker": "//",
            "block_open": "/*",
            "block_close": "*/",
        },
        "python": {
            "suffixes": [".py"],
            "line_marker": "#",
            "block_open": None,
            "block_close": None,
        },
    },
    # 同名符号多处定义时，按 kind 优先级挑一条写进语料。未列出的 kind 排最后。
    "preferred_symbol_kinds": [
        "class",
        "struct",
        "namespace",
        "enum",
        "function",
        "member",
        "macro",
        "enumerator",
        "prototype",
    ],
    "join_tokens": " ",
    "join_files": " ",
    "join_comments": " ",
    "join_documents": "\n",
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


def _md_suffixes() -> set[str]:
    return {suffix.lower() for suffix in CONFIG["md_suffixes"]}


def _comment_rule_for_suffix(suffix: str) -> dict | None:
    lowered = suffix.lower()
    for rule in CONFIG["comment_rules"].values():
        suffixes = {item.lower() for item in rule.get("suffixes", [])}
        if lowered in suffixes:
            return rule
    return None


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _iter_files(root: Path):
    skip = _skip_names()

    def _onerror(_err: OSError) -> None:
        return None

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_onerror):
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            yield Path(dirpath) / filename


def _rel_posix(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return "." if relative == "" else relative


def _read_text(path: Path) -> str | None:
    if not _is_file(path):
        return None
    for encoding in CONFIG["file_encodings"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def display(value) -> str:
    if value is None:
        return ""
    return str(value)


def split_name_tokens(name: str | None) -> str:
    if not isinstance(name, str) or not name:
        return ""
    separators = CONFIG["name_separators"]
    token_re = re.compile(CONFIG["name_token_pattern"])
    chunks = [name]
    for separator in separators:
        next_chunks: list[str] = []
        for chunk in chunks:
            next_chunks.extend(part for part in chunk.split(separator) if part)
        chunks = next_chunks
    tokens: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        parts = token_re.findall(chunk) or [chunk]
        for part in parts:
            token = part.lower()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return CONFIG["join_tokens"].join(tokens)


def _line_has_quotes(line: str) -> bool:
    return any(char in line for char in CONFIG["quote_chars"])


def _chinese_fragments(text: str) -> list[str]:
    if not text:
        return []
    pattern = re.compile(CONFIG["chinese_pattern"])
    min_len = int(CONFIG["min_chinese_len"])
    found: list[str] = []
    for match in pattern.findall(text):
        if len(match) >= min_len:
            found.append(match)
    return found


def _comment_rule_slices(line: str, rule: dict, in_block: bool) -> tuple[list[str], bool]:
    """按 CONFIG 里的行注释 / 块注释标记切出注释片段，并返回新的块注释状态。"""
    line_marker = rule.get("line_marker")
    block_open = rule.get("block_open")
    block_close = rule.get("block_close")
    slices: list[str] = []
    index = 0
    length = len(line)

    while index < length:
        if in_block:
            close_at = line.find(block_close, index) if block_close else -1
            if close_at < 0:
                slices.append(line[index:])
                return slices, True
            slices.append(line[index:close_at])
            index = close_at + len(block_close)
            in_block = False
            continue

        open_at = line.find(block_open, index) if block_open else -1
        marker_at = line.find(line_marker, index) if line_marker else -1
        if open_at < 0 and marker_at < 0:
            break
        if marker_at >= 0 and (open_at < 0 or marker_at < open_at):
            slices.append(line[marker_at + len(line_marker) :])
            break
        slices.append("")
        index = open_at + len(block_open)
        in_block = True

    return slices, in_block


def extract_line_chinese(lines: list[str], suffix: str) -> list[list[str]]:
    """对每一行抽出中文注释片段。含引号的行跳过。返回与 lines 等长的列表（0-based）。"""
    rule = _comment_rule_for_suffix(suffix)
    per_line: list[list[str]] = [[] for _ in lines]
    if rule is None:
        return per_line

    in_block = False
    for index, raw in enumerate(lines):
        slices, in_block = _comment_rule_slices(raw, rule, in_block)
        if _line_has_quotes(raw):
            continue
        fragments: list[str] = []
        for slice_text in slices:
            fragments.extend(_chinese_fragments(slice_text))
        per_line[index] = fragments
    return per_line


def unique_join(fragments: list[str], max_chars: int) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for fragment in fragments:
        if fragment in seen:
            continue
        seen.add(fragment)
        ordered.append(fragment)
    joined = CONFIG["join_comments"].join(ordered)
    limit = int(max_chars)
    if limit >= 0 and len(joined) > limit:
        return joined[:limit]
    return joined


def join_file_list(items) -> str:
    if not isinstance(items, list):
        return ""
    names = [str(item) for item in items if isinstance(item, str) and item]
    return CONFIG["join_files"].join(names)


# =============================================================================
# 3. 主流程区
# =============================================================================

def collect_module_files(module_dir: Path) -> tuple[list[str], list[Path], list[Path]]:
    """返回 (key_files 相对路径, src 源码路径, 文档路径)。"""
    extra_names = set(CONFIG["extra_key_file_names"])
    extra_found: list[str] = []
    source_paths: list[Path] = []
    src_root = module_dir / str(CONFIG["src_relpath"])
    source_suffixes = _source_suffixes()
    if _is_dir(src_root):
        for path in _iter_files(src_root):
            if path.suffix.lower() in source_suffixes:
                source_paths.append(path)
            if path.name in extra_names:
                try:
                    extra_found.append(_rel_posix(path, module_dir))
                except Exception:
                    continue

    doc_paths: list[Path] = []
    md_suffixes = _md_suffixes()
    if CONFIG["md_search_recursive"] and _is_dir(module_dir):
        for path in _iter_files(module_dir):
            if path.suffix.lower() in md_suffixes:
                doc_paths.append(path)
    return extra_found, source_paths, doc_paths


def read_document_text(module_dir: Path, card_doc_files: list, extra_md_paths: list[Path]) -> str:
    seen: set[str] = set()
    chunks: list[str] = []

    def _add(path: Path) -> None:
        try:
            key = os.path.normcase(os.fspath(path.resolve()))
        except OSError:
            key = os.path.normcase(os.fspath(path))
        if key in seen:
            return
        seen.add(key)
        text = _read_text(path)
        if text:
            chunks.append(text)

    if isinstance(card_doc_files, list):
        for rel in card_doc_files:
            if not isinstance(rel, str) or not rel:
                continue
            _add(module_dir / rel)
    for path in extra_md_paths:
        _add(path)
    return CONFIG["join_documents"].join(chunks)


def module_chinese_comments(source_paths: list[Path]) -> str:
    fragments: list[str] = []
    for path in source_paths:
        text = _read_text(path)
        if text is None:
            continue
        lines = text.splitlines()
        for line_fragments in extract_line_chinese(lines, path.suffix):
            fragments.extend(line_fragments)
    return unique_join(fragments, CONFIG["module_comment_max_chars"])


def build_module_text(card: dict, extra_key_files: list[str], document_text: str, chinese: str) -> str:
    fields = CONFIG["card_fields"]
    module_path = display(card.get(fields["module_path"]))
    module_name = display(card.get(fields["module_name"]))
    module_type = display(card.get(fields["module_type"]))
    entry_files = card.get(fields["entry_files"]) or []
    key_files = []
    if isinstance(entry_files, list):
        key_files.extend(str(item) for item in entry_files if isinstance(item, str) and item)
    for item in extra_key_files:
        if item not in key_files:
            key_files.append(item)
    config_files = card.get(fields["config_files"]) or []
    doc_files = card.get(fields["doc_files"]) or []
    parts = [
        f"[module] {module_path}",
        f"module_name: {module_name}",
        f"module_type: {module_type}",
        f"path_tokens: {split_name_tokens(card.get(fields['module_name']) if isinstance(card.get(fields['module_name']), str) else None)}",
        f"key_files: {join_file_list(key_files)}",
        f"config_files: {join_file_list(config_files)}",
        f"doc_files: {join_file_list(doc_files)}",
        "",
        "document_text:",
        document_text,
        "",
        "chinese_comments:",
        chinese,
    ]
    return "\n".join(parts)


def build_module_docs(repo_root: Path, cards: list) -> tuple[list[dict], int]:
    fields = CONFIG["card_fields"]
    docs: list[dict] = []
    with_chinese = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        module_path = card.get(fields["module_path"])
        if not isinstance(module_path, str) or not module_path:
            continue
        module_dir = repo_root / module_path
        extra_key_files, source_paths, extra_md_paths = collect_module_files(module_dir)
        document_text = read_document_text(
            module_dir,
            card.get(fields["doc_files"]) or [],
            extra_md_paths,
        )
        chinese = module_chinese_comments(source_paths) if _is_dir(module_dir) else ""
        if chinese:
            with_chinese += 1
        text = build_module_text(card, extra_key_files, document_text, chinese)
        docs.append({
            "doc_id": f"module:{module_path}",
            "kind": "module",
            "module_path": module_path,
            "text": text,
            "metadata": {
                "module_name": card.get(fields["module_name"]) if isinstance(card.get(fields["module_name"]), str) else None,
                "module_type": card.get(fields["module_type"]) if card.get(fields["module_type"]) is not None else None,
                "config_files": list(card.get(fields["config_files"]) or []) if isinstance(card.get(fields["config_files"]), list) else [],
                "doc_files": list(card.get(fields["doc_files"]) or []) if isinstance(card.get(fields["doc_files"]), list) else [],
            },
        })
    docs.sort(key=lambda item: item.get("module_path") or "")
    return docs, with_chinese


class SourceCommentCache:
    def __init__(self) -> None:
        self._lines: dict[str, list[list[str]]] = {}

    def fragments_in_window(self, repo_root: Path, relative_file: str, line: int) -> list[str]:
        key = relative_file.replace("\\", "/")
        if key not in self._lines:
            path = repo_root / key
            text = _read_text(path)
            if text is None:
                self._lines[key] = []
            else:
                rows = text.splitlines()
                self._lines[key] = extract_line_chinese(rows, path.suffix)
        rows = self._lines[key]
        if not rows:
            return []
        window = int(CONFIG["symbol_comment_window"])
        center = line if isinstance(line, int) and not isinstance(line, bool) else 1
        start = max(1, center - window)
        end = min(len(rows), center + window)
        fragments: list[str] = []
        for index in range(start - 1, end):
            fragments.extend(rows[index])
        return fragments


def pick_definition(rows: list) -> dict | None:
    fields = CONFIG["index_fields"]
    ranked: list[tuple] = []
    priority = {kind: index for index, kind in enumerate(CONFIG["preferred_symbol_kinds"])}
    fallback = len(priority)
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = row.get(fields["kind"])
        file_value = row.get(fields["file"])
        line = row.get(fields["line"])
        if not isinstance(file_value, str) or not file_value:
            continue
        if isinstance(line, bool) or not isinstance(line, int):
            continue
        rank = priority.get(kind, fallback) if isinstance(kind, str) else fallback
        ranked.append((rank, file_value, line, row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def build_symbol_text(name: str, row: dict, related_chinese: str) -> str:
    fields = CONFIG["index_fields"]
    kind = display(row.get(fields["kind"]))
    file_value = display(row.get(fields["file"]))
    module_path = display(row.get(fields["module_path"]))
    signature = display(row.get(fields["signature"]))
    parts = [
        f"[symbol] {name}",
        f"kind: {kind}",
        f"file: {file_value}",
        f"module_path: {module_path}",
        f"name_tokens: {split_name_tokens(name)}",
        f"signature: {signature}",
        "",
        "related_chinese:",
        related_chinese,
    ]
    return "\n".join(parts)


def build_symbol_docs(repo_root: Path, index: dict) -> tuple[list[dict], int]:
    fields = CONFIG["index_fields"]
    symbols = index.get(fields["symbols"]) if isinstance(index, dict) else None
    if not isinstance(symbols, dict):
        return [], 0

    cache = SourceCommentCache()
    docs: list[dict] = []
    with_chinese = 0
    for name in sorted(symbols):
        if not isinstance(name, str) or not name:
            continue
        rows = symbols.get(name)
        if not isinstance(rows, list):
            continue
        row = pick_definition(rows)
        if row is None:
            continue
        line = row.get(fields["line"])
        relative_file = row.get(fields["file"])
        fragments = cache.fragments_in_window(repo_root, str(relative_file), int(line))
        related = unique_join(fragments, CONFIG["symbol_comment_max_chars"])
        if related:
            with_chinese += 1
        module_path = row.get(fields["module_path"])
        if not isinstance(module_path, str) or not module_path:
            module_path = None
        kind = row.get(fields["kind"])
        docs.append({
            "doc_id": f"symbol:{name}",
            "kind": "symbol",
            "module_path": module_path,
            "text": build_symbol_text(name, row, related),
            "metadata": {
                "name": name,
                "symbol_kind": kind if isinstance(kind, str) else "",
                "file": relative_file if isinstance(relative_file, str) else "",
                "line": line if isinstance(line, int) and not isinstance(line, bool) else 0,
            },
        })
    return docs, with_chinese


def build_corpus(repo_root: Path, cards: list, index: dict) -> tuple[dict, dict]:
    module_docs, module_with_chinese = build_module_docs(repo_root, cards)
    symbol_docs, symbol_with_chinese = build_symbol_docs(repo_root, index)
    docs = module_docs + symbol_docs
    payload = {
        "meta": {
            "repo_root": os.fspath(repo_root),
            "total_docs": len(docs),
            "module_docs": len(module_docs),
            "symbol_docs": len(symbol_docs),
        },
        "docs": docs,
    }
    stats = {
        "module_docs": len(module_docs),
        "symbol_docs": len(symbol_docs),
        "total_docs": len(docs),
        "module_with_chinese": module_with_chinese,
        "symbol_with_chinese": symbol_with_chinese,
        "total_chars": sum(len(item.get("text") or "") for item in docs),
    }
    return payload, stats


# =============================================================================
# 4. 输出与统计区
# =============================================================================

def write_corpus(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_stats(stats: dict) -> None:
    print(f"module 文档数: {stats['module_docs']}")
    print(f"symbol 文档数: {stats['symbol_docs']}")
    print(f"总文档数: {stats['total_docs']}")
    print(f"有 chinese_comments 的 module 数量: {stats['module_with_chinese']}")
    print(f"有 related_chinese 的 symbol 数量: {stats['symbol_with_chinese']}")
    print(f"语料总字符数: {stats['total_chars']}")


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为仓库构建 RAG 检索语料 JSON")
    parser.add_argument("repo_root", help="仓库根目录路径")
    parser.add_argument("--module-cards", required=True, help="module_cards.json 路径")
    parser.add_argument("--symbol-index", required=True, help="symbol_index.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        print(f"仓库根目录不存在或不是目录: {repo_root}", file=sys.stderr)
        return 1

    cards = load_json(Path(args.module_cards))
    if not isinstance(cards, list):
        print(f"无法读取 module_cards.json: {args.module_cards}", file=sys.stderr)
        return 1

    index = load_json(Path(args.symbol_index))
    if not isinstance(index, dict):
        print(f"无法读取 symbol_index.json: {args.symbol_index}", file=sys.stderr)
        return 1

    payload, stats = build_corpus(repo_root, cards, index)
    write_corpus(payload, Path(args.output))
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
