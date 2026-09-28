import re   
from pathlib import Path
from typing import Any

from .framework import ToolRegistry, tool
from .config import (
    COMPANY_CODE_REPO_PATH,
    MAX_FILENAME_MATCHES,
    MAX_LIST_ITEMS,
    MAX_READ_LINES,
    MAX_SEARCH_MATCHES,
    READ_PREVIEW_LINES,
    RETRIEVE_TOP_K,
    RETRIEVE_TOP_K_MAX,
    SNIPPET_MAX_CHARS,  
)

def resolve_safe_path(relative_path: str) -> Path:
    """
    将相对路径解析为 workspace 下的安全路径。
    """
    cleaned_path = relative_path.strip().replace("\\", "/")
    if not cleaned_path:
        raise ValueError("relative_path 不能为空。")

    relative = Path(cleaned_path)
    if relative.is_absolute():
        raise ValueError("relative_path 不能是绝对路径。")

    target = (COMPANY_CODE_REPO_PATH / relative).resolve()
    base_dir = COMPANY_CODE_REPO_PATH.resolve()

    if target != base_dir and base_dir not in target.parents:
        raise ValueError("不允许访问 workspace 目录之外的路径。")

    return target


def resolve_safe_dir(relative_dir: str) -> Path:
    """解析目录路径，'.' 表示工作区根目录。"""
    if relative_dir.strip() in {"", "."}:
        return COMPANY_CODE_REPO_PATH.resolve()
    return resolve_safe_path(relative_dir)


def _is_workspace_root(target_dir: Path) -> bool:
    return target_dir.resolve() == COMPANY_CODE_REPO_PATH.resolve()


def _to_relative(path: Path) -> str:
    return str(path.relative_to(COMPANY_CODE_REPO_PATH)).replace("\\", "/")


def _normalize_line(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)

_SKIP_SUFFIXES = {
    ".o",
    ".so",
    ".a",
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".exe",
    ".dll",
    ".bin",
    ".zip",
}

# 一期宽度入口：中文业务名 -> 路径/符号关键词。
_MODULE_ALIASES: dict[str, list[str]] = {
    "生日礼包": ["birthday_gift", "birthday"],
    "生日": ["birthday"],
    "绝地飞驰": ["hurtle_across", "hurtle"],
    "掠夺": ["plunder", "hurtle"],
    "排行": ["rank", "RankNode"],
    "模块": ["event_module_list", "event_center"],
}


def _clip_snippet(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[: SNIPPET_MAX_CHARS - 3] + "..."


def _expand_retrieve_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip()
        if len(cleaned) < 2:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(cleaned)

    for alias, expansions in _MODULE_ALIASES.items():
        if alias in query:
            for item in expansions:
                add(item)

    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        add(token)
    return terms


def _is_skippable_file(path: Path) -> bool:
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return any(part in {".git", "__pycache__", "node_modules"} for part in path.parts)


def _guess_symbol(line: str, relative_path: str) -> str:
    match = re.search(r"\b(?:class|struct|def|bool|void|TVOID|TINT64)\s+([A-Za-z_][A-Za-z0-9_:]*)", line)
    if match:
        return match.group(1)
    return Path(relative_path).stem


def _window_around(lines: list[str], line_number: int) -> tuple[int, int, str]:
    start = max(1, line_number - 2)
    end = min(len(lines), line_number + 8)
    snippet = _clip_snippet("\n".join(lines[start - 1 : end]))
    return start, end, snippet


def _candidate_dirs(terms: list[str]) -> list[Path]:
    scored: list[tuple[int, Path]] = []
    root = COMPANY_CODE_REPO_PATH.resolve()
    lowered_terms = [term.lower() for term in terms]
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(root)
        if len(relative.parts) > 4:
            continue
        if any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in relative.parts):
            continue
        blob = str(relative).replace("\\", "/").lower()
        score = sum(1 for term in lowered_terms if term in blob)
        if score:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    return [path for _, path in scored[:5]]


def _make_card(
    relative_path: str,
    start_line: int,
    end_line: int,
    snippet: str,
    score: float,
    source: str,
    symbol: str = "",
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "symbol": symbol,
        "start_line": start_line,
        "end_line": end_line,
        "snippet": snippet,
        "score": round(score, 3),
        "source": source,
    }


def _retrieve_by_alias_and_grep(query: str, top_k: int) -> list[dict[str, Any]]:
    terms = _expand_retrieve_terms(query)
    cards: dict[tuple[str, int], dict[str, Any]] = {}

    def add_card(card: dict[str, Any]) -> None:
        key = (card["relative_path"], card["start_line"])
        existing = cards.get(key)
        if existing is None or card["score"] > existing["score"]:
            cards[key] = card

    if not terms:
        terms = [part for part in query.split() if len(part) >= 2]

    lowered_terms = [term.lower() for term in terms]
    candidate_dirs = _candidate_dirs(terms)

    filename_hits = 0
    search_roots = candidate_dirs or [COMPANY_CODE_REPO_PATH.resolve()]
    if not candidate_dirs:
        # 没有模块目录命中时，只按文件名收窄，绝不做全库正文扫描。
        for path in COMPANY_CODE_REPO_PATH.resolve().rglob("*"):
            if not path.is_file() or _is_skippable_file(path):
                continue
            name = path.name.lower()
            if not any(term in name for term in lowered_terms):
                continue
            relative_path = _to_relative(path)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            hit_line = 1
            for line_number, line in enumerate(lines, start=1):
                if any(term in line.lower() or term in line for term in terms):
                    hit_line = line_number
                    break
            start, end, snippet = _window_around(lines, hit_line) if lines else (1, 1, "")
            add_card(
                _make_card(
                    relative_path,
                    start,
                    end,
                    snippet,
                    1.5,
                    "alias",
                    Path(relative_path).stem,
                )
            )
            filename_hits += 1
            if filename_hits >= MAX_FILENAME_MATCHES:
                break
    else:
        for directory in candidate_dirs:
            dir_blob = str(directory).replace("\\", "/").lower()
            dir_bonus = 2.0 if any(term in dir_blob for term in lowered_terms) else 0.0
            grep_hits = 0
            for path in sorted(directory.rglob("*"), key=lambda item: str(item).lower()):
                if not path.is_file() or _is_skippable_file(path):
                    continue
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                relative_path = _to_relative(path)
                name_hit = any(term in path.name.lower() for term in lowered_terms)
                for line_number, line in enumerate(lines, start=1):
                    if not any(term in line or term.lower() in line.lower() for term in terms):
                        continue
                    start, end, snippet = _window_around(lines, line_number)
                    score = 1.0 + dir_bonus
                    if name_hit:
                        score += 0.4
                    source = "alias|grep" if dir_bonus else "grep"
                    add_card(
                        _make_card(
                            relative_path,
                            start,
                            end,
                            snippet,
                            score,
                            source,
                            _guess_symbol(line, relative_path),
                        )
                    )
                    grep_hits += 1
                    if grep_hits >= MAX_SEARCH_MATCHES:
                        break
                if grep_hits >= MAX_SEARCH_MATCHES:
                    break

    ranked = sorted(cards.values(), key=lambda item: (-item["score"], item["relative_path"]))
    return ranked[:top_k]

@tool(
    description=(
        "List one directory level under workspace. "
        "Use a specific subdirectory after you know the module path. "
        "Do not use '.' as a substitute for searching file contents."
    ),
    parameter_descriptions={
        "relative_dir": "Relative directory under the workspace. '.' lists the workspace root (one level, capped).",
    },
)
def list_files(relative_dir: str) -> dict[str, Any]:
    """列出工作区某一层目录中的文件。"""
    try:
        target_dir = resolve_safe_dir(relative_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_dir": relative_dir}

    if not target_dir.exists():
        return {"ok": False, "error": "目录不存在。", "path": str(target_dir)}
    if not target_dir.is_dir():
        return {"ok": False, "error": "目标路径不是目录。", "path": str(target_dir)}

    all_items = [
        {"name": item.name, "type": "dir" if item.is_dir() else "file"}
        for item in sorted(target_dir.iterdir(), key=lambda p: p.name.lower())
    ]
    truncated = len(all_items) > MAX_LIST_ITEMS
    result: dict[str, Any] = {
        "ok": True,
        "path": str(target_dir),
        "items": all_items[:MAX_LIST_ITEMS],
        "item_count": len(all_items),
        "truncated": truncated,
    }
    if _is_workspace_root(target_dir):
        result["hint"] = "根目录列表不能代替检索。请缩小到具体模块目录，或先按文件名搜索。"
    return result


@tool(
    description=(
        "Read a line window from a workspace text file. "
        "Always pass start_line and end_line after locating the code. "
        "Without line numbers this returns only a file-head preview, not the full file."
    ),
    parameter_descriptions={
        "relative_path": "Relative file path under the workspace.",
        "start_line": "1-based start line. Omit together with end_line to preview the file head.",
        "end_line": "1-based end line inclusive. Window is capped at MAX_READ_LINES.",
    },
)
def read_text_file(
    relative_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """读取工作区文本文件的指定行窗口；不给行号则只返回文件头。"""
    try:
        target_path = resolve_safe_path(relative_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_path": relative_path}

    if not target_path.exists():
        return {"ok": False, "error": "文件不存在。", "path": str(target_path)}
    if not target_path.is_file():
        return {"ok": False, "error": "目标路径不是文件。", "path": str(target_path)}

    try:
        lines = target_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"ok": False, "error": "文件不是可解码的 UTF-8 文本。", "path": str(target_path)}

    total_lines = len(lines)
    start = _normalize_line(start_line)
    end = _normalize_line(end_line)

    if (start is None) != (end is None):
        return {
            "ok": False,
            "error": "start_line 和 end_line 必须同时提供。",
            "relative_path": relative_path,
            "total_lines": total_lines,
        }

    if start is None:
        preview_end = min(READ_PREVIEW_LINES, total_lines)
        preview = "\n".join(lines[:preview_end])
        return {
            "ok": True,
            "path": str(target_path),
            "preview": True,
            "total_lines": total_lines,
            "start_line": 1 if total_lines else 0,
            "end_line": preview_end,
            "content": preview,
            "characters_read": len(preview),
            "hint": "未指定行号，只返回文件头。请根据搜索结果的 line_number 精读。",
            "context_updates": {"last_read_relative_path": relative_path},
        }

    if start < 1 or end < 1:
        return {
            "ok": False,
            "error": "start_line / end_line 必须 >= 1。",
            "relative_path": relative_path,
            "total_lines": total_lines,
        }
    if start > end:
        start, end = end, start

    clipped = False
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1
        clipped = True

    start = min(start, total_lines) if total_lines else 1
    end = min(end, total_lines) if total_lines else 0
    window = lines[start - 1 : end] if total_lines else []
    content = "\n".join(window)
    return {
        "ok": True,
        "path": str(target_path),
        "preview": False,
        "total_lines": total_lines,
        "start_line": start,
        "end_line": end,
        "truncated": clipped,
        "content": content,
        "characters_read": len(content),
        "context_updates": {
            "last_read_relative_path": relative_path,
            "last_retrieve": [
                {
                    "relative_path": relative_path,
                    "start_line": start,
                    "end_line": end,
                }
            ],
        },
    }


@tool(
    description=(
        "Search exact text inside a module directory. "
        "Do not pass '.' as relative_dir. "
        "Use this for known English identifiers after you have narrowed to a module."
    ),
    parameter_descriptions={
        "query": "Exact substring to search for.",
        "relative_dir": "Module subdirectory under the workspace. Workspace root is rejected.",
    },
)
def search_text(query: str, relative_dir: str) -> dict[str, Any]:
    """在已缩小的目录中搜索文本，命中条数有上限。"""
    try:
        target_dir = resolve_safe_dir(relative_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_dir": relative_dir}

    if not target_dir.exists() or not target_dir.is_dir():
        return {"ok": False, "error": "搜索目录不存在或不是目录。", "path": str(target_dir)}

    if _is_workspace_root(target_dir):
        return {
            "ok": False,
            "error": (
                "禁止对仓库根做全文搜索。请先用 retrieve_code 定位模块，"
                "或把 relative_dir 缩到具体子目录。"
            ),
            "relative_dir": relative_dir,
        }

    matches: list[dict[str, Any]] = []
    truncated = False
    for path in sorted(target_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        relative_path = _to_relative(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query not in line:
                continue
            matches.append(
                {
                    "relative_path": relative_path,
                    "line_number": line_number,
                    "line": line,
                }
            )
            if len(matches) >= MAX_SEARCH_MATCHES:
                truncated = True
                break
        if truncated:
            break

    last_retrieve = [
        {
            "relative_path": item["relative_path"],
            "start_line": item["line_number"],
            "end_line": item["line_number"],
        }
        for item in matches[:RETRIEVE_TOP_K]
    ]
    return {
        "ok": True,
        "query": query,
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "context_updates": {
            "last_search_query": query,
            "last_retrieve": last_retrieve,
        },
    }


@tool(
    description=(
        "Search files by name (partial match). "
        "Prefer this to locate a module before reading. Results are capped."
    ),
    parameter_descriptions={
        "name_query": "Filename or partial filename to search.",
        "relative_dir": "Relative directory under the workspace. '.' searches from the workspace root.",
    },
)
def search_files_by_name(name_query: str, relative_dir: str) -> dict[str, Any]:
    """按文件名搜索工作区中的文件，结果有上限。"""
    try:
        target_dir = resolve_safe_dir(relative_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_dir": relative_dir}

    if not target_dir.exists() or not target_dir.is_dir():
        return {"ok": False, "error": "搜索目录不存在或不是目录。", "path": str(target_dir)}

    normalized_query = name_query.lower()
    matches: list[str] = []
    truncated = False
    for path in sorted(target_dir.rglob("*"), key=lambda p: str(p).lower()):
        if path.is_file() and normalized_query in path.name.lower():
            matches.append(_to_relative(path))
            if len(matches) >= MAX_FILENAME_MATCHES:
                truncated = True
                break

    last_retrieve = [
        {"relative_path": item, "start_line": 1, "end_line": READ_PREVIEW_LINES}
        for item in matches[:RETRIEVE_TOP_K]
    ]
    return {
        "ok": True,
        "name_query": name_query,
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "context_updates": {
            "last_file_search_query": name_query,
            "last_retrieve": last_retrieve,
        },
    }


@tool(
    description=(
        "Wide retrieval for Chinese business questions or unknown file paths. "
        "Returns top-k location cards (path, lines, snippet), not full files. "
        "After this, call read_text_file with the returned start_line and end_line."
    ),
    parameter_descriptions={
        "query": "Natural-language or identifier query, e.g. 生日礼包命中逻辑 or RankNode.",
        "top_k": "How many location cards to return. Default 8, max 12.",
    },
)
def retrieve_code(query: str, top_k: int = RETRIEVE_TOP_K) -> dict[str, Any]:
    """宽度定位：别名 + 有界 grep，返回 Top-K 卡片。二期换成混合检索时保持此 JSON。"""
    cleaned = query.strip()
    if not cleaned:
        return {"ok": False, "error": "query 不能为空。"}

    limited = max(1, min(int(top_k), RETRIEVE_TOP_K_MAX))
    matches = _retrieve_by_alias_and_grep(cleaned, limited)
    last_retrieve = [
        {
            "relative_path": item["relative_path"],
            "start_line": item["start_line"],
            "end_line": item["end_line"],
        }
        for item in matches
    ]
    return {
        "ok": True,
        "query": cleaned,
        "matches": matches,
        "match_count": len(matches),
        "context_updates": {
            "last_search_query": cleaned,
            "last_retrieve": last_retrieve,
        },
    }


@tool(
    description=(
        "Replace exact text inside a file under workspace. "
        "Use this for small, precise code edits after reading the file."
    ),
    parameter_descriptions={
        "relative_path": "Relative file path under the workspace.",
        "old_text": "Exact old text to replace.",
        "new_text": "Exact new text to write.",
        "expected_occurrences": "Expected number of occurrences for safety.",
    },
)
def replace_text_in_file(
    relative_path: str,
    old_text: str,
    new_text: str,
    expected_occurrences: int,
) -> dict[str, Any]:
    """
    对文件做精确字符串替换。

    它要求模型给出：
    - 要改哪个文件
    - 旧文本是什么
    - 新文本是什么
    - 预期替换次数是多少

    `expected_occurrences` 的作用是做最基础的安全保护，
    避免模型误替换了太多地方。
    """
    try:
        target_path = resolve_safe_path(relative_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_path": relative_path}

    if not target_path.exists():
        return {"ok": False, "error": "文件不存在。", "path": str(target_path)}

    original = target_path.read_text(encoding="utf-8")
    occurrences = original.count(old_text)

    if occurrences != expected_occurrences:
        return {
            "ok": False,
            "error": (
                f"目标文本出现次数不符合预期：expected_occurrences={expected_occurrences}, "
                f"actual_occurrences={occurrences}"
            ),
            "path": str(target_path),
        }

    updated = original.replace(old_text, new_text)
    target_path.write_text(updated, encoding="utf-8")

    return {
        "ok": True,
        "path": str(target_path),
        "replaced_occurrences": occurrences,
        "context_updates": {"last_modified_relative_path": relative_path},
    }


@tool(
    description="Write a complete file under workspace.",
    parameter_descriptions={
        "relative_path": "Relative file path under the workspace.",
        "content": "Complete file content to write.",
        "overwrite": "Whether to overwrite an existing file.",
    },
)
def write_text_file(relative_path: str, content: str, overwrite: bool) -> dict[str, Any]:
    """
    在工作区写入完整文件内容。

    这个工具比 replace_text_in_file 更重，
    一般留给“需要重写完整文件”或“创建新文件”的情况。
    """
    try:
        target_path = resolve_safe_path(relative_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "relative_path": relative_path}

    target_path.parent.mkdir(parents=True, exist_ok=True)

    existed_before = target_path.exists()
    if existed_before and not overwrite:
        return {"ok": False, "error": "文件已存在，且 overwrite 为 False。", "path": str(target_path)}

    target_path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "path": str(target_path),
        "created": not existed_before,
        "overwritten": existed_before,
        "characters_written": len(content),
        "context_updates": {"last_modified_relative_path": relative_path},
    }


def register_coding_tools(registry: ToolRegistry) -> None:
    """
    注册 coding agent 使用的工具。

    这里故意把工具分成两类：
    - 观察型工具：retrieve_code / list / scoped search / ranged read
    - 修改型工具：replace / write
    """
    registry.register_many(
        retrieve_code,
        list_files,
        read_text_file,
        search_text,
        search_files_by_name,
        replace_text_in_file,
        write_text_file,
    )