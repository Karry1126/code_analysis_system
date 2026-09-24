"""扫描代码仓库，按目录结构为每个模块生成一张卡片，输出为 JSON 列表。

只做文件系统遍历和正则/通配匹配，不做 AST、ctags 或语法解析。
模块识别、入口文件、文档文件、signal 规则都在下方 CONFIG 区。
新增 signal：在 CONFIG["signal_rules"] 加一条即可，不必改框架代码。
抽不到的字段写成 None 或空列表，单个模块失败不会让整次扫描中断。

用法:
    python extract_module_cards.py <仓库根目录> [-o 输出路径]
默认输出 ./module_cards.json。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path


# =============================================================================
# 1. CONFIG 区
# =============================================================================

# serv_info.conf 的相对路径。模块识别和字段解析共用这一处，避免两处写死不一致。
_SERV_INFO_RELPATH = "conf/serv_info.conf"

CONFIG = {
    # 命中任意一条即视为模块。具体文件名/目录名只出现在这里。
    # kind:
    #   file_exists     相对路径上的文件存在
    #   dir_exists      相对路径上的目录存在
    #   dirname_suffix  目录名以 suffixes 中任一项结尾
    #   all / any       组合子规则
    "module_detect_rules": [
        {"kind": "file_exists", "relpath": _SERV_INFO_RELPATH},
        {
            "kind": "all",
            "checks": [
                {
                    "kind": "dirname_suffix",
                    "suffixes": ["_event_center", "_event_serv"],
                },
                {"kind": "dir_exists", "relpath": "src"},
            ],
        },
    ],
    # 相对模块根的 glob。** 可跨目录，* 不跨目录。大小写敏感。
    # 初始按本仓库进程入口 main.cpp；要加别的入口，在这里追加一行。
    "entry_file_patterns": [
        "**/main.cpp",
    ],
    # 每条规则求值为 bool。下游缺 key 时自行当 None，这里不补占位。
    # kind:
    #   dir_exists          模块根下的相对目录存在
    #   file_exists         模块根下的相对文件存在
    #   filename_anywhere   模块内任意深度，文件名 fnmatch 命中（大小写敏感）
    "signal_rules": {
        "has_src": {"kind": "dir_exists", "relpath": "src"},
        "has_global_serv": {"kind": "filename_anywhere", "pattern": "global_serv.h"},
        "has_event_process": {"kind": "filename_anywhere", "pattern": "event_process.h"},
        "has_rank_node": {"kind": "filename_anywhere", "pattern": "rank_node.h"},
        "has_conf_loader": {"kind": "filename_anywhere", "pattern": "*_conf.h"},
        "has_core_utils": {"kind": "dir_exists", "relpath": "src/core/utils"},
    },
    # 文件名 glob，大小写敏感。搜索位置见 doc_search_dirs。
    "doc_file_patterns": [
        "*.md",
        "*.txt",
    ],
    # 模块根只看直接子文件；doc/ 递归，避免漏掉子目录里的说明。
    "doc_search_dirs": [
        {"relpath": ".", "recursive": False},
        {"relpath": "doc", "recursive": True},
    ],
    # conf/ 下的直接子文件（不含子目录名）。
    "config_dir": "conf",
    "serv_info_relpath": _SERV_INFO_RELPATH,
    # 从 serv_info.conf 里尝试读取的键。卡片的 module_name、module_type 都用这里的值。
    "serv_info_fields": ["module_type", "module_name"],
    # 遍历时跳过这些目录名，不把它们当成模块候选，也不纳入 filename_anywhere。
    "skip_dir_names": [".git", ".svn", "__pycache__", "node_modules", ".venv"],
}


# =============================================================================
# 2. 工具函数区
# =============================================================================

def _skip_names() -> set[str]:
    return set(CONFIG["skip_dir_names"])


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


def _listdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _iter_files(root: Path):
    """遍历 root 下所有文件。跳过 CONFIG 里的目录名。权限错误则跳过该分支。"""
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


def _glob_to_regex(pattern: str) -> str:
    """把 posix glob 编译成整段匹配的正则。** 跨目录，* 与 ? 不跨目录。"""
    parts: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    parts.append("$")
    return "".join(parts)


def _compile_globs(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(_glob_to_regex(pattern)))
        except re.error:
            continue
    return compiled


def _filename_matches(filename: str, pattern: str) -> bool:
    try:
        return fnmatchcase(filename, pattern)
    except Exception:
        return False


# 假设 serv_info.conf 是宽松的键值文本，不要求全仓库格式一致：
# - 允许 [SECTION] 段头，解析时忽略
# - 允许空行，以及以 # 或 ; 开头的整行注释
# - 键值分隔符是 = 或 :，两侧可有空白；键可带引号
# - 值可带成对单/双引号；未加引号时，去掉行尾的 " #" / " ;" 注释
# - 只收集 CONFIG["serv_info_fields"] 里的键，大小写不敏感
# - 同一键出现多次时，保留第一次非空值
# - 文件缺失、编码失败、键不存在，对应值留 None，不抛异常
_SERV_INFO_KV = re.compile(
    r"""^\s*["']?(?P<key>[A-Za-z_][\w]*)["']?\s*[=:]\s*(?P<value>.*?)\s*$"""
)
_SERV_INFO_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin-1")


def _clean_serv_value(raw: str) -> str:
    value = raw.strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return re.sub(r"\s+[#;].*$", "", value).strip()


def _read_text(path: Path) -> str | None:
    for encoding in _SERV_INFO_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def read_serv_info(module_dir: Path) -> dict[str, str | None]:
    """读取 serv_info.conf，返回配置区声明的字段。失败时各字段为 None。"""
    fields = list(CONFIG["serv_info_fields"])
    found: dict[str, str | None] = {field: None for field in fields}
    wanted = {field.lower(): field for field in fields}
    path = module_dir / CONFIG["serv_info_relpath"]
    text = _read_text(path) if _is_file(path) else None
    if not text:
        return found

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        matched = _SERV_INFO_KV.match(stripped)
        if not matched:
            continue
        field = wanted.get(matched.group("key").lower())
        if field is None or found[field]:
            continue
        value = _clean_serv_value(matched.group("value"))
        if value:
            found[field] = value
    return found


def match_detect_rule(directory: Path, rule: dict) -> bool:
    kind = rule.get("kind")
    if kind == "file_exists":
        return _is_file(directory / str(rule.get("relpath", "")))
    if kind == "dir_exists":
        return _is_dir(directory / str(rule.get("relpath", "")))
    if kind == "dirname_suffix":
        name = directory.name
        return any(name.endswith(str(suffix)) for suffix in rule.get("suffixes", []))
    if kind == "all":
        checks = rule.get("checks", [])
        return bool(checks) and all(match_detect_rule(directory, check) for check in checks)
    if kind == "any":
        return any(match_detect_rule(directory, check) for check in rule.get("checks", []))
    return False


def is_module_dir(directory: Path) -> bool:
    try:
        return any(
            match_detect_rule(directory, rule)
            for rule in CONFIG["module_detect_rules"]
        )
    except Exception:
        return False


def extract_module_path(repo_root: Path, module_dir: Path) -> str:
    try:
        return _rel_posix(module_dir, repo_root)
    except Exception:
        return ""


def extract_module_name(serv_info: dict[str, str | None]) -> str | None:
    value = serv_info.get("module_name")
    return value if isinstance(value, str) and value else None


def extract_module_type(serv_info: dict[str, str | None]) -> str | None:
    value = serv_info.get("module_type")
    return value if isinstance(value, str) and value else None


def extract_top_dirs(module_dir: Path) -> list[str]:
    names = [entry.name for entry in _listdir(module_dir) if _is_dir(entry)]
    return sorted(names)


def extract_config_files(module_dir: Path) -> list[str]:
    config_dir = module_dir / str(CONFIG["config_dir"])
    if not _is_dir(config_dir):
        return []
    names = [entry.name for entry in _listdir(config_dir) if _is_file(entry)]
    return sorted(names)


def extract_doc_files(module_dir: Path) -> list[str]:
    patterns = list(CONFIG["doc_file_patterns"])
    found: set[str] = set()
    for spec in CONFIG["doc_search_dirs"]:
        base = module_dir / str(spec.get("relpath", "."))
        if not _is_dir(base):
            continue
        recursive = bool(spec.get("recursive", False))
        candidates = _iter_files(base) if recursive else (
            entry for entry in _listdir(base) if _is_file(entry)
        )
        for path in candidates:
            if not any(_filename_matches(path.name, pattern) for pattern in patterns):
                continue
            try:
                found.add(_rel_posix(path, module_dir))
            except Exception:
                continue
    return sorted(found)


def extract_entry_files(module_dir: Path) -> list[str]:
    compiled = _compile_globs(list(CONFIG["entry_file_patterns"]))
    if not compiled:
        return []
    found: set[str] = set()
    for path in _iter_files(module_dir):
        try:
            relative = _rel_posix(path, module_dir)
        except Exception:
            continue
        if any(pattern.search(relative) for pattern in compiled):
            found.add(relative)
    return sorted(found)


def _eval_signal(module_dir: Path, rule: dict, filenames: set[str] | None) -> tuple[bool, set[str] | None]:
    kind = rule.get("kind")
    if kind == "dir_exists":
        return _is_dir(module_dir / str(rule.get("relpath", ""))), filenames
    if kind == "file_exists":
        return _is_file(module_dir / str(rule.get("relpath", ""))), filenames
    if kind == "filename_anywhere":
        if filenames is None:
            filenames = {path.name for path in _iter_files(module_dir)}
        pattern = str(rule.get("pattern", ""))
        return any(_filename_matches(name, pattern) for name in filenames), filenames
    return False, filenames


def extract_signals(module_dir: Path) -> dict[str, bool]:
    signals: dict[str, bool] = {}
    filenames: set[str] | None = None
    for name, rule in CONFIG["signal_rules"].items():
        try:
            value, filenames = _eval_signal(module_dir, rule, filenames)
            signals[name] = bool(value)
        except Exception:
            signals[name] = False
    return signals


# =============================================================================
# 3. 主流程区
# =============================================================================

def iter_module_dirs(repo_root: Path):
    skip = _skip_names()

    def _onerror(_err: OSError) -> None:
        return None

    for dirpath, dirnames, _filenames in os.walk(repo_root, followlinks=False, onerror=_onerror):
        dirnames[:] = [name for name in dirnames if name not in skip]
        directory = Path(dirpath)
        if is_module_dir(directory):
            yield directory


def build_card(repo_root: Path, module_dir: Path) -> dict:
    try:
        serv_info = read_serv_info(module_dir)
    except Exception:
        serv_info = {}

    def _take(fn, default, *args):
        try:
            return fn(*args)
        except Exception:
            return default

    return {
        "module_path": _take(extract_module_path, "", repo_root, module_dir),
        "module_name": _take(extract_module_name, None, serv_info),
        "module_type": _take(extract_module_type, None, serv_info),
        "top_dirs": _take(extract_top_dirs, [], module_dir),
        "config_files": _take(extract_config_files, [], module_dir),
        "doc_files": _take(extract_doc_files, [], module_dir),
        "entry_files": _take(extract_entry_files, [], module_dir),
        "signals": _take(extract_signals, {}, module_dir),
    }


def build_cards(repo_root: Path) -> list[dict]:
    cards = [build_card(repo_root, module_dir) for module_dir in iter_module_dirs(repo_root)]
    cards.sort(key=lambda card: card.get("module_path") or "")
    return cards


# =============================================================================
# 4. 输出与统计区
# =============================================================================

def write_cards(cards: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(cards, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _print_sample(paths: list[str], limit: int = 10) -> None:
    for path in paths[:limit]:
        print(f"  - {path}")


def print_stats(cards: list[dict]) -> None:
    missing_type = [card["module_path"] for card in cards if card.get("module_type") is None]
    no_signals = [
        card["module_path"]
        for card in cards
        if not any((card.get("signals") or {}).values())
    ]
    print(f"识别到的模块总数: {len(cards)}")
    print(f"module_type 为 None 的模块数: {len(missing_type)}")
    _print_sample(missing_type)
    print(f"signals 全为 False 的模块数: {len(no_signals)}")
    _print_sample(no_signals)


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描代码仓库并生成模块卡片 JSON")
    parser.add_argument("repo_root", help="仓库根目录路径")
    parser.add_argument(
        "-o",
        "--output",
        default="./module_cards.json",
        help="输出 JSON 路径，默认 ./module_cards.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        print(f"仓库根目录不存在或不是目录: {repo_root}", file=sys.stderr)
        return 1
    cards = build_cards(repo_root)
    write_cards(cards, Path(args.output))
    print_stats(cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
