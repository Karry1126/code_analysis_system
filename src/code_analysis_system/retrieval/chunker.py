"""按符号切块：函数/类/struct、文件摘要、JSON 顶层 key、markdown 小节。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..config import SNIPPET_MAX_CHARS
from .aliases import aliases_for_path

_CPP_START = re.compile(
    r"^(?:class|struct|enum)\s+([A-Za-z_]\w*)"
    r"|^(?:[\w:<>\*&]+(?:\s+|\s*::\s*))+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\("
)
_PY_START = re.compile(r"^(?:class|def)\s+([A-Za-z_]\w*)")
_MD_HEAD = re.compile(r"^#{1,3}\s+(.+)$")


def infer_module(relative_path: str) -> str:
    parts = Path(relative_path).parts
    for part in parts:
        if "event" in part.lower() or part.endswith("_serv") or part.endswith("_center"):
            return part
    return parts[0] if parts else ""


def infer_lang(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".py": "python",
        ".md": "markdown",
        ".json": "json",
        ".conf": "conf",
    }
    return mapping.get(suffix, "text")


def _clip_snippet(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[: SNIPPET_MAX_CHARS - 3] + "..."


def _chunk_id(relative_path: str, start_line: int, body: str) -> str:
    digest = hashlib.sha1(f"{relative_path}:{start_line}:{body}".encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def _make_chunk(
    relative_path: str,
    module: str,
    lang: str,
    kind: str,
    symbol: str,
    start_line: int,
    end_line: int,
    body: str,
) -> dict[str, Any]:
    aliases = aliases_for_path(relative_path)
    alias_text = " ".join(aliases)
    text = "\n".join(
        [
            f"module: {module}",
            f"file: {relative_path}",
            f"symbol: {symbol}",
            f"aliases: {alias_text}",
            body,
        ]
    )
    return {
        "chunk_id": _chunk_id(relative_path, start_line, body),
        "relative_path": relative_path,
        "module": module,
        "lang": lang,
        "kind": kind,
        "symbol": symbol,
        "start_line": start_line,
        "end_line": end_line,
        "snippet": _clip_snippet(body),
        "text": text,
    }


def _slice_by_starts(
    lines: list[str],
    starts: list[tuple[int, str, str]],
) -> list[tuple[int, int, str, str, str]]:
    if not starts:
        return []
    slices: list[tuple[int, int, str, str, str]] = []
    for index, (start, kind, symbol) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start - 1 : end])
        slices.append((start, end, kind, symbol, body))
    return slices


def chunk_cpp(relative_path: str, module: str, lang: str, lines: list[str]) -> list[dict[str, Any]]:
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        match = _CPP_START.search(stripped)
        if not match:
            continue
        if stripped.startswith("class") or stripped.startswith("struct") or stripped.startswith("enum"):
            symbol = match.group(1) or ""
            kind = "class"
        else:
            symbol = match.group(2) or ""
            kind = "func"
        if symbol:
            starts.append((line_number, kind, symbol))
    chunks = [
        _make_chunk(relative_path, module, lang, kind, symbol, start, end, body)
        for start, end, kind, symbol, body in _slice_by_starts(lines, starts)
    ]
    if not chunks and lines:
        chunks.append(
            _make_chunk(relative_path, module, lang, "file", Path(relative_path).stem, 1, len(lines), "\n".join(lines[:80]))
        )
    return chunks


def chunk_python(relative_path: str, module: str, lang: str, lines: list[str]) -> list[dict[str, Any]]:
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _PY_START.match(line)
        if match:
            kind = "class" if line.lstrip().startswith("class") else "func"
            starts.append((line_number, kind, match.group(1)))
    chunks = [
        _make_chunk(relative_path, module, lang, kind, symbol, start, end, body)
        for start, end, kind, symbol, body in _slice_by_starts(lines, starts)
    ]
    if not chunks and lines:
        chunks.append(
            _make_chunk(relative_path, module, lang, "file", Path(relative_path).stem, 1, len(lines), "\n".join(lines[:80]))
        )
    return chunks


def chunk_markdown(relative_path: str, module: str, lang: str, lines: list[str]) -> list[dict[str, Any]]:
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _MD_HEAD.match(line.strip())
        if match:
            starts.append((line_number, "doc", match.group(1).strip()))
    if not starts:
        return [
            _make_chunk(relative_path, module, lang, "doc", Path(relative_path).stem, 1, len(lines), "\n".join(lines))
        ]
    return [
        _make_chunk(relative_path, module, lang, kind, symbol, start, end, body)
        for start, end, kind, symbol, body in _slice_by_starts(lines, starts)
    ]


def chunk_json(relative_path: str, module: str, lang: str, text: str, lines: list[str]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [
            _make_chunk(relative_path, module, lang, "config", Path(relative_path).stem, 1, len(lines), text[:1200])
        ]
    if isinstance(payload, dict):
        chunks: list[dict[str, Any]] = []
        for key, value in list(payload.items())[:40]:
            body = json.dumps({key: value}, ensure_ascii=False)[:1200]
            chunks.append(_make_chunk(relative_path, module, lang, "config", str(key), 1, len(lines), body))
        return chunks or [
            _make_chunk(relative_path, module, lang, "config", Path(relative_path).stem, 1, len(lines), text[:1200])
        ]
    return [_make_chunk(relative_path, module, lang, "config", Path(relative_path).stem, 1, len(lines), text[:1200])]


def chunk_file(relative_path: str, text: str) -> list[dict[str, Any]]:
    path = Path(relative_path)
    module = infer_module(relative_path)
    lang = infer_lang(path)
    lines = text.splitlines()
    if lang == "cpp":
        chunks = chunk_cpp(relative_path, module, lang, lines)
    elif lang == "python":
        chunks = chunk_python(relative_path, module, lang, lines)
    elif lang == "markdown":
        chunks = chunk_markdown(relative_path, module, lang, lines)
    elif lang == "json":
        chunks = chunk_json(relative_path, module, lang, text, lines)
    else:
        chunks = [
            _make_chunk(relative_path, module, lang, "file", path.stem, 1, len(lines), "\n".join(lines[:80]))
        ]

    summary_body = "\n".join(
        [
            f"file {relative_path} module {module}",
            " ".join(chunk["symbol"] for chunk in chunks[:20]),
        ]
    )
    chunks.insert(
        0,
        _make_chunk(relative_path, module, lang, "summary", path.stem, 1, min(40, len(lines) or 1), summary_body),
    )
    return chunks
