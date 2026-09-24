"""把 company_project 切块后落到 .rag_index/。索引是缓存，代码才是真相。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import COMPANY_CODE_REPO_PATH, RAG_INDEX_DIR
from .chunker import chunk_file

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
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".rag_index"}
_MAX_FILE_BYTES = 400_000


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS or part.startswith(".") for part in path.parts):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def build_index(repo_path: Path | None = None, index_dir: Path | None = None) -> list[dict[str, Any]]:
    root = (repo_path or COMPANY_CODE_REPO_PATH).resolve()
    target = index_dir or RAG_INDEX_DIR
    target.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    files = iter_text_files(root)
    print(f"[rag] indexing {len(files)} files from {root}")
    for path in files:
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        chunks.extend(chunk_file(relative, text))

    chunk_path = target / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (target / "meta.json").write_text(
        json.dumps({"chunk_count": len(chunks), "repo": str(root)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[rag] wrote {len(chunks)} chunks to {chunk_path}")
    return chunks


def load_chunks(index_dir: Path | None = None) -> list[dict[str, Any]]:
    chunk_path = (index_dir or RAG_INDEX_DIR) / "chunks.jsonl"
    if not chunk_path.exists():
        return []
    chunks: list[dict[str, Any]] = []
    with chunk_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
