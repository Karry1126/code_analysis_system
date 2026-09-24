"""混合检索：别名扩词 + BM25 + n-gram 向量，RRF 融合。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config import RETRIEVE_TOP_K, SNIPPET_MAX_CHARS
from .aliases import expand_query
from .bm25 import BM25, tokenize
from .embeddings import cosine, embed_text
from .indexer import build_index, load_chunks

_CHUNKS: list[dict[str, Any]] | None = None
_BM25: BM25 | None = None
_VECTORS: list[list[float]] | None = None
RRF_K = 60
CANDIDATE_LIMIT = 80


def _ensure_index() -> tuple[list[dict[str, Any]], BM25, list[list[float]]]:
    global _CHUNKS, _BM25, _VECTORS
    if _CHUNKS is not None and _BM25 is not None and _VECTORS is not None:
        return _CHUNKS, _BM25, _VECTORS
    chunks = load_chunks()
    if not chunks:
        print("[rag] index missing, building...")
        chunks = build_index()
    _CHUNKS = chunks
    _BM25 = BM25([tokenize(chunk.get("text") or "") for chunk in chunks])
    _VECTORS = [embed_text(chunk.get("text") or "") for chunk in chunks]
    return _CHUNKS, _BM25, _VECTORS


def reset_index_cache() -> None:
    global _CHUNKS, _BM25, _VECTORS
    _CHUNKS = None
    _BM25 = None
    _VECTORS = None


def _rrf(rank_lists: list[list[int]]) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] += 1.0 / (RRF_K + rank)
    return scores


def _clip_window(start_line: int, end_line: int) -> tuple[int, int]:
    start = max(1, start_line)
    end = max(start, end_line)
    if end - start + 1 > 40:
        end = start + 39
    return start, end


def _to_card(chunk: dict[str, Any], score: float, source: str) -> dict[str, Any]:
    start, end = _clip_window(int(chunk.get("start_line") or 1), int(chunk.get("end_line") or 1))
    snippet = chunk.get("snippet") or ""
    if len(snippet) > SNIPPET_MAX_CHARS:
        snippet = snippet[: SNIPPET_MAX_CHARS - 3] + "..."
    return {
        "relative_path": chunk.get("relative_path") or "",
        "symbol": chunk.get("symbol") or "",
        "start_line": start,
        "end_line": end,
        "snippet": snippet,
        "score": round(score, 4),
        "source": source,
    }


def retrieve(query: str, top_k: int = RETRIEVE_TOP_K) -> list[dict[str, Any]]:
    chunks, bm25, vectors = _ensure_index()
    if not chunks:
        return []

    expanded = expand_query(query)
    terms = [str(term) for term in expanded["terms"]]
    search_text = " ".join([query, *terms])
    query_tokens = tokenize(search_text)

    bm25_ranked = [index for index, _ in bm25.rank(query_tokens, CANDIDATE_LIMIT)]

    query_vec = embed_text(search_text)
    vector_scored = [(index, cosine(query_vec, vectors[index])) for index in range(len(vectors))]
    vector_scored.sort(key=lambda item: item[1], reverse=True)
    vector_ranked = [index for index, score in vector_scored[:CANDIDATE_LIMIT] if score > 0]

    fused = _rrf([bm25_ranked, vector_ranked])
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)

    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for chunk_index, score in ordered:
        chunk = chunks[chunk_index]
        key = (str(chunk.get("relative_path")), int(chunk.get("start_line") or 1))
        if key in seen:
            continue
        seen.add(key)
        in_bm25 = chunk_index in set(bm25_ranked[:20])
        in_vec = chunk_index in set(vector_ranked[:20])
        if in_bm25 and in_vec:
            source = "bm25|vector"
        elif in_bm25:
            source = "bm25"
        else:
            source = "vector"
        cards.append(_to_card(chunk, score, source))
        if len(cards) >= top_k:
            break
    return cards
