"""术语解析工具：resolve_term。

用分类型独立索引做混合检索：module 与 symbol 各有一套 BM25 和一套 FAISS。
两侧分别召回、分别做 RRF，再按配额合并为模块/符号候选，不调用 LLM。
索引目录在 config.py 的 RAG_INDEX_DIR。
分类型文件：bm25_module_corpus.pkl、bm25_symbol_corpus.pkl、
docs_module.pkl、docs_symbol.pkl、faiss_module/、faiss_symbol/。
旧混检路径常量仍保留，不参与加载；签名变化后会自动重建分类型索引。
模型名、正则、k 值在下方 CONFIG 区。
本模块导入时一次性加载。文件缺失或依赖不可用时工具返回 ok=false，不抛异常。
"""

from __future__ import annotations
import json
import hashlib
import os
import pickle
import re
import jieba
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from .config import RAG_CORPUS_PATH, RAG_INDEX_DIR
from .framework import ToolRegistry, tool

# =============================================================================
# CONFIG 区
# =============================================================================

EMBEDDING_MODEL_NAME = "bge-m3"
ENGLISH_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
CHINESE_TOKEN_PATTERN = r"[\u4e00-\u9fff]+"
CAMEL_SPLIT_PATTERN = r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)"
EMBED_BATCH_SIZE = 16

INDEX_SIGNATURE_PATH = RAG_INDEX_DIR / "index_meta.json"
BM25_CORPUS_PATH = RAG_INDEX_DIR / "bm25_corpus.pkl"
FAISS_INDEX_PATH = RAG_INDEX_DIR / "index.faiss"
FAISS_DOCSTORE_PATH = RAG_INDEX_DIR / "index.pkl"
DOCS_PATH = RAG_INDEX_DIR / "docs.pkl"

BM25_MODULE_CORPUS_PATH = RAG_INDEX_DIR / "bm25_module_corpus.pkl"
BM25_SYMBOL_CORPUS_PATH = RAG_INDEX_DIR / "bm25_symbol_corpus.pkl"
FAISS_MODULE_DIR = RAG_INDEX_DIR / "faiss_module"
FAISS_SYMBOL_DIR = RAG_INDEX_DIR / "faiss_symbol"
DOCS_MODULE_PATH = RAG_INDEX_DIR / "docs_module.pkl"
DOCS_SYMBOL_PATH = RAG_INDEX_DIR / "docs_symbol.pkl"

PER_SIDE_K = 20  # 每侧每路召回条数
RRF_K = 60
SCORE_DECIMALS = 4
DEFAULT_TOP_K = 10

_ENGLISH_TOKEN_RE = re.compile(ENGLISH_TOKEN_PATTERN)
_CHINESE_TOKEN_RE = re.compile(CHINESE_TOKEN_PATTERN)
_CAMEL_SPLIT_RE = re.compile(CAMEL_SPLIT_PATTERN)

def tokenize_mixed(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _ENGLISH_TOKEN_RE.finditer(text):
        word = match.group(0)
        for part in word.split("_"):
            expanded = _CAMEL_SPLIT_RE.findall(part) or [part]
            tokens.extend(piece.lower() for piece in expanded if piece)
    for match in _CHINESE_TOKEN_RE.finditer(text):
        tokens.extend(jieba.cut(match.group(0)))
    return [token.strip().lower() for token in tokens if token.strip()]

@dataclass
class RagIndex:
    module_records: list[LCDocument]
    symbol_records: list[LCDocument]
    bm25_module: Any
    bm25_symbol: Any
    faiss_module: Any
    faiss_symbol: Any

def _gen_signature()-> dict:
    with open(RAG_CORPUS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "embeddings_model": EMBEDDING_MODEL_NAME,
        "english_token_pattern": ENGLISH_TOKEN_PATTERN,
        "chinese_token_pattern": CHINESE_TOKEN_PATTERN,
        "camel_split_pattern": CAMEL_SPLIT_PATTERN,
        "index_layout": "split_by_kind_v1",
    }

def _load_signature() -> dict | None:
    if not os.path.exists(INDEX_SIGNATURE_PATH):
        return None
    try:
        with open (INDEX_SIGNATURE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

def _is_document_list(payload: object) -> bool:
    return isinstance(payload, list) and all(
        isinstance(item, LCDocument) for item in payload
    )

def _judge_can_reload(sign: dict) -> bool:
    return (
        sign == _load_signature()
        and os.path.exists(BM25_MODULE_CORPUS_PATH)
        and os.path.exists(BM25_SYMBOL_CORPUS_PATH)
        and os.path.exists(DOCS_MODULE_PATH)
        and os.path.exists(DOCS_SYMBOL_PATH)
        and os.path.exists(FAISS_MODULE_DIR / "index.faiss")
        and os.path.exists(FAISS_SYMBOL_DIR / "index.faiss")
    )

def _load_corpus_docs(corpus_path: Path) -> list[dict] | None:
    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    docs = payload.get("docs")
    if not isinstance(docs, list):
        return None
    return docs

def to_internal_record(doc: dict) -> LCDocument:
    extra = doc.get("metadata")
    if not isinstance(extra, dict):
        extra = {}
    return LCDocument(
        page_content= doc["text"],
        metadata={
            "doc_id": doc["doc_id"],
            "kind": doc["kind"],
            "module_path": doc["module_path"],
            "extra": extra,
        }
    )

def build_records(raw_docs: list[dict]) -> list[LCDocument]:
    records: list[LCDocument] = []
    for doc in raw_docs:
        if not isinstance(doc, dict):
            continue
        if "doc_id" not in doc or "text" not in doc:
            continue
        records.append(to_internal_record(doc))
    return records

def split_records_by_kind(records: list[LCDocument]) -> tuple[list[LCDocument], list[LCDocument]]:
    """按 metadata['kind'] 拆分为 (module_records, symbol_records)。"""
    module_records: list[LCDocument] = []
    symbol_records: list[LCDocument] = []
    for record in records:
        kind = record.metadata.get("kind")
        if kind == "module":
            module_records.append(record)
        elif kind == "symbol":
            symbol_records.append(record)
    return module_records, symbol_records

def build_bm25_corpus(records: list[LCDocument]) -> list[list[str]]:
    return [tokenize_mixed(str(record.page_content or "")) for record in records]

def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def write_pickle(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

def embed_documents_batched(
    embeddings_model: OllamaEmbeddings,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Ollama 一次 tokenize 过多文本会失败，按批调用 embed。"""
    vectors: list[list[float]] = []
    total = len(texts)
    if total == 0:
        return vectors
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embeddings_model.embed_documents(batch))
        done = min(start + batch_size, total)
        if done == total or done % (batch_size * 10) == 0:
            print(f"embedding {done}/{total}")
    return vectors

def build_faiss_index(
    records: list[LCDocument],
    embeddings_model: OllamaEmbeddings,
) -> FAISS:
    texts = [str(record.page_content or "") for record in records]
    metadatas = [dict(record.metadata) for record in records]
    vectors = embed_documents_batched(embeddings_model, texts)
    text_embedding_pairs = list(zip(texts, vectors, strict=True))
    return FAISS.from_embeddings(
        text_embedding_pairs,
        embeddings_model,
        metadatas=metadatas,
    )

def build_split_indexes(
    records: list[LCDocument],
    embeddings_model: OllamaEmbeddings,
) -> RagIndex:
    module_records, symbol_records = split_records_by_kind(records)

    # BM25 module
    module_corpus = build_bm25_corpus(module_records)
    bm25_module = BM25Okapi(module_corpus)
    write_pickle(module_corpus, BM25_MODULE_CORPUS_PATH)

    # BM25 symbol
    symbol_corpus = build_bm25_corpus(symbol_records)
    bm25_symbol = BM25Okapi(symbol_corpus)
    write_pickle(symbol_corpus, BM25_SYMBOL_CORPUS_PATH)

    # FAISS module
    faiss_module = build_faiss_index(module_records, embeddings_model)
    FAISS_MODULE_DIR.mkdir(parents=True, exist_ok=True)
    faiss_module.save_local(FAISS_MODULE_DIR)

    # FAISS symbol
    faiss_symbol = build_faiss_index(symbol_records, embeddings_model)
    FAISS_SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    faiss_symbol.save_local(FAISS_SYMBOL_DIR)

    # docs
    write_pickle(module_records, DOCS_MODULE_PATH)
    write_pickle(symbol_records, DOCS_SYMBOL_PATH)

    return RagIndex(
        module_records=module_records,
        symbol_records=symbol_records,
        bm25_module=bm25_module,
        bm25_symbol=bm25_symbol,
        faiss_module=faiss_module,
        faiss_symbol=faiss_symbol,
    )

def load_split_indexes(embeddings_model: OllamaEmbeddings) -> RagIndex | None:
    try:
        with BM25_MODULE_CORPUS_PATH.open("rb") as handle:
            module_corpus = pickle.load(handle)
        with BM25_SYMBOL_CORPUS_PATH.open("rb") as handle:
            symbol_corpus = pickle.load(handle)
        with DOCS_MODULE_PATH.open("rb") as handle:
            module_records = pickle.load(handle)
        with DOCS_SYMBOL_PATH.open("rb") as handle:
            symbol_records = pickle.load(handle)

        if not _is_document_list(module_records):
            return None
        if not _is_document_list(symbol_records):
            return None
        if len(module_corpus) != len(module_records):
            print("module BM25 语料与 docs 条数不一致")
            return None
        if len(symbol_corpus) != len(symbol_records):
            print("symbol BM25 语料与 docs 条数不一致")
            return None

        faiss_module = FAISS.load_local(
            FAISS_MODULE_DIR, embeddings_model, allow_dangerous_deserialization=True
        )
        faiss_symbol = FAISS.load_local(
            FAISS_SYMBOL_DIR, embeddings_model, allow_dangerous_deserialization=True
        )
        return RagIndex(
            module_records=module_records,
            symbol_records=symbol_records,
            bm25_module=BM25Okapi(module_corpus),
            bm25_symbol=BM25Okapi(symbol_corpus),
            faiss_module=faiss_module,
            faiss_symbol=faiss_symbol,
        )
    except Exception:
        traceback.print_exc()
        return None

def _load_rag_index() -> RagIndex | None:
    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        sign = _gen_signature()
        if _judge_can_reload(sign):
            loaded = load_split_indexes(embeddings_model)
            if loaded is not None:
                print("reload split rag index success")
                print(f"  module records: {len(loaded.module_records)}")
                print(f"  symbol records: {len(loaded.symbol_records)}")
            return loaded

        # 初始化流程
        raw_docs = _load_corpus_docs(RAG_CORPUS_PATH)
        if raw_docs is None:
            print(f"无法读取语料: {RAG_CORPUS_PATH}")
            return None
        records = build_records(raw_docs)
        print(f"内部记录数: {len(records)}")
        if not records:
            return None

        t0 = time.perf_counter()
        index = build_split_indexes(records, embeddings_model)
        t1 = time.perf_counter()
        print(f"分类型索引构建完成，耗时 {t1-t0:.2f}s")

        write_json(sign, INDEX_SIGNATURE_PATH)
        return index

    except Exception as exc:
        traceback.print_exc()
        print(f"\n执行失败：{exc}")
        return None

_RAG_INDEX = _load_rag_index()

def _index_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "index not available"}

def _sparse_retrieve(
    query: str,
    bm25: Any,
    records: list[LCDocument],
    k: int,
) -> list[dict]:
    tokens = tokenize_mixed(query)
    scores = bm25.get_scores(tokens)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"doc": records[i], "score": scores[i]}
        for i in top if scores[i] > 0
    ]

def _dense_retrieve(
    query: str,
    faiss_index: Any,
    records: list[LCDocument],
    k: int,
) -> list[dict]:
    docs_with_scores = faiss_index.similarity_search_with_score(query, k=k)
    by_id = {r.metadata["doc_id"]: r for r in records}
    result = []
    for doc, score in docs_with_scores:
        doc_id = doc.metadata.get("doc_id")
        canonical = by_id.get(doc_id, doc)
        result.append({"doc": canonical, "score": score})
    return result

def _doc_key(doc: LCDocument):
    doc_id = doc.metadata.get("doc_id")
    return doc_id if isinstance(doc_id, str) and doc_id else None

def reciprocal_rank_fusion(results: list[list[dict]]) -> list[dict]:
    doc_scores = {}
    for retriever_results in results:
        for rank, retriever_result in enumerate(retriever_results, start=1):
            doc_id = _doc_key(retriever_result["doc"])
            if doc_id is None:
                continue
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": retriever_result["doc"],
                    "rrf_score": 0
                }
            doc_scores[doc_id]["rrf_score"] += 1 / (rank+RRF_K)

    fused = sorted(doc_scores.values(), key=lambda v : v["rrf_score"], reverse=True)
    return fused

def _format_score(score: float) -> float:
    return round(float(score), SCORE_DECIMALS)

def _module_candidate(record: LCDocument, score: float) -> dict[str, Any]:
    return {
        "module_path": record.metadata.get("module_path"),
        "score": _format_score(score),
    }

def _symbol_candidate(record: LCDocument, score: float) -> dict[str, Any]:
    extra = record.metadata.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    return {
        "name": extra.get("name"),
        "file": extra.get("file"),
        "kind": extra.get("symbol_kind"),
        "score": _format_score(score),
    }

def _hybrid_search(query: str, top_k: int, bundle: RagIndex) -> dict[str, Any]:
    module_quota = max(1, top_k // 2)
    symbol_quota = max(1, top_k - module_quota)

    # module 侧
    bm25_mod = _sparse_retrieve(query, bundle.bm25_module, bundle.module_records, PER_SIDE_K)
    faiss_mod = _dense_retrieve(query, bundle.faiss_module, bundle.module_records, PER_SIDE_K)
    rrf_mod = reciprocal_rank_fusion([bm25_mod, faiss_mod])

    # symbol 侧
    bm25_sym = _sparse_retrieve(query, bundle.bm25_symbol, bundle.symbol_records, PER_SIDE_K)
    faiss_sym = _dense_retrieve(query, bundle.faiss_symbol, bundle.symbol_records, PER_SIDE_K)
    rrf_sym = reciprocal_rank_fusion([bm25_sym, faiss_sym])

    # 按配额取候选
    seen_modules: dict[str, dict[str, Any]] = {}
    for item in rrf_mod:
        mp = item["doc"].metadata.get("module_path")
        if not mp:
            continue
        if mp in seen_modules:
            continue
        seen_modules[mp] = _module_candidate(item["doc"], item["rrf_score"])
        if len(seen_modules) >= module_quota:
            break
    modules = list(seen_modules.values())
    symbols = [
        _symbol_candidate(item["doc"], item["rrf_score"])
        for item in rrf_sym[:symbol_quota]
    ]

    return {
        "ok": True,
        "query": query,
        "candidates": {
            "modules": modules,
            "symbols": symbols,
        },
        "retrieval": {
            "bm25_module_hits": len(bm25_mod),
            "faiss_module_hits": len(faiss_mod),
            "bm25_symbol_hits": len(bm25_sym),
            "faiss_symbol_hits": len(faiss_sym),
            "module_candidates": len(modules),
            "symbol_candidates": len(symbols),
        },
    }

@tool(
    description=(
        "Resolve a Chinese or English term to candidate modules and symbols "
        "using split hybrid BM25 + FAISS retrieval. "
        "Module and symbol indexes are searched separately, then merged by quota. "
        "Returns ranked candidates only, not an answer. "
        "Does not fuzzy-match or prefix-match."
    ),
    parameter_descriptions={
        "query": "Natural-language or identifier query, Chinese and English mixed is fine.",
        "top_k": (
            "Total candidate slots. Module quota is max(1, top_k // 2); "
            "symbol quota is the remainder. Default 10."
        ),
    },
)
def resolve_term(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """分类型混合检索术语，按配额返回模块和符号候选。"""
    if _RAG_INDEX is None:
        return _index_unavailable()
    return _hybrid_search(query, top_k, _RAG_INDEX)


def register_rag_tools(registry: ToolRegistry) -> None:
    """注册 RAG 术语解析工具。"""
    registry.register_many(
        resolve_term,
    )
