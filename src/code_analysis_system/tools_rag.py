"""术语解析工具：resolve_term。

用 BM25 + FAISS 混合检索，RRF 融合后返回模块/符号候选，不调用 LLM。
索引目录在 config.py 的 RAG_INDEX_DIR。
文件名：docs.pkl、bm25_corpus.pkl、faiss.index。
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

RETRIEVER_K = 50
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
    records: list[LCDocument]
    bm25: Any
    faiss_index: Any
    # model: Any

def _gen_signature()-> dict:
    with open(RAG_CORPUS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "embeddings_model": EMBEDDING_MODEL_NAME,
        "english_token_pattern": ENGLISH_TOKEN_PATTERN,
        "chinese_token_pattern": CHINESE_TOKEN_PATTERN,
        "camel_split_pattern": CAMEL_SPLIT_PATTERN,
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
        and os.path.exists(BM25_CORPUS_PATH)
        and os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(FAISS_DOCSTORE_PATH)
        and os.path.exists(DOCS_PATH)
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

def _load_rag_index() -> RagIndex | None:
    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        sign = _gen_signature()
        reload_flag = _judge_can_reload(sign)
        if True == reload_flag: #重启流程   
            with BM25_CORPUS_PATH.open("rb") as handle:
                tokenized_corpus = pickle.load(handle)
            if not isinstance(tokenized_corpus, list):
                return None
            bm25 = BM25Okapi(tokenized_corpus)

            with DOCS_PATH.open("rb") as handle:
                records = pickle.load(handle)
            if not _is_document_list(records):
                return None
            if len(tokenized_corpus) != len(records):
                print("BM25 语料与 docs.pkl 条数不一致，放弃 reload")
                return None

            vectorstore = FAISS.load_local(
                RAG_INDEX_DIR,
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
            print("reload local rag index success")
            return RagIndex(
                records=records,
                bm25=bm25,
                faiss_index=vectorstore
            )
        else:   #初始化流程
            raw_docs = _load_corpus_docs(RAG_CORPUS_PATH)
            if raw_docs is None:
                print(f"无法读取语料，或顶层缺少 docs 列表: {RAG_CORPUS_PATH}")
                return None
            records = build_records(raw_docs)
            print(f"内部记录数: {len(records)}")

            t0 = time.perf_counter()
            tokenized_corpus = build_bm25_corpus(records)
            bm25 = BM25Okapi(tokenized_corpus)
            write_pickle(tokenized_corpus, BM25_CORPUS_PATH)
            t1 = time.perf_counter()
            print(f"已写入 {BM25_CORPUS_PATH} 耗时:{t1-t0}")

            # vectorstore = FAISS.from_documents(records, embeddings_model)
            if not records:
                print("语料为空，跳过 FAISS 构建")
                return None
            vectorstore = build_faiss_index(records, embeddings_model)
            vectorstore.save_local(RAG_INDEX_DIR)
            t2 = time.perf_counter()
            print(f"已写入 {FAISS_INDEX_PATH} 耗时:{t2-t1}")

            write_json(sign, INDEX_SIGNATURE_PATH)
            write_pickle(records, DOCS_PATH)
            print(f"已写入 {INDEX_SIGNATURE_PATH} {DOCS_PATH}")

            return RagIndex(
                records=records,
                bm25=bm25,
                faiss_index=vectorstore
            )

    except Exception as exc:
        traceback.print_exc()
        print(f"\n执行失败：{exc}")
        return None

_RAG_INDEX = _load_rag_index()

def _index_unavailable() -> dict[str, Any]:
    return {"ok": False, "error": "index not available"}

def _sparse_retrieve(query: str, bundle: RagIndex) -> list[dict]:
    tokenized_question = tokenize_mixed(query)
    scores = bundle.bm25.get_scores(tokenized_question)
    top_k = sorted(range(len(scores)), key=lambda i : scores[i], reverse=True)[:RETRIEVER_K]
    for rank, idx in enumerate(top_k, 1):
        rec = bundle.records[idx]
        if rec.metadata.get("kind") == "module":
            print(f"[BM25 rank {rank}] {rec.metadata['doc_id']}")
    return [
        {
            "doc":bundle.records[idx],
            "score":scores[idx]
        }
        for idx in top_k if scores[idx] > 0
    ]

def _dense_retrieve(query: str, bundle: RagIndex) -> list[dict]:
    docs_with_scores = bundle.faiss_index.similarity_search_with_score(query, k=RETRIEVER_K)
    for rank, (doc, _score) in enumerate(docs_with_scores, 1):
        if doc.metadata.get("kind") == "module":
            print(f"[FAISS rank {rank}] {doc.metadata['doc_id']}")
    return [
        {
            "doc": doc,
            "score": score
        }
        for doc, score in docs_with_scores 
    ]

def _doc_key(doc: LCDocument):
    doc_id = doc.metadata.get("doc_id")
    return doc_id if doc_id is not None else doc.page_content

def reciprocal_rank_fusion(results: list[list[dict]]) -> list[dict]:
    doc_scores = {}
    for retriever_results in results:
        for rank, retriever_result in enumerate(retriever_results, start=1):
            doc_id = _doc_key(retriever_result["doc"])
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
    bm25_hits = _sparse_retrieve(query, bundle)
    faiss_hits = _dense_retrieve(query, bundle)
    rrf_resutl = reciprocal_rank_fusion([bm25_hits, faiss_hits])
    selected = rrf_resutl[: max(top_k, 0)]

    seen_modules: dict[str, dict[str, Any]] = {}
    symbols: list[dict[str, Any]] = []
    for item in selected:
        kind = item["doc"].metadata.get("kind")
        if kind == "module":
            mp = item["doc"].metadata.get("module_path")
            if not mp:
                continue
            candidate = _module_candidate(item["doc"], item["rrf_score"])
            prev = seen_modules.get(mp)
            if prev is None or candidate["score"] > prev["score"]:
                seen_modules[mp] = candidate
        elif kind == "symbol":
            symbols.append(_symbol_candidate(item["doc"], item["rrf_score"]))

    modules = sorted(seen_modules.values(), key=lambda x: x["score"], reverse=True)

    return {
        "ok": True,
        "query": query,
        "candidates": {
            "modules": modules,
            "symbols": symbols,
        },
        "retrieval": {
            "bm25_hits": len(bm25_hits),
            "faiss_hits": len(faiss_hits),
            "fused_count": len(rrf_resutl),
            "module_candidates": len(modules),
            "symbol_candidates": len(symbols),
        },
    }

@tool(
    description=(
        "Resolve a Chinese or English term to candidate modules and symbols "
        "using hybrid BM25 + FAISS retrieval. "
        "Returns ranked candidates only, not an answer. "
        "Does not fuzzy-match or prefix-match."
    ),
    parameter_descriptions={
        "query": "Natural-language or identifier query, Chinese and English mixed is fine.",
        "top_k": "Number of fused documents to keep after RRF. Default 10.",
    },
)
def resolve_term(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """混合检索术语，返回模块和符号候选。"""
    if _RAG_INDEX is None:
        return _index_unavailable()
    return _hybrid_search(query, top_k, _RAG_INDEX)


def register_rag_tools(registry: ToolRegistry) -> None:
    """注册 RAG 术语解析工具。"""
    registry.register_many(
        resolve_term,
    )