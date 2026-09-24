"""极简 BM25，打精确符号和路径词。"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text or "")]


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {
            token: math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in df.items()
        }

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        doc = self.corpus[doc_index]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = self.doc_len[doc_index]
        total = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = self.idf.get(token, 0.0)
            freq = tf[token]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += idf * (freq * (self.k1 + 1)) / denom
        return total

    def rank(self, query_tokens: list[str], limit: int) -> list[tuple[int, float]]:
        scored = [(index, self.score(query_tokens, index)) for index in range(self.doc_count)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in scored[:limit] if score > 0]
