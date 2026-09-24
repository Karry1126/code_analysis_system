"""字符 n-gram 哈希向量。不依赖外部 embedding 模型，便于 Python 3.14 先跑通。"""

from __future__ import annotations

import math

VECTOR_DIM = 256


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vec = [0.0] * dim
    lowered = text.lower()
    if not lowered:
        return vec
    for n_gram in (2, 3):
        if len(lowered) < n_gram:
            continue
        for index in range(len(lowered) - n_gram + 1):
            gram = lowered[index : index + n_gram]
            bucket = hash(gram) % dim
            vec[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return vec
    return [value / norm for value in vec]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
