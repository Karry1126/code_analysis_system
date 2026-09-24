"""中文业务名与仓库路径/符号之间的别名。"""

from __future__ import annotations

import re

MODULE_ALIASES: dict[str, list[str]] = {
    "生日礼包": ["birthday_gift", "birthday"],
    "生日": ["birthday"],
    "绝地飞驰": ["hurtle_across", "hurtle"],
    "掠夺": ["plunder", "hurtle"],
    "排行": ["rank", "RankNode"],
    "模块": ["event_module_list", "event_center"],
}


def expand_query(query: str) -> dict[str, object]:
    """拆出语义原句、别名扩词和英文标识符。"""
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

    for alias, expansions in MODULE_ALIASES.items():
        if alias in query:
            add(alias)
            for item in expansions:
                add(item)

    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        add(token)

    if not terms:
        for part in query.split():
            add(part)

    return {"query": query, "terms": terms}


def aliases_for_path(relative_path: str) -> list[str]:
    """路径命中英文模块名时，把对应中文别名拼进索引文本。"""
    blob = relative_path.replace("\\", "/").lower()
    names: list[str] = []
    for zh, expansions in MODULE_ALIASES.items():
        if any(item.lower() in blob for item in expansions):
            names.append(zh)
    return names
