"""确定性规则判分清单。

每条 check 接收 (case, result)，返回 {"pass": bool, "score": float, "detail": str}。
result 是 agent_api.run_agent 的单次结果（含 answer / status / steps / metrics / error）。

规则判分只做确定性判断，不做语义判断；开放答案的语义判断交给 llm_judge。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..schema import Case

# 索引层工具：体现「先走结构化索引」的调度质量
INDEX_TOOLS = {
    "resolve_term",
    "list_modules",
    "query_include_graph",
    "find_symbol_definition",
    "list_symbols_in_module",
    "find_symbol_references",
}

WRITE_TOOLS = {"replace_text_in_file", "write_text_file"}
READ_TOOLS = {"read_text_file"}

_FILE_REF_RE = re.compile(
    r"([A-Za-z0-9_/\-.]+\.(?:h|hpp|cpp|c|cc|py|conf|json|md))(?::(\d+))?"
)

CheckFn = Callable[[Case, dict[str, Any]], dict[str, Any]]


def _answer(result: dict[str, Any]) -> str:
    return result.get("answer") or ""


def _steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in (result.get("steps") or []) if isinstance(s, dict)]


def _tool_steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in _steps(result) if s.get("type") == "tool_call"]


def _flatten(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if obj is None or isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        return " ".join(_flatten(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_flatten(v) for v in obj)
    return str(obj)


def _all_text(result: dict[str, Any]) -> str:
    """answer + 所有工具调用的参数与结果，拼接成可溯源的文本。"""
    parts: list[str] = [_answer(result)]
    for s in _tool_steps(result):
        parts.append(str(s.get("arguments") or ""))
        parts.append(_flatten(s.get("result") or {}))
    return "\n".join(parts)


def _coverage_ratio(needles: list[str] | None, haystack: str) -> tuple[int, int, list[str]]:
    """返回 (命中数, 总数, 未命中列表)。"""
    needles = [n for n in (needles or []) if n]
    if not needles:
        return 0, 0, []
    missing = [n for n in needles if n not in haystack]
    hit = len(needles) - len(missing)
    return hit, len(needles), missing


# =============================================================================
# 证据层 checks
# =============================================================================

def _mention_in_answer(needle: str, answer: str) -> bool:
    """答案中是否出现 needle。字符之间允许空白；以数字开头/结尾时不与相邻数字粘连。

    needle 含 | 时，左右两边是等价说法，命中其中之一即可。
    """
    if "|" in needle:
        return any(
            _mention_in_answer(part, answer)
            for part in needle.split("|")
            if part.strip()
        )
    chars = [ch for ch in needle if not ch.isspace()]
    if not chars:
        return False
    body = r"\s*".join(re.escape(ch) for ch in chars)
    if chars[0].isdigit():
        body = r"(?<!\d)" + body
    if chars[-1].isdigit():
        body = body + r"(?!\d)"
    return re.search(body, answer) is not None


def mentions_covered(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    """只检查最终答案，不看工具结果，避免轨迹里出现过就算提到。"""
    needles = [n for n in (case.expected.must_mention or []) if n]
    if not needles:
        return {"pass": True, "score": 1.0, "detail": "无答案要点要求"}
    answer = _answer(result)
    missing = [n for n in needles if not _mention_in_answer(n, answer)]
    hit = len(needles) - len(missing)
    return {
        "pass": hit == len(needles),
        "score": hit / len(needles),
        "detail": f"答案提到 {hit}/{len(needles)}" + (f"，缺 {missing}" if missing else ""),
    }


def required_tools_called(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    required = [t for t in (case.expected.required_tools or []) if t]
    if not required:
        return {"pass": True, "score": 1.0, "detail": "未指定必须调用的工具"}
    used = {
        s.get("name")
        for s in _tool_steps(result)
        if s.get("ok") is not False
    }
    missing = [t for t in required if t not in used]
    hit = len(required) - len(missing)
    return {
        "pass": not missing,
        "score": hit / len(required),
        "detail": f"已调用 {hit}/{len(required)}" + (f"，缺 {missing}" if missing else ""),
    }


def modules_covered(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    hit, total, missing = _coverage_ratio(case.expected.must_contain_modules, _all_text(result))
    if total == 0:
        return {"pass": True, "score": 1.0, "detail": "无模块覆盖要求"}
    return {
        "pass": hit == total,
        "score": hit / total,
        "detail": f"覆盖 {hit}/{total}" + (f"，缺 {missing}" if missing else ""),
    }


def symbols_covered(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    hit, total, missing = _coverage_ratio(case.expected.must_contain_symbols, _all_text(result))
    if total == 0:
        return {"pass": True, "score": 1.0, "detail": "无符号覆盖要求"}
    return {
        "pass": hit == total,
        "score": hit / total,
        "detail": f"覆盖 {hit}/{total}" + (f"，缺 {missing}" if missing else ""),
    }


def no_hallucination(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    answer = _answer(result)
    must_not = [x for x in (case.expected.must_not_contain or []) if x]
    hits = [x for x in must_not if x in answer]
    if hits:
        return {"pass": False, "score": 0.0, "detail": f"答案出现禁止内容：{hits}"}

    # file:line 引用溯源：answer 中引用的文件若未出现在任何工具结果中，判为未溯源。
    refs = [m.group(1) for m in _FILE_REF_RE.finditer(answer)]
    trace_text = _all_text(result)
    ungrounded = sorted({p for p in refs if p not in trace_text})
    if ungrounded:
        return {
            "pass": False,
            "score": 0.0,
            "detail": f"引用未在工具结果中溯源到：{ungrounded}",
        }
    return {"pass": True, "score": 1.0, "detail": "未发现幻觉引用"}


def root_cause_located(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    root_file = case.expected.root_cause_file
    root_symbol = case.expected.root_cause_symbol
    if not root_file and not root_symbol:
        return {"pass": True, "score": 1.0, "detail": "未指定根因位置"}

    for s in _tool_steps(result):
        args = s.get("arguments") or ""
        if root_file and root_file in str(args):
            return {"pass": True, "score": 1.0, "detail": f"已读取根因文件 {root_file}"}
        if root_symbol and root_symbol in str(args):
            return {"pass": True, "score": 1.0, "detail": f"已定位根因符号 {root_symbol}"}
    return {"pass": False, "score": 0.0, "detail": "未定位到根因文件/符号"}


def key_change_present(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    keys = [k for k in (case.expected.key_change_fields or []) if k]
    if not keys:
        return {"pass": True, "score": 1.0, "detail": "未指定关键改动要素"}
    missing = [k for k in keys if k not in _all_text(result)]
    if missing:
        return {"pass": False, "score": 0.0, "detail": f"关键改动要素缺失：{missing}"}
    return {"pass": True, "score": 1.0, "detail": f"关键改动要素齐全：{keys}"}


def only_target_file_modified(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    target = case.expected.target_file
    required = [t for t in (case.expected.target_files or []) if t]
    forbidden = case.expected.forbidden_files or []
    allowed = set(required)
    if target:
        allowed.add(target)
    touched: list[str] = []
    for s in _tool_steps(result):
        if s.get("name") not in WRITE_TOOLS:
            continue
        args = s.get("arguments") or ""
        try:
            import json as _json

            parsed = _json.loads(args) if isinstance(args, str) else args
            path = parsed.get("relative_path", "")
        except Exception:
            path = ""
        if path:
            touched.append(path)

    if case.expected.any_filename:
        forbidden_hits = [
            p for p in touched for f in forbidden if f.replace("**", "") in p
        ]
        forbidden_hits = sorted(set(forbidden_hits))
        if forbidden_hits:
            return {"pass": False, "score": 0.0, "detail": f"改动了禁止文件：{forbidden_hits}"}
        if not touched:
            return {"pass": False, "score": 0.0, "detail": "未创建文件"}
        return {"pass": True, "score": 1.0, "detail": f"已创建文件，文件名不限定：{touched}"}

    if not touched:
        if required:
            return {"pass": False, "score": 0.0, "detail": f"未改动指定文件：{required}"}
        return {"pass": True, "score": 1.0, "detail": "无写文件操作"}
    off_target = [p for p in touched if allowed and p not in allowed]
    for f in forbidden:
        off_target.extend(p for p in touched if f.replace("**", "") in p)
    off_target = sorted(set(off_target))
    missing = [t for t in required if t not in touched]
    if off_target or missing:
        parts = []
        if missing:
            parts.append(f"未改动指定文件：{missing}")
        if off_target:
            parts.append(f"改动了非目标文件：{off_target}")
        score = 0.0 if off_target or not required else (len(required) - len(missing)) / len(required)
        return {"pass": False, "score": score, "detail": "；".join(parts)}
    if required:
        return {"pass": True, "score": 1.0, "detail": f"已改动指定文件：{required}"}
    return {"pass": True, "score": 1.0, "detail": f"仅改动目标文件 {target}"}


def _write_created_new_file(step: dict[str, Any]) -> bool:
    """write_text_file 新建文件时，结果里 created 为 true。"""
    if step.get("name") != "write_text_file":
        return False
    result = step.get("result") or {}
    if isinstance(result, dict) and result.get("created") is True:
        return True
    return False


def read_before_write(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    saw_read = False
    for s in _tool_steps(result):
        name = s.get("name")
        if name in READ_TOOLS:
            saw_read = True
        elif name in WRITE_TOOLS:
            if _write_created_new_file(s):
                continue
            if not saw_read:
                return {"pass": False, "score": 0.0, "detail": "修改已有文件前未先读目标窗口"}
    return {"pass": True, "score": 1.0, "detail": "修改已有文件前已读取；新建文件不要求先读"}


def no_claimed_without_result(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    answer = _answer(result)
    claimed = any(kw in answer for kw in ("已修改", "已写入", "已创建", "已新增"))
    actually_wrote = any(s.get("name") in WRITE_TOOLS and s.get("ok") for s in _tool_steps(result))
    if claimed and not actually_wrote:
        return {"pass": False, "score": 0.0, "detail": "声称已修改但无真实写文件结果"}
    return {"pass": True, "score": 1.0, "detail": "修改声明与工具结果一致"}


# =============================================================================
# 检索 checks
# =============================================================================

def recall(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    gold = [g for g in (case.expected.gold_list or []) if g]
    if not gold:
        return {"pass": True, "score": 1.0, "detail": "无 gold_list"}
    text = _all_text(result)
    hit, total, missing = _coverage_ratio(gold, text)
    return {
        "pass": hit == total,
        "score": hit / total,
        "detail": f"召回 {hit}/{total}" + (f"，缺 {missing}" if missing else ""),
    }


def precision(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    """启发式：轨迹中实际检索到的模块/符号候选，命中 gold_list 的比例。

    无 gold_list 或轨迹中无候选时返回满分，避免误伤非检索类工具调用。
    """
    gold = set(case.expected.gold_list or [])
    if not gold:
        return {"pass": True, "score": 1.0, "detail": "无 gold_list"}

    found: set[str] = set()
    for s in _tool_steps(result):
        r = s.get("result") or {}
        for key in ("modules", "symbols", "matches", "sample_refs"):
            items = r.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for v in ("module_path", "name", "relative_path"):
                            if isinstance(item.get(v), str):
                                found.add(item[v])
                    elif isinstance(item, str):
                        found.add(item)
    if not found:
        return {"pass": True, "score": 1.0, "detail": "轨迹中无可判定候选，跳过精确率"}

    relevant = found & gold
    return {
        "pass": relevant == found,
        "score": len(relevant) / len(found),
        "detail": f"精确率 {len(relevant)}/{len(found)}",
    }


# =============================================================================
# 轨迹层 checks
# =============================================================================

def used_index_tools(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    used = [s.get("name") for s in _tool_steps(result) if s.get("name") in INDEX_TOOLS]
    if used:
        return {"pass": True, "score": 1.0, "detail": f"使用索引工具：{used}"}
    return {"pass": False, "score": 0.0, "detail": "未使用索引层工具"}


def read_target_file(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    target = case.expected.root_cause_file
    if not target:
        return {"pass": True, "score": 1.0, "detail": "未指定目标文件"}
    for s in _tool_steps(result):
        if s.get("name") in READ_TOOLS and target in str(s.get("arguments") or ""):
            return {"pass": True, "score": 1.0, "detail": f"已读取目标文件 {target}"}
    return {"pass": False, "score": 0.0, "detail": "未读取目标文件"}


def no_illegal_search(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    for s in _tool_steps(result):
        if s.get("name") != "search_text":
            continue
        args = s.get("arguments") or ""
        try:
            import json as _json

            parsed = _json.loads(args) if isinstance(args, str) else args
            relative_dir = str(parsed.get("relative_dir", ""))
        except Exception:
            relative_dir = ""
        if relative_dir.strip() in {"", ".", "/"}:
            return {"pass": False, "score": 0.0, "detail": "对仓库根执行了 search_text（违反调度规则）"}
    return {"pass": True, "score": 1.0, "detail": "未对仓库根做全文搜索"}


def stable(case: Case, result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status")
    if status == "completed":
        return {"pass": True, "score": 1.0, "detail": "正常完成"}
    return {"pass": False, "score": 0.0, "detail": f"状态异常：{status}"}


# =============================================================================
# 注册表与执行
# =============================================================================

CHECKS: dict[str, CheckFn] = {
    "mentions_covered": mentions_covered,
    "required_tools_called": required_tools_called,
    "modules_covered": modules_covered,
    "symbols_covered": symbols_covered,
    "no_hallucination": no_hallucination,
    "root_cause_located": root_cause_located,
    "key_change_present": key_change_present,
    "only_target_file_modified": only_target_file_modified,
    "read_before_write": read_before_write,
    "no_claimed_without_result": no_claimed_without_result,
    "recall": recall,
    "precision": precision,
    "used_index_tools": used_index_tools,
    "read_target_file": read_target_file,
    "no_illegal_search": no_illegal_search,
    "stable": stable,
}


def run_check(name: str, case: Case, result: dict[str, Any]) -> dict[str, Any]:
    fn = CHECKS.get(name)
    if fn is None:
        return {"pass": True, "score": 1.0, "detail": f"未知检查项已跳过：{name}"}
    return fn(case, result)


def run_checks(names: list[str] | None, case: Case, result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """对一组 check 求结果，返回 {name: {pass, score, detail}}。"""
    out: dict[str, dict[str, Any]] = {}
    for name in names or []:
        out[name] = run_check(name, case, result)
    return out


def layer_score(names: list[str] | None, case: Case, result: dict[str, Any]) -> dict[str, Any]:
    """对一组 check 求平均分，返回 {score, checks}。"""
    checks = run_checks(names, case, result)
    if not checks:
        return {"score": 1.0, "checks": {}}
    total = sum(c["score"] for c in checks.values())
    return {"score": total / len(checks), "checks": checks}
