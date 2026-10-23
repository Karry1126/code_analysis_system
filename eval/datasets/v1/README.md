# 评测数据集 v1

本目录存放结构化的评测用例，供 `eval/harness` 加载。

## 版本与来源

- 版本：v1
- 来源：`doc/五轮测试集.md` 的 T01–T11，共 11 题，四类场景。
- 场景分布：
  - consult（咨询）：T01、T02、T03、T07、T11
  - retrieval（检索）：T04、T10
  - bug（Bug 分析）：T05、T06
  - modify（改代码/新建文件）：T08、T09

## 文件

- `train.jsonl`：回归集（当前 11 题），用于迭代回归与基线。
- `holdout.jsonl`：保留集，当前为空，后续扩充，不参与调参。

## Schema

每行一个 JSON 对象，字段见 `eval/harness/schema.py`：

- `id` / `scenario` / `question` / `tags` / `allow_modify` / `n_runs`
- `expected`：标准答案要点（按场景取用不同字段）
- `grading`：可选，缺省时用 `dataset.py::DEFAULT_GRADING` 按场景给默认分层判分

## 分层抽样规则（后续扩充）

1. 四类场景均需覆盖，每场景 ≥5 条。
2. train / holdout 不重叠（id 唯一、题目不重复）。
3. holdout 仅用于最终验证与 A/B 对比，不用于规则/rubric 迭代调参。
4. 多次运行取均值/方差，`n_runs` 默认 3（modify 场景强制 1，因有文件副作用）。

## 待 review 清单（重要）

以下 `expected` 字段当前为占位或未精确补全，需人工核对代码/索引后修正：

- `T04.gold_list`：rank 组件的真实使用方模块全集（当前仅占位 `plt/rank_event_center`）。
- `T06.root_cause_file`：最强领主跨场积分的真实根因文件（当前占位 `plt/strongest_lord_event_center/src/event_process.cpp`）。
- `T10.gold_list`：iap_proxy 的 conf 真实配置项集合（当前仅占位 `iap_proxy`）。
- 各 consult/bug 题的 `must_contain_symbols`：除已确认的 `CHurtleAcrossMatchPool`、`CheckArrivalMailAndSend`、`ClearArrivalShip` 外，其余符号建议逐题核对补齐。
- 已核对无误：`T08.target_file` = `plt/mine_area_user_event_center/src/rank_node.h`（经 symbol_index.json 确认存在）。
