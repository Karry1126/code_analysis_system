## 工程概述

本工程是一个面向企业 C++/Python 活动服仓库的代码分析助手。目标场景是：研发、策划或运维在处理业务问题时，需要以代码为依据做决策，助手通过离线索引 + Agent 工具链，快速定位相关代码并给出分析，减少人工逐文件翻阅的成本。

## 业务场景/解决问题

- 产品或策划咨询指定模块现有功能设计
- 线下|线上出现bug，需要结合现象对代码进行bug分析
- 研发新需求时，搜索现有代码，了解是否已有生产环境中运行的成熟方案，辅助设计开发方案
- 协助编写代码

## 迭代过程

整体节奏：每一轮基于上一轮暴露的问题做定向优化。六轮下来，从"能跑通"到"能准确回答多模块问题"。

### 第一轮：快速搭建基础工程，验证可行性并确定方向

构建基础 ReAct Agent，提供文本搜索、文本阅读、文本修改等基础工具；构建 10 道测试题，覆盖咨询、检索、Bug 分析、代码修改四类场景。

**结果**：目录级咨询与检索可以处理；指定目录的简单咨询可以处理。稍复杂的问题会失败。

**问题**：

- **工具过于暴力**：`search_text` 全量返回检索内容，prompt 迅速超出上下文上限，任务中断。
- **message 合法性被破坏**：按轮次维护 messages 时，截断会切断 `assistant.tool_calls` 与后续 `tool` 消息的配对，导致 API 拒绝请求。
- **token 不可控**：按轮次维护而不是按 token 数维护，无法预估下一轮是否会超限。



### 第二轮：修复 message 维护，给工具返回加长度保护

将 message 维护从"按轮次"改成"按 token 数"，截断时保留 `tool_calls` 与 `tool` 消息的配对；所有检索与阅读工具加入返回长度保护，检索结果做截断处理。

**结果**：工程合法性得到保证，10 道题均不再中断。但**检索到的信息不可控**——工具可能只返回一部分就被截断，LLM 据此做出错误判断。正确率低，幻觉率高。

**问题**：

- **工具语义与问题语义未对齐**：问题需要基于全量数据作答，但工具返回的是截断数据。截断后 LLM 不知道"数据不完整"。
- **使用者必须熟悉仓库**：否则无法判断 LLM 的结论是否正确。



### 第三轮：引入结构化层（模块与 include 索引）

新增仓库地图 `module_cards.json` 与工具 `list_modules`；新增 `include_index.json` 与工具 `query_include_graph`；修改 system prompt，指引 LLM 按问题类型调度索引工具。

**结果**：涉及多模块的咨询问题，LLM 能正确回答，不再漏模块。

**问题**：

- 第一版 `list_modules` 支持过滤参数，效果不佳。**仓库地图的语义应是"列出模块"，不是"按条件筛选"**。删除过滤参数后修复。



### 第四轮：引入符号索引与引用索引

新增符号索引 `symbol_index.json` 与工具 `find_symbol_definition`、`list_symbols_in_module`；新增引用索引 `reference_index.json` 与工具 `find_symbol_references`。

**结果**：多模块结构分析、Bug 分析、代码修改辅助等问题，LLM 均能正确回答，并能根据问题描述定位到目标代码。

**问题**：

- **用户 query 需与符号名对应**：索引工具才能命中目标。如果用户不熟悉工程，用中文业务词提问，检索会不稳定甚至失败——代码里大多是英文符号。



### 第五轮：引入 RAG，把中文业务词映射到英文符号

用 RAG 把用户问题解析成候选实体列表（modules / symbols）：扫描仓库生成 RAG 语料；FAISS 做稠密检索，BM25 做稀疏检索，RRF 融合两路结果，输出 Top-K 供下游索引工具调用。

**结果**：LLM 能通过用户问题检索出候选实体，但**召回效果不佳，映射不到对应的 module**。

**问题**：

- **语料结构失衡**：13000+ 条语料里 module 只有 300 条，混检时 module 被 symbol 挤出 Top-10。



### 第六轮：module 与 symbol 分索引

建两套 BM25 + 两套 FAISS；检索时分别取 Top-K，再按配额合并。

**结果**：LLM 能通过中文业务用语检索到对应英文模块，进一步读取代码并回答问题。

**当前状态**：六轮迭代后，系统在四类任务上均能给出可追溯的答案。

## 核心设计

四层。索引离线建好，工具只读索引或读工作区文件，Agent 按问题类型选工具。

```
Agent 层
  coding_agent.py 注册工具，CodingAgentRuntime 循环
  system prompt（coding_runtime.py）+ 每轮调度消息
  循环上限 MAX_AGENT_LOOPS = 10
        |
        v
工具层
  术语：resolve_term
  索引：list_modules / query_include_graph
        find_symbol_definition / list_symbols_in_module
        find_symbol_references
  文本：retrieve_code / search_text / search_files_by_name
        read_text_file / list_files
  改文件：replace_text_in_file / write_text_file
        |
        v
索引层
  module cards / include index / symbol index
  reference index / RAG corpus
  data/rag 下按 module、symbol 分开的 BM25 与 FAISS
        |
        v
数据层
  data/*.json 与 data/rag/*
  活动服源码在 COMPANY_CODE_REPO_PATH
```



## 快速开始

1. 准备代码仓库：编辑 .env，把代码路径指向要分析的仓库根目录。
2. 构建索引：按顺序执行以下脚本，前一脚本的输出是后一脚本的输入，不要跳步或调换顺序（`build_symbol_index.py` 依赖本机 `Universal Ctags`）。

```
extract_module_cards.py → build_include_index.py → build_symbol_index.py → build_reference_index.py → build_rag_corpus.py
```

3. 配置 LLM：申请 DeepSeek API Key，注册为环境变量 DEEPSEEK_API_KEY。
4. 准备 embedding 模型：本地安装 Ollama，执行 ollama pull bge-m3。
5. 启动：用 uv 安装依赖并进入虚拟环境，然后运行：

```
python -m code_analysis_system.coding_agent
```



## 工具清单



### 术语解析


| 工具             | 输入          | 输出        | 用途            |
| -------------- | ----------- | --------- | ------------- |
| `resolve_term` | 中文或英文 query | 候选模块 / 符号 | 把业务词解析成英文符号候选 |




### 索引查询


| 工具                       | 输入       | 输出       | 用途              |
| ------------------------ | -------- | -------- | --------------- |
| `list_modules`           | 无        | 全部模块     | 枚举仓库模块          |
| `query_include_graph`    | header 名 | 引用它的模块列表 | 谁 include 了某头文件 |
| `find_symbol_definition` | 符号名      | 定义位置     | 符号定义在哪          |
| `list_symbols_in_module` | 模块路径     | 该模块的符号列表 | 模块里有哪些符号        |
| `find_symbol_references` | 符号名      | 引用位置     | 谁引用了某符号         |




### 文本检索


| 工具                     | 输入          | 输出       | 用途       |
| ---------------------- | ----------- | -------- | -------- |
| `retrieve_code`        | query       | 候选代码位置卡片 | 宽度定位     |
| `search_text`          | query + 子目录 | 命中行      | 子目录内文本匹配 |
| `search_files_by_name` | 文件名片段       | 命中文件     | 按文件名定位   |
| `read_text_file`       | 文件路径 + 行号范围 | 行窗口内容    | 精读代码     |
| `list_files`           | 目录路径        | 一层文件列表   | 枚举目录     |




### 文件修改


| 工具                     | 输入             | 输出   | 用途      |
| ---------------------- | -------------- | ---- | ------- |
| `replace_text_in_file` | 文件 + 旧文本 + 新文本 | 替换结果 | 精确字符串替换 |
| `write_text_file`      | 文件 + 内容        | 写入结果 | 创建或重写文件 |


## 评测结果（自建回归集）

当前建立 11 道端到端回归用例，覆盖咨询、检索、Bug 分析、代码修改四类场景，当前 11/11 通过。初始为 10 道；引入 RAG 后新增 T11（飞船活动），用于验证中文业务词能否经 `resolve_term` 正确映射到英文符号，进而定位代码并分析逻辑——未引入 RAG 前该题无法定位到飞船活动对应模块。该集合用于迭代回归，不是生产环境评测；后续会扩展样本并接入自动化。

| 场景 | 验收标准 | 结果 |
|---|---|---|
| 功能咨询 | 回答基于代码证据，模块/符号不遗漏 | 通过 |
| 代码检索 | 检索结果无漏项，能定位到目标文件/符号 | 通过 |
| Bug 分析 | 能定位到错误代码行并给出依据 | 通过 |
| 代码修改 | 修改符合预期，不破坏上下文 | 通过 |

代表性用例（脱敏）：
| 编号 | 场景 | 输入问题（示例） | 期望结果 | 实际结果 | 证据 |
|---|---|---|---|---|---|
| C01 | 咨询 | 某活动的 结算逻辑是什么？ | 指出模块、关键符号、结算流程 | 通过 | file:line / 工具调用记录 |
| R01 | 检索 | 所有使用排行榜组件的模块有哪些| 完整列出引用模块 | 通过 | 引用索引结果 |
| B01 | Bug 分析 | 某活动排行榜出现了上次活动数据，帮我分析下原因 | 定位到错误代码行 | 通过 | file:line |
| M01 | 代码修改 | 给排行榜新增一个属性ksid_name | 修改符合预期 | 通过 | diff |
| T11 | 咨询 | 飞船活动的回收期能做什么业务 | 经 RAG 映射到飞船活动模块后分析逻辑 | 通过 | resolve_term → 索引工具 |

**基线对比**：第一轮 10 道题存在上下文溢出/任务中断；第二轮不中断但检索截断导致幻觉；第六轮 10/10 通过。跨写法引用查询从漏 20 个模块提升到完整召回 75 个模块。


## 评测 Harness（自动化评测系统）

在上文回归集之上，`eval/` 实现了自动化评测 harness，把「跑题 → 判分 → 出报告 → 环境锁版本 → judge 校准」全流程串起来。核心目标：给原生实现与后续 LangChain/LangGraph 迁移实现提供**同一套题、同一套判分、同一套环境**下的可归因 A/B 对比。

### 目录结构

```
eval/
  config.yaml              # 全局配置：模型/价格/n_runs/judge 阈值/路径
  datasets/v1/             # 结构化用例（train.jsonl + holdout.jsonl）
  harness/                 # 评测核心（框架中立，迁移实现可复用）
    schema.py              # Case / Expected / Grading 数据结构
    dataset.py             # 数据集加载与场景级校验
    runner.py              # 驱动执行 + 多次运行 + 轨迹落盘
    trace.py               # 轨迹 schema（llm_call / tool_call / final_answer）
    grader/
      rules.py             # 确定性规则判分（14 项检查）
      llm_judge.py         # LLM-as-a-judge 语义判分
      calibration.py       # Cohen's κ 校准
      __init__.py          # 三层加权组合器
    metrics.py             # 聚合：均分、P50/P95、token、成本、稳定率
    report.py              # 报告生成（report.md + report.json）
    snapshot.py            # 环境快照：锁版本保证可归因
    sandbox.py             # modify 场景的工作区隔离（copytree + 子进程）
  baselines/               # 基线产物（执行结果，git 忽略）
  artifacts/               # 校准产物（执行结果，git 忽略）
scripts/
  run_eval.py              # 主入口：加载→快照→执行→判分→聚合→报告→A/B
  run_calibration.py       # judge 校准：generate(抽样) / evaluate(算 κ)
```

### 分层判分

每道题按三层加权（权重和 = 1，按场景给默认值）：

| 层 | 判什么 | 判分方式 |
| --- | --- | --- |
| `task` | 答案质量（是否正确、有据、无幻觉） | `llm_judge`（主观题）或 `rules`（retrieval 的 recall/precision） |
| `evidence` | 证据充分性（模块/符号覆盖、无幻觉、根因定位、改动范围） | `rules` |
| `trace` | 工具选择质量（是否用索引、是否读目标文件、是否稳定） | `rules` |

- 主观层用与被测模型不同的 judge（`deepseek-v4-pro`）打分，避免自评乐观偏差。
- 客观层用确定性规则，可复现、无随机性；`--no-judge` 时 task 层跳过、仅剩两层重新归一化（口径偏松，仅用于调试）。

### 环境快照与可归因

`snapshot.json` 锁定评测环境，保证「同一套题、同一套工具」下对比才可归因。校验范围：被测仓库 git commit、评测集哈希、索引文件（`data/`）、system prompt、工具 schema、harness 判分代码。`--snapshot` 传已有快照则先校验，不一致即终止。

### judge 校准（Cohen's κ）

LLM judge 上线前用 Cohen's κ 与人工标注对齐：抽样作答 → 人工标「对 / 部分对 / 错」→ judge 独立打分 → 算 κ（扣除随机一致成分）。`κ ≥ 0.6` 视为 rubric 可用，否则回改 rubric 重校。流程为两段式，中间必须有人工标注。

### 使用方式

```bash
# 首次：创建快照并跑全量，产出基线
python scripts/run_eval.py --run-id baseline_native_v0 --create-snapshot

# 后续：校验快照一致性 + 对比已有基线（迁移到 LangChain 后）
python scripts/run_eval.py --run-id langchain_v0 --runtime-impl langchain \
    --snapshot eval/baselines/baseline_native_v0/snapshot.json \
    --compare eval/baselines/baseline_native_v0/report.json

# 调试：只跑前 3 题、跳过 judge
python scripts/run_eval.py --limit 3 --no-judge

# judge 校准（两段式，中间人工填 human 字段）
python scripts/run_calibration.py generate \
    --traces-dir eval/baselines/baseline_native_v0/traces \
    --out eval/artifacts/calibration_labels.jsonl
python scripts/run_calibration.py evaluate \
    --labels eval/artifacts/calibration_labels.jsonl \
    --out eval/artifacts/calibration_report.json
```

> 当前基线（`baseline_native_v0`）为自动化跑分产物：总体均分约 0.746、稳定率 0.897、估算成本约 2.84 元；`report.md` 中的 P50/P95 为端到端与单次 LLM 调用的百分位延迟。该分数由 harness 三层判分自动产出，与上文「11/11 通过」的人工验收口径不同（后者是规则通过性，前者是加权质量分）。

## 优化方向

1. 评测体系扩充：当前 11 题样本有限，按分层抽样扩到每场景 ≥5 条、补全 holdout，并接入 CI 自动化回归
2. 提升RAG质量：引入重排序，模型候选：BAAI/bge-reranker-v2-m3，增加query与doc的相关性
3. 证据呈现优化：让工具的返回自带代码片段，让LLM的答案自带file:line引用
4. 会话记忆：如果有长对话的需求，引入message的会话记忆
5. 索引热更机制：自动化热更索引，如监控文件改动或定时执行

## 已知限制
- 回归集当前为 10 道自建用例，样本量有限，不代表生产准确率。
- 主要面向 C++/Python 活动服仓库，复杂跨文件重构和大规模仓库性能仍需验证。
- 当前为离线索引 + CLI/Agent 循环，尚未接入权限、多租户、长期记忆和生产监控。
- 索引为离线构建，尚未实现热更新。

