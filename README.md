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
2. 构建索引：按顺序执行以下脚本，前一脚本的输出是后一脚本的输入，不要跳步或调换顺序。

```
build_module_cards.py → build_include_index.py → build_symbol_index.py → build_rag_corpus.py
```

1. 配置 LLM：申请 DeepSeek API Key，注册为环境变量 DEEPSEEK_API_KEY。
2. 准备 embedding 模型：本地安装 Ollama，执行 ollama pull bge-m3。
3. 启动：用 uv 安装依赖并进入虚拟环境，然后运行：

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




## 优化方向

1. 评测自动化：设计测试集、评测指标和评测系统，优化完检索质量后，能准确、有效评估出优化效果
2. 提升RAG质量：引入重排序，模型候选：BAAI/bge-reranker-v2-m3，增加query与doc的相关性
3. 证据呈现优化：让工具的返回自带代码片段，让LLM的答案自带file:line引用
4. 会话记忆：如果有长对话的需求，引入message的会话记忆
5. 索引热更机制：自动化热更索引，如监控文件改动或定时执行

