# Sunus

> 本地记忆增强主动陪伴型 Agent。她的边界，由你创造。

Sunus 是运行在本机的中文智能体。它把模型判断、受控工具执行、证据化记忆、可审计事件和主动陪伴放在同一条运行时链路上：既能完成代码与文档任务，也能记住长期偏好并在适当时间主动发起交流。

Sunus 的定位是可信本机、可信工作区中的本地 Agent。它使用规则、路径守卫、任务授权和进程管理降低误操作风险，但不提供 OS 或容器级安全隔离。

## 核心能力

- 多轮对话与任务分解
- 代码、配置和文档检索
- 文件创建、修改与结果对账
- 本地命令、脚本和测试执行
- 实时可见的思考与执行时间线
- 可复用的本地技能库
- DeepSeek、OpenAI API、MiniMax 与其他兼容 Provider
- SQLite + FTS5 证据化长期记忆
- 会话 JSONL 事件与完成回执
- 5 分钟起步的指数退避主动陪伴

## 快速开始

```bash
cd ~/Documents/Sunus
./bin/sunus doctor --network
./bin/sunus
```

安装全局入口后，可以在任意目录启动：

```bash
sunus
```

启动目录就是当前工作区，包括用户主目录。大范围检索由工具自身的超时、文件数量和敏感路径规则约束。

单次任务：

```bash
sunus exec "检查项目，修复测试并运行验证"
sunus exec --json "总结当前目录结构"
```

## 架构总览

```text
CLI / REPL / exec
    |
    +-- RuntimeEventRenderer ----------> 终端实时执行时间线
    |
    +-- AgentLoop
          |
          +-- ContextAssembler
          |     +-- 热记忆与相关记忆
          |     +-- 本地技能目录
          |     +-- 字符数与 Token 估算
          |     +-- Token 感知压缩与工具配对修复
          |     +-- 会话消息与工具结果
          |
          +-- ModelClient -----------------> DeepSeek 兼容接口
          |
          +-- TaskGrant -------------------> Policy / PathGuard / 审批
          |
          +-- ToolRegistry
          |     +-- ToolSpec：风险、只读、并行、超时、审批
          |     +-- 文件 / 搜索 / 命令 / 网络 / 技能工具
          |     +-- MCP stdio tools
          |
          +-- EventBus --------------------> 终端渲染 + JSONL 审计
          |
          +-- SessionLogger ---------------> 会话记录 + Receipt
          |
          +-- MemoryStore -----------------> SQLite / FTS5 / 信用记录

ProactiveScheduler
    |
    +-- ProactiveService ----------------> 最近记忆 + 模型 + 主动消息
```

## 代码结构

```text
Sunus/
├── bin/
│   └── sunus                 # 启动脚本
├── src/sunus/
│   ├── __main__.py           # 模块入口
│   ├── cli.py                # CLI、REPL、事件渲染、命令分发
│   ├── agent.py              # Agent 主循环与系统提示
│   ├── context.py            # Token 感知压缩、摘要与工具消息配对修复
│   ├── control.py            # Task budget、进展检测与熔断
│   ├── result.py             # 结构化终态 ResultEnvelope 与 Markdown 渲染
│   ├── documents.py          # 结构化 Markdown 文档模板
│   ├── model.py              # Provider 工厂、Chat Completions、Responses
│   ├── model_events.py       # 结构化流式事件
│   ├── policy.py             # TaskGrant、风险分级、路径守卫、审批菜单
│   ├── tool_registry.py      # ToolSpec 与统一工具注册表
│   ├── tools.py              # 内置工具与执行器
│   ├── mcp.py                # stdio MCP 客户端与生命周期
│   ├── events.py             # 事件写盘、渲染回调与脱敏
│   ├── memory.py             # 证据化记忆、检索、信用与清理
│   ├── proactive.py          # 主动陪伴调度与消息生成
│   ├── storage.py            # 会话 JSONL、回执和 Artifact
│   ├── skills.py             # 本地技能发现与解析
│   └── config.py             # 配置、凭据解析和脱敏
├── scripts/install_cli.sh
├── tests/
├── README.md
└── pyproject.toml
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `cli.py` | 解析命令；启动 REPL/exec；维护组件生命周期；把事件渲染成思考与执行时间线。 |
| `agent.py` | 管理一轮任务的消息、模型调用、工具循环、回执与记忆写入。 |
| `context.py` | 估算上下文 Token；在软阈值触发压缩；保留早期约束和近期消息；修复 tool call/result 配对。 |
| `control.py` | 管理模型/工具/时间/Token 预算、空转检测、恢复提示与熔断。 |
| `result.py` | 校验最终结果结构，并渲染为简洁 Markdown 回答。 |
| `documents.py` | 为 README、设计、报告、变更和分析文档提供章节模板。 |
| `model.py` | Provider 工厂；统一 Chat Completions / Responses 请求、流式与非流式回退、重试和错误。 |
| `model_events.py` | 定义文本、推理、工具调用、usage 与完成事件。 |
| `policy.py` | 区分 R0-R3 风险；管理工作区边界、命令规则和交互式授权。 |
| `tool_registry.py` | 注册工具 schema、来源、风险、只读与并行能力、超时和审批模式。 |
| `tools.py` | 执行文件、目录、搜索、命令、网络和技能工具；限制输出与搜索范围。 |
| `mcp.py` | 启动 stdio MCP 服务器，完成初始化、工具发现、调用、超时和进程清理。 |
| `events.py` | 同步写入 JSONL；将已脱敏事件交给终端渲染器或其他消费者。 |
| `memory.py` | 保存偏好、情景和语义记忆；执行检索、使用记录、信用、蒸馏与显式删除。 |
| `proactive.py` | 持久化主动消息排期；生成基于近期记忆的消息。 |
| `storage.py` | 保存会话记录、工具轨迹、变更文件与 Receipt；大输出转 Artifact。 |
| `skills.py` | 从 `skills/<name>/SKILL.md` 发现可复用流程，只按需加载正文。 |
| `config.py` | 合并环境变量、用户配置和本机已保存凭据；提供输出脱敏。 |

## 一次任务如何运行

1. CLI 创建会话、记忆库、事件总线、工具执行器、调度器与 Agent。
2. 每个任务开始时创建任务级 `TaskGrant`，默认风险上限为 R1；同时设置 Quick/Standard/Deep 执行预算。
3. Agent 读取最近偏好、相关记忆和技能目录，组装系统上下文，并把记忆块字符数、请求字符数和 Token 估算写入 Receipt。
4. 模型返回文本、工具调用或二者并存。
5. 工具调用先做风险分级与路径校验，再决定自动执行或请求确认。
6. 工具结果回填到模型上下文；事件实时显示在终端并写入 JSONL。
7. 每次模型请求前检查上下文；超过软阈值时生成可追溯摘要，并只保留安全的消息尾部。
8. 模型结束、预算耗尽或熔断后，生成包含执行指标的 Receipt。
9. 会话摘要写入记忆；如果存在明确偏好表达，同时提取偏好记忆。
10. 任务 Grant 自动失效，等待下一次用户输入。

## 执行预算与熔断

`TurnController` 同时监控多个维度，避免只限制模型轮次：

| Profile | 模型调用 | 工具调用 | 墙钟时间 | 总 Token | 单次回答 Token | 终态预留 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `quick` | 8 | 24 | 3 分钟 | 50,000 | 1,024 | 1 |
| `standard` | 60 | 180 | 60 分钟 | 2,000,000 | 8,192 | 2 |
| `deep` | 120 | 360 | 120 分钟 | 4,000,000 | 16,384 | 2 |

```bash
sunus --budget quick
sunus --budget deep
```

控制器追踪新证据、文件变化、重复动作和连续失败。连续 2 轮无新证据或同类错误重复 2 次时，会注入结构化恢复提示，要求模型换一个最小验证方案；继续无改善则熔断。

错误指纹包含工具、退出码或异常类型，并对错误正文中的时间戳、PID、临时路径、UUID 和内存地址做归一化。已经重复失败的具体工具调用会被阻止，不能靠再次调用同一参数绕过。

命令执行还会检查被管道掩盖的失败，例如退出码为 0 但输出中出现 Python traceback、命令不存在或 DNS 失败。此时运行时标记语义失败，而不是把整条命令视为成功。

接近模型轮次上限，或累计 Token 达到 `warning_ratio`（默认 80%）时，会提前预留 1-2 次终态调用。终态阶段只向模型暴露 `submit_result`，禁止读取、执行、安装或修改；模型必须基于已有证据提交 `completed`、`partial` 或 `blocked` 结果。即使终态请求使本次总 Token 超过硬上限，合法 `submit_result` 仍会被接收，不能被预算检查提前拦截。

### 上下文压缩

上下文按估算 Token 管理，不再按消息条数直接截断：

| Profile | 软阈值 | 硬目标 | 保留近期消息 | 摘要上限 |
| --- | ---: | ---: | ---: | ---: |
| `quick` | 12,000 | 24,000 | 10 | 8,000 字符 |
| `standard` | 200,000 | 960,000 | 48 | 48,000 字符 |
| `deep` | 400,000 | 960,000 | 64 | 64,000 字符 |

压缩优先从用户消息边界切入；单用户多工具轮次会固定保留原始用户要求，再从安全的 assistant 边界截断，不拆开 tool call 与 tool result。Provider 上一轮返回的 `prompt_tokens` 也参与阈值判断，避免字符估算偏低而错过压缩。旧消息按用户要求、工具错误、文件变更和 Agent 结论生成可追溯摘要；完整原始记录仍保留在 Session JSONL。恢复会话后会使用最后一次 `context_checkpoint`，避免重复加载已摘要的历史。

发生熔断时：

- 不再执行后续工具
- 已完成文件修改保留
- 生成 `circuit_open` Receipt
- 写入模型轮次、工具次数、耗时、Token、空转轮次和恢复次数
- 明确报告阻塞原因和续跑所需信息

### 回答输出规范

Sunus 的最终回答默认遵循：

- 结论先行，只保留关键依据、验证结果和剩余风险。
- 简单回答 1-3 句；复杂任务最多 3 个主要部分。
- 不复述完整工具过程、文件内容、diff 或测试日志。
- 修改任务只汇总改动文件、测试状态和未验证部分。
- 普通问答和分析不展示“改动/验证/风险”审计栏，也不重复内部 Token、receipt 或工具统计。
- `changes` 只包含真实文件变更；读取、目录列举和搜索不算改动。
- `verification` 只保留本轮真实执行且与结论相关的验证命令；读取文件不算验证。
- 风险只保留已确认的材料限制，默认最多 1 条。
- 避免无必要的标题、粗体、表格和长列表。
- 诊断、解释和评审类回答不主动展示完整代码、diff、函数实现或大段日志。
- 状态查询只检查用户点名的仓库或目录；同级备份、worktree 和副本不主动展开，
  直接回答状态和关键标识。
- 用户明确要求详细报告时可以扩展，但不重复已知信息。

单次回答的 token 上限只是硬保险，输出风格仍由系统提示约束。

日常问候、情绪交流、关系话题和简短闲聊会自动进入“对话模式”：本轮不挂载
工具，回复目标为 1-2 句、约 80 个汉字，单次输出预算收紧到 512 tokens。
若模型只生成内部推理、正文为空且 `finish_reason=length`，运行时会自动扩大
到配置的单次回答上限并重试一次。

规则无法明确判断的短消息会交给当前模型做一次轻量判定：只输出 `CASUAL` 或
`TASK`。承接上一轮内容的追加要求优先判为 `TASK`，例如“加点卖萌的元素”；
判定失败时默认按任务模式继续，避免把待生成正文截断在 512 tokens。明确的技术
问题、授权动作和社交问候仍走规则快速路径，不额外调用模型。

技术问答仍可按需使用只读工具；当前消息没有授权的写文件、shell 和联网工具
不会直接出现。任务确实需要时，模型可调用 `request_authorization` 发起授权菜单；
批准后立即刷新当前回合的工具列表。

### 结构化终态

模型完成、部分完成或阻塞时调用 `submit_result`，提交经过运行时校验的 `ResultEnvelope`：

```json
{
  "kind": "change_report",
  "status": "completed",
  "summary": "修复完成。",
  "changes": [
    {"path": "src/sunus/model.py", "action": "modified", "purpose": "处理断连"}
  ],
  "verification": [
    {"command": "python3 -m unittest", "status": "passed", "summary": "66/66"}
  ],
  "risks": [],
  "artifacts": [],
  "next_action": ""
}
```

支持的 `kind`：`answer`、`change_report`、`analysis`、`document`、`blocked`。状态必须为 `completed`、`partial` 或 `blocked`；`blocked` 必须给出下一步所需信息。

运行时验证字段和枚举，再交给 Markdown Renderer 生成简洁回答。结构化结果同时写入 Receipt。若模型未调用 `submit_result`，生成兼容回退结果并记录 `result_schema_fallback` 事件。

运行时还会按 `kind` 二次归一化：

- `answer` / `analysis`：最多 3 条关键依据，自动清除 changes、verification 和多余风险。
- `blocked`：只保留阻塞结论、最多 2 条依据、1 条限制和恢复条件。
- `change_report`：只保留真实 changed files 和实际 run_command 验证；没有真实文件变更时自动按普通回答渲染。

交付 Markdown 文件时优先使用 `write_document`：

- `readme`：概述、快速开始、架构、配置、测试、边界
- `design`：目标、架构、数据流、安全边界、验证
- `report`：结论、证据、风险、下一步
- `changelog`：变更、验证、兼容性
- `analysis`：问题、证据、根因、方案、验证

生成文件带 Sunus 元数据和来源列表；缺少必选章节时写入会被拒绝并要求修正。

### 动作边界

Sunus 只执行用户当前消息明确点名的动作：

- 问题、假设、示例、建议和可选优化不会被当作执行指令。
- 普通咨询可读文件、搜索和列目录，但不能自行写文件、跑命令或访问网络。
- 缺少依赖或 tokenizer 时，只读问题直接说明能力边界和估算依据，不擅自安装；只有用户明确要求安装时才进入带授权的安装流程。
- 未被请求的重构、补测试、新增功能和扩展改动会被运行时作用域守卫拒绝。
- 完成结果不会附加未经请求的“下一步”；只有阻塞状态或用户主动要求建议时才保留。
- 主动陪伴模块不受交互任务作用域限制，仍可自由生成自然主动消息。

## 实时执行时间线

交互终端默认显示可观察执行过程：

```text
Sunus
  ◇ 先检查项目说明和目录结构…
  ✓ 完成 · 4 次分析 · 8 个动作 · 读取 4 · 命令 2 · 目录 1 · 技能 1
```

运行中在同一行滚动显示当前思考或动作，结束后只保留一条汇总，不把每个工具结果堆在界面上。不会展示隐藏推理、文件正文、写入正文或凭据。完整事件仍保存在：

```text
~/.sunus/events/YYYY-MM-DD.jsonl
```

整个任务回合保持逐字符输入。输入行固定在活动状态下方，打字立即回显；按 Enter
会把消息加入队列，当前回合结束后按顺序处理。模型等待期间仍可按 `Tab` 或单独
按 `Esc` 打断思考。方向键不会误触发中断。中断只取消当前模型请求，其中尚未
执行的工具调用不会执行；同一轮前面已经完成的文件修改会保留并写入 Receipt，
不会静默回滚。

交互终端启动时会自动启用 UTF-8 行编辑（`IUTF8`），中文、emoji 和组合字符
按完整字符退格，不会只删掉一个字节。输入框按终端宽度折行，不省略开头字符；
状态更新时先清理活动区，再重绘底部输入区。未提交的半截输入会保留到下一轮。
Tab/Esc 仍只用于打断。取消下一轮输入会同时清空暂存。

终端粘贴使用 bracketed-paste 事务读取：`ESC[200~` 与 `ESC[201~` 只作为控制
标记处理，不会进入消息；粘贴块内部的换行不会逐行提前提交，整块内容只写入
一次。粘贴块结束前不会把内部换行当作提交。
重复残影。方向键、Home/End、Delete、Ctrl-A/E/K 可用于编辑。

## 工具系统

| 工具 | 行为 |
| --- | --- |
| `read_file` | 读取工作区内 UTF-8 文本并返回行号。 |
| `list_files` | 按深度和数量限制列出目录；可按 mtime 排序并按最近天数过滤。 |
| `search_text` | 固定字符串搜索；可先按文件聚合命中数，再读取候选文件。 |
| `inspect_runtime_history` | 查询最近会话、最近失败、365 天错误趋势或单回合时间线，只返回有界摘要。 |
| `memory_context_metrics` | 统计当前查询实际注入的偏好、召回记忆字符数和无依赖 Token 估算；只读，并标记 `exact=false`。 |
| `write_file` | 创建或完整覆盖文件。 |
| `write_document` | 按固定模板生成带元数据和必选章节的 Markdown 文档。 |
| `patch_file` | 替换一个文件中的精确文本。 |
| `run_command` | 在工作区运行 shell 命令，返回退出码和输出。 |
| `fetch_url` | 获取 HTTP(S) 文本，属于高风险网络操作。 |
| `list_skills` | 列出本地技能及适用场景。 |
| `read_skill` | 读取某个技能的完整流程。 |
| `submit_result` | 提交最终结构化结果；校验通过后结束回合。 |

通用检索遵循“元数据优先”：先用路径、时间、文件类型或按文件命中数缩小范围，
再读取候选文件的相关行。即使全部历史都有错误，也只向模型返回聚合结果和少量
最新样本；只有用户指定时间/turn_id 时才下钻原始记录。

搜索默认最多 15 秒、20,000 个文件，单条匹配最多返回 500 字符，并跳过
`.sunus`、`.git`、依赖目录和敏感目录。单次工具输出最多注入模型上下文
16,000 字符；更大的结果自动转存 Artifact，并保留路径供后续按需读取。
完整工具输出硬上限仍为 60,000 字符。

工具由 `ToolRegistry` 统一注册。只读且无请求间依赖的工具可以在同一轮模型中
有限并行执行，默认最多 4 个；结果仍按原始 `tool_calls` 顺序回填。写文件、
命令、网络、MCP 工具和 `submit_result` 始终串行，`submit_result` 是终止屏障。

stdio MCP 使用用户级 `~/.sunus/config.json` 配置。MCP 工具默认风险为 R2，
默认要求审批，不继承任务的 R1 自动授权：

```json
{
  "runtime": {
    "streaming": true,
    "max_parallel_tools": 4
  },
  "mcp_servers": {
    "docs": {
      "command": "npx",
      "args": ["-y", "example-mcp-server"],
      "enabled_tools": [],
      "disabled_tools": [],
      "tool_approval_modes": {"search": "ask"},
      "startup_timeout_sec": 10,
      "tool_timeout_sec": 60
    }
  }
}
```

上下文统计直接复用本轮真实检索结果，不会为了计算再执行一次不同的查询。字符数是精确值；`estimated_tokens` 使用内置的 CJK/ASCII 启发式估算，并始终标记 `exact=false`。需要 Provider 精确 Token 时，应使用 API 返回的 usage，而不是把估算值当成账单或上下文窗口结论。

## 模型 Provider

Sunus 通过统一 Provider 接口选择模型服务：

| Provider | 默认接口 | 说明 |
| --- | --- | --- |
| `deepseek` | Chat Completions | 默认 Provider，保持原有 DeepSeek 行为。 |
| `openai` | Chat Completions 或 Responses | 必须显式提供 OpenAI API 模型。 |
| `minimax` | Chat Completions | 默认 `MiniMax-M3.1-Flash`，使用 `max_completion_tokens`。 |
| `openai_compatible` | Chat Completions | 用于其他兼容服务，需要显式 base URL 和模型。 |

Chat Completions 默认使用 SSE 流式事件，CLI 在回合结束后统一渲染最终正文，
避免 Markdown 碎片和工具活动混入答案。工具活动只写入临时状态行，非 TTY
环境默认不输出活动日志；`--verbose` 可查看完整事件。Responses 当前仍使用
结构化响应并转换为同一事件接口。流式内容在进入终端前会经过 DSML 标记延迟
缓冲，协议异常的文本不会直接展示。

选择 Provider：

```bash
MINIMAX_API_KEY="..." sunus --provider minimax
OPENAI_API_KEY="..." sunus --provider openai \
  --api-mode responses --model <模型 ID>
sunus --provider openai_compatible \
  --base-url http://127.0.0.1:11434/v1 \
  --model <本地模型>
```

查看当前 Provider 可访问的模型：

```bash
sunus --provider minimax models
sunus --provider openai --model <模型 ID> models
```

ChatGPT 订阅和 OpenAI API 是不同产品。Sunus 只能使用 OpenAI API Key，不能复用网页版登录态或 Cookie。

MiniMax 使用官方 OpenAI 兼容地址 `https://api.minimax.io/v1`。工具调用属于标准 `tools` 流程；其 `reasoning_content` 会保留在多轮上下文中。详细字段以 [MiniMax 官方 OpenAI SDK 文档](https://platform.minimax.io/docs/api-reference/text-openai-api) 为准。

## 授权与边界

| 风险 | 示例 | 默认行为 |
| --- | --- | --- |
| R0 | 读取、列目录、无副作用查询 | 自动执行 |
| R1 | 工作区写文件、修改代码、测试、脚本、本地开发命令 | 任务内自动执行 |
| R2 | 网络、外部路径、删除类或未知高风险操作 | 显示三选项确认 |
| R3 | 提权、系统关机、磁盘擦除、访问受保护凭据 | 硬拒绝 |

需要确认时提供：

```text
  1. 允许本次
  2. 本会话全部允许
  3. 拒绝
```

支持上下方向键加 Enter、数字键 `1/2/3`、快捷键 `y/x`。系统级高风险规则无法通过菜单绕过。

任务结束后 R1 Grant 自动过期。用户可在会话内按能力覆盖自动判断：

```text
/approve status
/approve on write,command
/approve off network
/approve reset write
/permissions
```

`write`、`command`、`network` 可分别允许、拒绝或恢复 `auto`。显式拒绝优先于
请求意图，且 `request_authorization` 不能绕过；需要恢复时使用 `reset`。
`--yes` 和 `/approve on`（不带能力）仍会放开全部高风险操作，但不关闭硬拒绝规则。

## 路径与执行边界

- 文件工具默认只能访问当前工作区。
- `.git`、`.sunus` 和运行时缓存目录禁止被文件工具修改。
- 相对路径基于启动目录解析。
- 搜索不跟随符号链接，不读取过大的文件。
- 命令通过独立进程组执行，超时后先终止再强制收敛。
- 这是进程和策略级保护，不是恶意代码隔离。不要在不可信仓库中关闭确认或扩大路径权限。

## 记忆系统

记忆数据库位于：

```text
~/.sunus/memory.db
```

主要数据表：

- `memories`：内容、作用域、类型、状态、使用与信用计数。
- `memory_evidence`：记忆来源、Receipt 和证据类型。
- `memory_usage`：哪一轮使用了哪条记忆。
- `memory_credit`：有效、无效和未知反馈。
- `memory_fts`：FTS5 全文检索索引；不可用时回退普通表。
- `persona_traits`：Sunus 可演化性格画像，记录证据数、置信度、版本与状态。
- `proactive_state`：主动陪伴状态。
- `proactive_messages`：已生成主动消息与指纹。

记忆按 `global` 或项目作用域隔离。明确偏好语句形成 `preference`，普通任务形成 `episode`，明确的表达/沟通反馈形成 `persona`，实际发出的主动消息形成 `self_share`。支持检索、信用、蒸馏和显式物理删除。

性格设定分为两层：`SUNUS_GENDER`、名字等属于明确身份设定；`persona_traits`
属于可演化画像。你明确指定的性格或沟通方式会立即成为稳定特质；Sunus 从
实际 `self_share` 中观察到的倾向先进入候选，累计 3 次独立证据后才稳定。
画像保留置信度、证据数、版本和最近证据时间，默认最多 6 个稳定特质，避免
单次表达被固化成性格或画像无限膨胀。

偏好分为两类：

- **短期偏好**：带 `recent` 标签，仅在 `recent_memory_window_hours`（默认 168 小时）内自动注入。“最近喜欢咖啡”“这周想早点睡”属于此类。
- **持久设定**：带 `persistent` 标签，不受 7 天窗口限制，持续参与系统提示和主动陪伴。“收藏钢笔”“喜欢摄影”“以后先给结论”“称呼我阿峰”等属于此类。

用户自定义癖好同时带 `preference,persistent,quirk` 标签。旧数据库中符合持久设定规则的偏好在启动时会自动补上 `persistent` 标签，不会因为升级丢失长期设定。

常用命令：

```bash
sunus memory recent
sunus memory search "主动消息"
sunus memory context-metrics "本地记忆 token 有多少"
sunus memory score
sunus memory distill
sunus memory persona
sunus memory persona --distill
sunus memory add "以后先给结论" --kind preference --global-scope --tags communication,persistent
sunus memory forget <id> --yes
```

`forget` 会同时清理索引、证据、使用记录和信用记录，属于显式物理删除。

## 本地技能

技能是可复用的 Markdown 流程，不修改源码：

```text
<workspace>/skills/<name>/SKILL.md
```

文件可包含 `name`、`description`、`when_to_use` 和 `version` 元数据。Agent 默认只看到名称和简述，需要时再通过 `read_skill` 加载完整正文。技能只提供方法，不改变安全原则与授权模型。

## 主动陪伴

启动后默认 5 分钟发送第一条主动消息。未回复时按以下间隔指数退避：

```text
5 → 10 → 20 → 40 → 80 分钟 ...
```

任何真实用户输入都会复位为 5 分钟。消息生成前会读取最近 7 天内的有效记忆，并额外保留所有带 `persistent` 标签的长期设定；结合最近交流生成自然、简短的消息。

主动消息的五个主题本质上对应 Sunus 的五类记忆。系统先按持久化权重选择记忆类别，
再只从该类别召回证据生成消息；不会因为某类内容更容易生成，就跨类别借用其他记忆：

| 记忆类别 | 权重 | 取用内容 |
| --- | ---: | --- |
| 日常与情绪 | 20% | 定向召回近期生活、状态和情绪记忆后发起交流 |
| 工作与 coding | 20% | 定向召回近期项目、代码、测试和目标记忆后轻量提醒 |
| Sunus 性格与成长 | 20% | 使用逐步积累的 `persona` 互动习惯和性格记忆 |
| Sunus 性别风格融合 | 20% | 使用 Sunus 自身性别设定和用户明确称呼、性别记忆 |
| 用户自定义癖好 | 20% | 使用用户设定；未设定时从明确兴趣和习惯记忆学习 |

调度器使用平滑加权队列交错主题，默认五类等权轮换。`topic_index` 随主动消息持久化，进程重启不会总是从第一个主题开始；`proactive reset` 会同时重置主题序列。

工作与 coding 记忆类别有单独的证据门槛：只有记忆中最近的项目仍标记为
`partial_result`、`user_interrupt`，或正文明确包含“还没完成 / 未解决 /
待修复”等未完成状态时才会进入候选集合。工作类别没有该证据时会从其余可用
记忆类别中按原权重比例重新分配，不会借用项目记忆催进度，也不会把消息错误
归到日常类别。其余类别默认写成 1-2 句、约 30 个汉字的自然生活短语；模型
若返回建议、步骤、总结或待办式内容，会改走短句回退。

日常、工作和自定义癖好主题会使用独立的关键词召回，不只读取最近 8 条对话。Sunus 自身性别未设置时使用中性表达；用户性别只从用户明确设置或明确记忆学习，禁止根据姓名、职业或语气猜测。性别槽位不单独聊身份，而是融合进日常或工作内容。

明确反馈如“先给结论”“简洁一点”“称呼我阿峰”会沉淀成 `kind=persona`、`scope=global` 的互动习惯记忆，供性格与成长主题逐步使用。用户未填写自定义癖好时，20% 槽位仍然保留，由 Sunus 从真实兴趣记忆补充；没有依据时使用低压力开放问题。

可用 `SUNUS_GENDER` 设置 Sunus 自身性别，用 `SUNUS_USER_GENDER` 设置用户性别，用 `SUNUS_PROACTIVE_CUSTOM_QUIRKS="摄影,散步"` 设置自定义癖好；也可以在 `~/.sunus/config.json` 的 `proactive` 块中配置。

主动陪伴状态跨进程持久化；休眠或恢复不会补发消息洪峰。生成失败使用 60 秒短重试，不计入未回复次数。

主动消息本身也是 Sunus 的自我分享记忆。只有实际送达或由用户手动 trigger 后，
才会以 `kind=self_share`、`scope=global` 写入记忆库，并记录消息 ID、原记忆类别、
触发方式和工作区；单纯生成、预览或测试不会写入。后续主动消息会读取最近
`self_share` 维持表达连续性，自我成长类别还会把它与 `persona` 记忆一起使用。

默认比例为五类各 20%；用户明确要求时可以调整，修改会立即应用并持久化：

```bash
sunus proactive ratios
sunus proactive ratios 日常=30 工作=20 成长=20 性别=10 癖好=20
sunus proactive ratios daily_emotion=20 work_coding=20 self_evolution=20 user_gender=20 user_custom=20
```

交互模式：

```text
/proactive ratios
/proactive ratios 日常=30 工作=20 成长=20 性别=10 癖好=20
```

所有比例必须合计 100，并支持中英文主题别名。修改写入 `~/.sunus/config.json`，不会覆盖其他配置。

```bash
sunus proactive status
sunus proactive pause
sunus proactive resume
sunus proactive reset
sunus proactive trigger
```

## 会话与审计

运行数据有两处：

- 工作区 `.sunus/`：当前项目的会话和 Artifact，权限 0700。
- `~/.sunus/`：全局记忆、配置、事件日志和会话回退目录。

每轮结束生成 Receipt，包含：

会话消息和工具配对会以增量记录写入 Session JSONL。`submit_result` 也会写入对应 tool 回执，防止下一轮出现“assistant tool_calls 缺少 tool messages”的 API 400 错误。

- `turn_id` 与 `receipt_id`
- 用户输入和最终回答摘要
- 工具轨迹、变更文件
- 结束原因和创建时间

事件和回执在写入前统一脱敏。Artifact 使用随机文件名并限制权限。

## 配置

常用环境变量：

```bash
export SUNUS_API_KEY="..."
export SUNUS_PROVIDER="deepseek"
export SUNUS_API_MODE="chat_completions"
export SUNUS_MODEL="deepseek-flash"
export SUNUS_BASE_URL="https://api.deepseek.com"
export SUNUS_TOKEN_PARAMETER="max_tokens"
export SUNUS_REASONING_EFFORT=""
export SUNUS_DATA_DIR="$HOME/.sunus"
export SUNUS_PROVIDER_CONFIG="$HOME/.sunus/provider.toml"
export SUNUS_PROVIDER_HOME="$HOME/.sunus/runtime"
export SUNUS_RG_PATH="/path/to/rg"
```

凭据解析顺序：

1. `SUNUS_API_KEY`
2. Provider 专用环境变量：`DEEPSEEK_API_KEY`、`OPENAI_API_KEY` 或 `MINIMAX_API_KEY`
3. `SUNUS_PROVIDER_CONFIG` 指向的本地 TOML
4. 默认 Provider 为本机已有的 DeepSeek 配置

凭据不会复制进项目，也不会写入日志或诊断输出。

用户配置可写在 `~/.sunus/config.json`：

`persona.name` 和 `persona.prompt` 是可修改的人物设定。主对话和主动陪伴都会
读取同一份配置；配置中的人物设定优先于旧的演化性格画像，但不会绕过安全原则、
工具授权和事实准确性要求。

`persona.style` 可设置 `sweetness`、`playfulness`、`warmth`、
`independence`、`initiative`，取值 0-1。`context_intensity` 控制闲聊、
任务和风险提示中的表现强度。当前用户消息中的明确修正优先于配置刻度；历史
身份画像不会覆盖配置。`learn_from_self_share` 默认关闭，主动消息只作为表达
记录，不自动成为性格证据。

`persona.examples` 可提供少量语气样本，模型学习其语感而不照抄。主动消息会
结合时段、空闲时间和可选的情境提示，在好奇、关心、分享、俏皮等表达意图中
轮换；用户刚操作或空闲过久时自动延后。`proactive.min_idle_sec`、
`max_idle_sec` 控制发言窗口，`typing_chars_per_second` 和 `bubble_seconds`
供语音或宿主前端实现逐字呈现。

```json
{
  "provider": "deepseek",
  "api_mode": "chat_completions",
  "model": "deepseek-flash",
  "token_parameter": "max_tokens",
  "reasoning_effort": null,
  "model_options": {},
  "budget": {
    "profile": "standard",
    "no_progress_soft": 2,
    "no_progress_hard": 3,
    "same_error_soft": 2,
    "same_error_hard": 3,
    "consecutive_failure_hard": 4,
    "max_recoveries": 2,
    "finalization_reserve": 2,
    "context_soft_tokens": 48000,
    "context_hard_tokens": 96000,
    "context_keep_recent": 24,
    "context_summary_chars": 16000,
    "max_response_tokens": 8192,
    "max_total_tokens": 2000000
  },
  "max_steps": 60,
  "max_tokens": 16384,
  "request_timeout_sec": 120,
  "runtime": {
    "streaming": true,
    "max_parallel_tools": 4
  },
  "mcp_servers": {},
  "persona": {
    "name": "",
    "prompt": "",
    "examples": [],
    "style": {
      "sweetness": 0.8,
      "playfulness": 0.7,
      "warmth": 0.8,
      "independence": 0.7,
      "initiative": 0.5
    },
    "context_intensity": {
      "chat": 1.0,
      "task": 0.65,
      "risk": 0.3
    },
    "learn_from_self_share": false
  },
  "proactive": {
    "enabled": true,
    "base_interval_sec": 300,
    "backoff_factor": 2,
    "recent_memory_window_hours": 168,
    "recent_memory_limit": 32,
    "preference_items_in_context": 8,
    "topic_weights": {
      "daily_emotion": 40,
      "work_coding": 20,
      "self_evolution": 10,
      "user_gender": 10,
      "user_custom": 20
    },
    "sunus_gender": "",
    "user_gender": "",
    "custom_quirks": [],
    "min_idle_sec": 8,
    "max_idle_sec": 600,
    "typing_chars_per_second": 28,
    "bubble_seconds": 15
  }
}
```

## 常用命令

```bash
sunus
sunus --no-proactive
sunus --verbose
sunus --budget quick
sunus -C /path/to/project
sunus exec "任务"
sunus exec --json "任务"
sunus exec --resume last "继续上一轮"
sunus exec --fork <session_id> "换一种方案"
sunus sessions
sunus resume --last
sunus fork <session_id>
sunus doctor
sunus doctor --network
sunus proactive status|ratios|pause|resume|reset|trigger
sunus memory recent|search|context-metrics|score|distill|add|forget
```

`sunus doctor` 会在启动时实时扫描登录 shell 与进程 `PATH`，
列出实际可执行命令、Python/pytest 候选及对应解释器，并核对权威数据目录、
SQLite 主库、事件和会话文件。它不会依赖代码中写死的工具路径；安装或卸载
命令后可用 `inspect_environment` 的 `refresh=true` 立即重新扫描。

交互命令：

```text
/help
/new
/status
/memory
/tools
/proactive status|ratios|pause|resume|reset|trigger
/permissions
/approve status
/approve on [write,command,network|all]
/approve off [write,command,network|all]
/approve reset [write,command,network|all]
/exit
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

- 模型 HTTP 请求与本地回环
- Agent 工具循环、事件和 Receipt
- 路径守卫、风险分级、授权菜单
- 文件、搜索、命令和技能工具
- ToolRegistry 能力元数据、只读并行与副作用串行屏障
- SSE 文本增量、usage、分片 tool_call 与 DSML 延迟显示
- stdio MCP 握手、工具过滤、审批、调用和进程清理
- 记忆检索、上下文度量、去重、信用与删除
- Token 感知压缩、工具配对修复、会话恢复与分叉
- 语义错误识别、恢复提示和终态预算预留
- 主动陪伴首触、指数退避和持久化
- CLI 启动页与实时执行渲染
- PATH 动态能力发现、pytest 解释器关联和诊断证据门控

## 已知边界

- Chat Completions 支持 SSE 流式；Responses 仍以非流式响应转换为统一事件回退。
- 当前支持 Chat Completions 与 Responses 两类协议；不直接支持 Anthropic、Gemini 的原生消息协议。
- OpenAI 必须使用 API Key；ChatGPT 网页订阅、Cookie 和网页会话不能作为 Sunus 的 API 凭据。
- 不提供操作系统级沙箱，不能安全执行来源不明的恶意代码。
- 命令的静态风险分级不能替代系统隔离、容器或虚拟机。
- 网络能力需要显式授权；主动消息不会自行访问第三方系统。
- 内置上下文 Token 是依赖无关的估算值；Provider 的精确 usage 只能在模型响应返回后获得。
- 当前语音能力由外部桥负责，Sunus 内核保留文本、事件和中断边界。
