# Project Note

## 1. 如何复刻当前开发环境，并把 Web Agent 跑起来

### 1.1 从头开始的推荐操作流程

以下步骤默认在 Windows + PowerShell + Conda 下执行。

1. 安装 Miniconda 或 Anaconda。
2. 拉取仓库。
3. 进入仓库根目录 `game_assistant`。
4. 新建 Python 3.11.15 环境。
5. 激活环境。
6. 安装依赖。
7. 复制 `.env.example` 为本地 `.env`。
8. 填入 API key。
9. 从仓库根目录启动 Streamlit。

推荐命令如下：

```powershell
git clone <your-repo-url>
cd game_assistant

conda create -n game_assistant_py311 python=3.11.15 -y
conda activate game_assistant_py311

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

如果你不想用命名环境，也可以在仓库目录里创建路径环境：

```powershell
conda create -p .conda python=3.11.15 -y
conda activate .\.conda
```

### 1.2 `.env` 应该怎么设置

项目当前支持三种文本大模型 provider：

- OpenAI
- Groq
- Anthropic

最简单、当前最推荐的文本配置是 OpenAI：

```dotenv
# LLM provider selection
LLM_PROVIDER=openai
LLM_MODEL=

# OpenAI API
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# Anthropic Claude API
ANTHROPIC_API_KEY=

# Groq API
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
```

说明：

- `LLM_PROVIDER=openai` 表示主文本链路走 OpenAI。
- `OPENAI_MODEL=gpt-4o-mini` 是当前更稳妥的默认选择；也可以改成其他你账号可用的模型。
- `LLM_MODEL` 可以留空，因为代码会优先读取 provider 对应的模型变量，例如 `OPENAI_MODEL`。

### 1.3 如何启动 Web Agent

必须从仓库根目录 `game_assistant` 启动，不要从上一级目录启动。

```powershell
streamlit run app/streamlit_app.py
```

如果你没有激活 Conda 环境，也可以这样启动：

```powershell
conda run -n game_assistant_py311 streamlit run app/streamlit_app.py
```

### 1.4 如何验证环境是否正常

先跑测试：

```powershell
pytest -q
```

如果测试通过，再启动网页。

### 1.5 运行时常见问题

#### 问题 1：从错误目录启动，Streamlit 直接失败

- 现象：`streamlit run app/streamlit_app.py` 报路径或导入异常。
- 原因：启动目录不是 `game_assistant` 根目录。
- 解决：先 `cd game_assistant` 再启动。

#### 问题 2：OpenAI 超时或返回参数错误

- 当前代码已经兼容 OpenAI，并对 GPT-5 类模型做了额外参数适配。
- 为了稳定性，仍然推荐优先使用 `gpt-4o-mini`。

#### 问题 3：截图解析不可用

- 原因通常是没有配置 Anthropic key，或者当前 provider 不支持视觉链路。
- 解决：补 `ANTHROPIC_API_KEY`。

#### 问题 4：Windows + OneDrive 路径下 Chroma 不稳定

- 当前代码已经在 Windows 下对 Chroma 持久化目录做了本地迁移保护。
- 所以看到它把索引迁到 `AppData\Local` 一类目录是正常现象，不是报错。

## 2. 关于修改记录

我们已经将所有修改记录的纲要记录在 `modification_log.md` 文件中。后续如果希望知道修改了什么，可以直接让 AI 读取那个 md 文件。

## 3. 关于英文测试集与当前能力评测

我们为了检测 Web Agent，设置了 `eval/english_manual_test_set.md` 文件，里面包含了 8 个英文问题。

我们还实际跑了一遍问题集合，答案输出和人工判读结果记录在 `eval/english_manual_test_results.md` 文件里。

如果想快速了解当前 agent 已经具备哪些能力、哪些题已经通过、哪些题还存在缺陷，可以直接读取这两个文件：

- `eval/english_manual_test_set.md`
- `eval/english_manual_test_results.md`

## 4. 目前项目已经实现的功能

当前项目已经实现的核心能力包括：

1. 文本问答 Web Agent
   - 用户可以在 Streamlit 页面输入问题，系统会返回结构化答案。

2. 多工作流路由
   - 当前支持 `boss_strategy`、`fact_lookup`、`navigation`、`decision_making` 四类主要工作流。

3. 基于本地知识库的检索增强
   - 项目会从本地 wiki、社区数据与索引中检索证据，而不是只依赖模型裸生成。

4. 来源展示与可解释性面板
   - 页面右侧会展示 citations、consensus、trace、执行摘要、停止原因与置信度。

5. 中英文支持
   - 英文问题的答案、trace、右侧 debug 面板文案目前都已经做了专门处理，尽量避免中英文混杂。

6. OpenAI / Groq / Anthropic 文本接入
   - 当前文本链路已经支持三种 provider。

7. 失败降级与兜底
   - 主生成失败时，系统会尝试 fallback；如果还不行，会退回 extractive fallback，而不是完全崩掉。

8. 英文手工评测集
   - 已经建立英文问题测试集，并形成了实际运行结果文件，方便后续继续验收。

## 5. 目前项目在 agentic 方面的设计

### 5.1 设计目标

这个项目的 agentic 设计不是“完全开放式自治 agent”，而是一个 bounded multi-agent system。

这意味着：

- 它有 agentic 行为。
- 但它不是无限制地自己规划任意动作。
- 所有行为都被工作流、工具集和共享状态约束住。

这是为了在课程项目里兼顾：

- 可控性
- 可解释性
- 可测试性
- 不太容易跑飞

### 5.2 目前已经实现的 agentic 机制

1. Router 先做工作流分流
   - 系统不会把所有问题都塞进一个统一 prompt，而是先决定当前问题属于哪类任务。

2. Planner 会生成有边界的执行计划
   - 不是所有 agent 每次都必跑。
   - 系统会根据 goals、evidence gaps 和前置条件决定哪些 agent 执行、哪些跳过。

3. BaseAgent 使用统一 ReAct 循环
   - WikiAgent、CommunityAgent、AnalysisAgent 都通过共享的 ReAct 框架进行下一步动作决策、工具调用和状态写回。

4. AgentState 作为共享白板
   - 每个 agent 不是彼此孤立，而是通过 `AgentState` 共享检索文档、识别实体、共识分析、执行步骤等信息。

5. CommunityAgent 不是一次性检索
   - 它已经具备 goal-driven retrieval 的特征，会根据当前搜索目标构造查询并尝试不同社区来源。

6. AnalysisAgent 会做跨来源共识/冲突分析
   - 当证据足够时，它不是只把检索结果原样丢给生成模型，而是先做结构化分析。

7. SynthesisAgent 是 grounded synthesis
   - 最终生成不是自由发挥，而是基于已有检索结果、citation、consensus 和不确定性约束进行整合。

### 5.3 这种设计的优点

1. 比单一 RAG 更可解释
   - 审稿人或读者可以看到为什么系统这样回答。

2. 比完全开放 agent 更可控
   - 工具边界清楚，工作流可测，不容易出现不可预期的无限行为。

3. 更容易做局部修复
   - 比如修路由、修 CommunityAgent、修 SynthesisAgent，不需要把整个系统推翻重来。

## 6. 当前仍然存在的缺陷

目前主要缺陷包括：

1. 英文文本 profile signal 仍然不够稳定
   - 在英文测试 E6 中，文本里的章节和 build 信息没有成功触发 ProfileAgent，所以个性化还没有完全打通。

2. 冲突分析依赖证据量
   - 如果社区证据不够，AnalysisAgent 可能被跳过，系统就无法稳定展示强冲突处理能力。

3. 社区语料覆盖仍有限
   - 某些问题只能拿到单源或弱证据，答案虽然能保持诚实，但上限会受检索覆盖限制。

4. 截图解析依赖视觉 provider
   - 当前截图链路对 Anthropic 依赖较强，视觉能力还不是 provider-agnostic。

5. 当前 reranker 仍有临时降载设置
   - `retrieval.reranker_mode=lexical` 仍然是激活状态，这是为了降低查询时 LLM rerank 压力，但会牺牲一部分歧义问题上的排序质量。

6. 环境锁定还不够严格
   - 目前能做到功能复刻，但还没有提交 lockfile 来实现完全一致的依赖重建。

## 7. 给后续协作者的建议

如果后续还有人继续维护这个仓库，建议优先关注：

1. 把英文文本 profile signal 做稳。
2. 提升社区数据覆盖和索引质量。
3. 视情况补充严格的环境锁定文件。
4. 继续用 `modification_log.md` 记录每次重要改动。
