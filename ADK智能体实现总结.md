# ADK 智能体实现总结

## 一、核心架构模式

ADK 智能体基于 **Agent + Tool + Prompt** 三要素架构：

```
用户输入 → Agent（携带 Prompt） → 调用 Tools → LLM 推理 → 返回结果
```

### 智能体类型分类

- **单智能体**（Single Agent）：一个 LLM 完成任务
- **多智能体**（Multi Agent）：多个智能体协作
- **顺序智能体**（Sequential Agent）：智能体按顺序执行

---

## 二、Python 实现（最成熟，40+ 示例）

### 1. 单智能体示例（RAG）

**核心代码：**

```python
# rag/agent.py
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval

# 定义工具
ask_vertex_retrieval = VertexAiRagRetrieval(
    name='retrieve_rag_documentation',
    description='从 RAG 语料库检索文档',
    rag_resources=[rag.RagResource(rag_corpus=os.environ.get("RAG_CORPUS"))],
    similarity_top_k=10,
)

# 创建智能体
root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='ask_rag_agent',
    instruction=INSTRUCTION_PROMPT,  # 提示词
    tools=[ask_vertex_retrieval],     # 工具列表
)
```

**实现流程：**

1. 定义 `instruction`（系统提示词）
2. 配置 `tools`（可用工具）
3. 实例化 `Agent` 对象

### 2. 多智能体示例（LLM Auditor）

**核心代码：**

```python
# llm_auditor/agent.py
from google.adk.agents import SequentialAgent
from .sub_agents.critic import critic_agent
from .sub_agents.reviser import reviser_agent

# 顺序执行的多智能体
llm_auditor = SequentialAgent(
    name='llm_auditor',
    description='评估并改进 LLM 生成的答案',
    sub_agents=[critic_agent, reviser_agent],  # 子智能体列表
)
```

**子智能体定义：**

```python
# sub_agents/critic/agent.py
critic_agent = Agent(
    model='gemini-2.5-flash',
    name='critic_agent',
    instruction=CRITIC_PROMPT,
    tools=[google_search],              # 使用 Google 搜索工具
    after_model_callback=_render_reference,  # 回调函数
)
```

### 3. 复杂单智能体（Customer Service）

**核心代码：**

```python
# customer_service/agent.py
root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='customer_service_agent',
    global_instruction=GLOBAL_INSTRUCTION,  # 全局上下文
    instruction=INSTRUCTION,                 # 任务指令
    tools=[
        send_call_companion_link,
        approve_discount,
        access_cart_information,
        modify_cart,
        # ... 12 个自定义工具
    ],
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
)
```

**自定义工具示例：**

```python
# tools/tools.py
def access_cart_information(customer_id: str) -> dict:
    """检索客户购物车内容"""
    # 模拟数据库查询
    return {"items": [...], "total": 150.00}
```

---

## 三、Java 实现

### 核心代码（Software Bug Assistant）

```java
// SoftwareBugAssistant.java
public class SoftwareBugAssistant {
    public static BaseAgent initAgent() {
        // 1. 连接 MCP Toolbox（数据库工具）
        String mcpServerUrl = System.getenv("MCP_TOOLBOX_URL");
        SseServerParameters params = SseServerParameters.builder()
            .url(mcpServerUrl).build();

        McpToolset.McpToolsAndToolsetResult result =
            McpToolset.fromServer(params, new ObjectMapper()).get();
        List<BaseTool> mcpTools = result.getTools();

        // 2. 添加 Google Search 工具
        LlmAgent googleSearchAgent = LlmAgent.builder()
            .model("gemini-2.5-flash")
            .name("google_search_agent")
            .tools(new GoogleSearchTool())
            .build();

        // 3. 创建主智能体
        return LlmAgent.builder()
            .model("gemini-2.5-flash")
            .name("SoftwareBugAssistant")
            .instruction(INSTRUCTION_PROMPT)
            .tools(allTools)  // MCP + Google Search
            .build();
    }
}
```

**关键特性：**

- 使用 **MCP (Model Context Protocol)** 连接外部系统（PostgreSQL）
- 通过 `McpToolset` 动态加载数据库工具
- 支持 RAG 向量搜索（Cloud SQL + text-embeddings-005）

**MCP 工具配置：**

```yaml
# mcp-toolbox/tools.yaml
sources:
  postgresql:
    kind: cloud-sql-postgres
    project: your-project
    instance: software-assistant
    database: tickets-db

tools:
  - name: get-tickets-by-status
    source: postgresql
    query: "SELECT * FROM tickets WHERE status = :status"
```

---

## 四、TypeScript 实现

```typescript
// customer_service/agent.ts
import { LlmAgent } from "@google/adk";

export const rootAgent = new LlmAgent({
  model: "gemini-2.0-flash-001",
  name: "customer_service_agent",
  instruction: COMBINED_INSTRUCTION,
  tools: [
    sendCallCompanionLinkTool,
    approveDiscountTool,
    accessCartInformationTool,
    // ... 其他工具
  ],
  beforeToolCallback: beforeTool,
  afterToolCallback: afterTool,
  beforeModelCallback: rateLimitCallback,
});
```

**工具定义：**

```typescript
// tools/function_tools.ts
export const accessCartInformationTool = FnTool.from({
  name: 'access_cart_information',
  description: '检索客户购物车内容',
  parameters: z.object({
    customer_id: z.string(),
  }),
  func: async ({ customer_id }) => {
    // 实现逻辑
    return { items: [...], total: 150.00 };
  },
});
```

---

## 五、Go 实现

```go
// auditor/auditor.go
func GetLLmAuditorAgent(ctx context.Context) agent.Agent {
    model, _ := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{})

    // 创建子智能体
    criticAgent, _ := critic.New(model)
    reviserAgent, _ := reviser.New(model)

    // 顺序智能体
    rootAgent, _ := sequentialagent.New(sequentialagent.Config{
        AgentConfig: agent.Config{
            Name: "llm_auditor",
            SubAgents: []agent.Agent{criticAgent, reviserAgent},
        },
    })
    return rootAgent
}
```

**子智能体定义：**

```go
// critic/critic.go
func New(model model.LLM) (agent.Agent, error) {
    return llmagent.New(llmagent.Config{
        Model:       model,
        Name:        "critic_agent",
        Instruction: CriticPrompt,
        Tools:       []tool.Tool{geminitool.GoogleSearch{}},
        AfterModelCallbacks: []llmagent.AfterModelCallback{
            renderReference,  // 渲染引用来源
        },
    })
}
```

---

## 六、跨语言实现对比

| 特性         | Python       | Java          | TypeScript    | Go         |
| ------------ | ------------ | ------------- | ------------- | ---------- |
| **语法风格** | 简洁，函数式 | 构建器模式    | 类实例化      | 配置结构体 |
| **工具定义** | 装饰器/函数  | 接口实现      | FnTool.from() | 接口实现   |
| **回调机制** | 函数参数     | Lambda 表达式 | 箭头函数      | 函数类型   |
| **MCP 支持** | ✅           | ✅            | ✅            | ✅         |
| **示例数量** | 40+          | 2             | 1             | 1          |
| **成熟度**   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐      | ⭐⭐⭐        | ⭐⭐⭐     |

---

## 七、核心工作流程

### 1. 单智能体执行流程

```
用户输入
  ↓
Agent 接收消息
  ↓
LLM 分析 + Instruction
  ↓
决定是否调用 Tool
  ↓ (需要工具)
执行 Tool（如数据库查询、搜索）
  ↓
Tool 返回结果
  ↓
LLM 综合结果生成回复
  ↓
返回给用户
```

### 2. 多智能体执行流程（Sequential）

```
用户输入
  ↓
Router Agent（可选）
  ↓
Sub-Agent 1 (Critic)
  ├─ 提取声明
  ├─ 调用 Google Search
  └─ 生成验证报告
  ↓
Sub-Agent 2 (Reviser)
  ├─ 接收验证报告
  ├─ 修正错误信息
  └─ 生成最终答案
  ↓
返回给用户
```

---

## 八、关键技术点

### 1. Prompt Engineering

**Python 示例：**

```python
INSTRUCTION = """
你是专业的客服助手。

核心能力：
1. 个性化客户服务
2. 产品识别与推荐
3. 订单管理

工具使用：
- access_cart_information: 查看购物车
- modify_cart: 修改购物车
- get_product_recommendations: 获取推荐

约束条件：
- 使用 Markdown 格式化表格
- 不要暴露内部实现细节
- 执行操作前需确认
"""
```

### 2. 回调机制

**Python 示例：**

```python
def after_tool_callback(context, tool_result):
    """工具执行后的处理"""
    logger.info(f"Tool {context.tool_name} returned: {tool_result}")
    # 可以修改返回结果
    return tool_result
```

**Go 示例：**

```go
func renderReference(
    ctx agent.CallbackContext,
    llmResponse *model.LLMResponse,
    respErr error,
) (*model.LLMResponse, error) {
    // 处理 grounding metadata
    for _, chunk := range llmResponse.GroundingMetadata.GroundingChunks {
        // 添加引用信息
    }
    return llmResponse, nil
}
```

---

## 九、典型智能体模式

### 1. RAG 模式（检索增强生成）

- **用途**：文档问答、知识库查询
- **核心**：Vertex AI RAG Engine + Vector Search
- **示例**：Documentation Retrieval Agent
- **关键代码**：
  ```python
  VertexAiRagRetrieval(
      rag_resources=[rag.RagResource(rag_corpus=corpus_id)],
      similarity_top_k=10,
      vector_distance_threshold=0.6,
  )
  ```

### 2. Multi-Agent 协作模式

- **用途**：复杂任务分解
- **核心**：Sequential/Parallel Agent
- **示例**：LLM Auditor（Critic + Reviser）
- **工作流**：
  1. Critic Agent：提取声明 → 验证事实
  2. Reviser Agent：修正错误 → 生成最终答案

### 3. Tool-Heavy 模式

- **用途**：业务系统集成
- **核心**：大量自定义工具 + MCP
- **示例**：Customer Service（12+ 工具）
- **特点**：
  - 购物车管理
  - 产品推荐
  - 预约调度
  - CRM 集成

### 4. Database Integration 模式

- **用途**：数据库操作
- **核心**：MCP Toolbox + Cloud SQL
- **示例**：Software Bug Assistant
- **功能**：
  - SQL 查询（get-tickets-by-status）
  - 向量搜索（search-tickets）
  - 数据更新（update-ticket-priority）

---

## 十、项目结构对比

### Python 项目结构

```
agent-name/
├── agent_name/              # 核心代码
│   ├── sub_agents/          # 子智能体
│   │   ├── critic/
│   │   │   ├── agent.py
│   │   │   └── prompt.py
│   │   └── reviser/
│   ├── tools/               # 自定义工具
│   ├── shared_libraries/    # 共享库
│   ├── agent.py             # 主智能体
│   └── prompts.py           # 提示词
├── deployment/              # 部署脚本
├── eval/                    # 评估测试
├── tests/                   # 单元测试
├── pyproject.toml           # 依赖管理
└── README.md
```

### Java 项目结构

```
agent-name/
├── agent-module/
│   ├── src/main/java/
│   │   └── Agent.java       # 主智能体
│   ├── src/main/resources/
│   │   └── prompts/         # 提示词
│   └── pom.xml              # Maven 配置
├── deployment/
│   ├── mcp-toolbox/
│   │   └── tools.yaml       # MCP 工具配置
│   └── Dockerfile
└── README.md
```

### TypeScript 项目结构

```
agent-name/
├── customer_service/
│   ├── entities/            # 数据实体
│   ├── tools/               # 工具定义
│   ├── shared_libraries/    # 共享库
│   ├── agent.ts             # 主智能体
│   ├── prompts.ts           # 提示词
│   └── config.ts            # 配置
├── package.json
└── README.md
```

### Go 项目结构

```
agent-name/
├── auditor/
│   └── auditor.go           # 主智能体
├── critic/
│   └── critic.go            # 子智能体
├── reviser/
│   └── reviser.go           # 子智能体
├── cmd/
│   └── main.go              # 入口
└── README.md
```

---

## 十一、部署方式

### 本地开发

**Python:**

```bash
adk run .          # CLI 模式
adk web            # Web UI 模式
```

**Java:**

```bash
mvn compile exec:java "-Dexec.args=--server.port=8080 \
    --adk.agents.source-dir=src/"
```

**TypeScript:**

```bash
npm install
npm run dev
```

**Go:**

```bash
go run ./cmd
```

### 云端部署

**1. Vertex AI Agent Engine（推荐）**

```python
# deployment/deploy.py
from vertexai.preview.reasoning_engines import AdkApp

app = AdkApp(agent=root_agent)
deployed_app = app.deploy(
    project=PROJECT_ID,
    location=LOCATION,
)
```

**2. Cloud Run（容器化）**

```bash
# 构建镜像
docker build -t gcr.io/$PROJECT_ID/agent:latest .

# 推送镜像
docker push gcr.io/$PROJECT_ID/agent:latest

# 部署到 Cloud Run
gcloud run deploy agent \
  --image gcr.io/$PROJECT_ID/agent:latest \
  --region us-central1
```

---

## 十二、智能体示例分类

### 按复杂度分类

**Easy（简单）:**

- Academic Research（学术研究）
- LLM Auditor（LLM 审计）
- Personalized Shopping（个性化购物）
- Currency Agent（货币兑换）

**Intermediate（中等）:**

- Customer Service（客户服务）
- RAG Agent（文档检索）
- Medical Pre-Authorization（医疗预授权）
- Software Bug Assistant（软件 Bug 助手）

**Advanced（高级）:**

- Data Science Agent（数据科学）
- Data Engineering Agent（数据工程）
- Deep Search（深度搜索）
- Travel Concierge（旅行管家）

### 按垂直领域分类

**Horizontal（通用）:**

- LLM Auditor
- RAG Agent
- Data Engineering
- Marketing Agency

**Retail（零售）:**

- Customer Service
- Personalized Shopping
- Brand Search Optimization

**Financial Services（金融）:**

- Financial Advisor
- FOMC Research
- Auto Insurance Agent
- Currency Agent

**Healthcare（医疗）:**

- Medical Pre-Authorization

**IT Support（IT 支持）:**

- Software Bug Assistant

---

## 十三、核心代码片段总结

### Agent 创建（四种语言）

**Python:**

```python
agent = Agent(
    model='gemini-2.0-flash',
    name='agent_name',
    instruction=PROMPT,
    tools=[tool1, tool2]
)
```

**Java:**

```java
LlmAgent.builder()
    .model("gemini-2.5-flash")
    .name("agent_name")
    .instruction(PROMPT)
    .tools(toolList)
    .build()
```

**TypeScript:**

```typescript
new LlmAgent({
  model: "gemini-2.0-flash",
  name: "agent_name",
  instruction: PROMPT,
  tools: [tool1, tool2],
});
```

**Go:**

```go
llmagent.New(llmagent.Config{
    Model: model,
    Name: "agent_name",
    Instruction: PROMPT,
    Tools: []tool.Tool{tool1, tool2},
})
```

### 多智能体创建

**Python:**

```python
SequentialAgent(
    name='multi_agent',
    sub_agents=[agent1, agent2]
)
```

**Go:**

```go
sequentialagent.New(sequentialagent.Config{
    AgentConfig: agent.Config{
        SubAgents: []agent.Agent{agent1, agent2},
    },
})
```

---

## 总结

ADK 智能体实现的核心是 **统一的抽象模型**：

1. **统一架构**：所有语言都遵循 `Agent(model, instruction, tools)` 结构
2. **工具生态**：支持内置工具（Google Search）、自定义工具、MCP 工具
3. **多智能体**：支持顺序、并行、层级等多种协作模式
4. **RAG 集成**：原生支持 Vertex AI RAG Engine
5. **云原生**：无缝部署到 Vertex AI 和 Cloud Run

**语言选择建议：**

- **Python**：生态最成熟，示例最多，推荐首选
- **Java**：企业级集成，MCP 支持完善
- **TypeScript**：前端集成友好
- **Go**：高性能场景
