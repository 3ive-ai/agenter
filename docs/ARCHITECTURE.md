# Architecture Design

## TL;DR

A summary of the system architecture, component design, and data flows of **Agenter**.

The SDK follows a simple layered design pattern from top user-facing layers to bottom coding agent backends:

* **Adapters** (LangGraph, PydanticAI)
* → **Facade** (AutonomousCodingAgent)
* → **Runtime** (CodingSession)
* → **Backends** (Anthropic, Claude Code, Codex, OpenHands, ACP).

---

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    User Applications                        │
├─────────────────────────────┬───────────────────────────────┤
│  LangGraph Adapter          │  PydanticAI Adapter           │
│  (adapters/langgraph.py)    │  (adapters/pydantic_ai.py)    │
└─────────────────────────────┴───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              AutonomousCodingAgent (Facade)                 │
│  - execute(request) -> CodingResult                         │
│  - stream_execute() -> AsyncIterator[CodingEvent]           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CodingSession                             │
│  - Manages iteration loop (code → validate → fix → retry)   │
│  - Emits events for observability                           │
│  - Enforces budget limits                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CodingBackend Protocol (Abstract)                                  │
├───────────────────────┬───────────────────────┬─────────────────────┬───────────────────┬─────────────┤
│  AnthropicSDKBackend  │   ClaudeCodeBackend   │    CodexBackend     │ OpenHandsBackend  │  ACPBackend │
│  (anthropic SDK)      │  (claude-code-sdk)    │  (openai-agents)    │  (openhands-sdk)  │  (ACP CLI)  │
└───────────────────────┴───────────────────────┴─────────────────────┴───────────────────┴─────────────┘
```

---

## Layer Responsibilities

### Layer 1: User Applications (Adapters)

The top layer represents adapters for open-ended agentic coding frameworks that integrate the SDK into different agentic workflows.

| Adapter | Framework | Use Case |
|---------|-----------|----------|
| `langgraph.py` | LangGraph | `create_coding_node()` for StateGraph workflows |
| `pydantic_ai.py` | PydanticAI | `CodingAgent` for direct execution with PydanticAI-like interface |

Adapters are thin wrappers and facades that translate between framework conventions and the SDK's unified interface.

### Layer 2: AutonomousCodingAgent (Facade)

The public API surface. Users interact only with this class.

**Responsibilities**:

- Instantiate and configure coding agent backends
- Provide sync/async execution methods
- Aggregate results from coding sessions
- Hide all internal complexity of the layers below

**Interface**:

- `execute(request: CodingRequest)` -> `CodingResult`
- `stream_execute(request: CodingRequest)` -> `AsyncIterator[CodingEvent]`

### Layer 3: CodingSession

The core orchestration logic for a coding session. Manages the iteration loop within the coding session.

**Responsibilities**:

- Execute a backend with a prompt
- Run validators on outputs
- Retry on validation failures (with error context in the updated prompt)
- Enforce budget limits (tokens, cost, time, iterations)
- Emit events for observability

**Iteration Loop**:

```text
┌─────────────────────────────────────────────────────────────┐
│                    CodingSession.run()                      │
├─────────────────────────────────────────────────────────────┤
│  FOREACH iteration (up to max_iterations):                  │
│    1. Check budget limits                                   │
│    2. Execute backend.execute(prompt)                       │
│    3. Collect files modified                                │
│    4. Run validators (e.g. syntax or security)              │
│    5. IF validation passed THEN return COMPLETED            │
│    6. IF budget exceeded THEN return BUDGET_EXCEEDED        │
│    7. Prepare a retry prompt with validation errors         │
│  END → return FAILED                                        │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: CodingBackend Protocol

Abstract interfaces implemented for each distinct coding agent backend.

**Responsibilities**:

- Connect to a backend (subprocess, SDK client)
- Execute prompts and obtain stream responses
- Track files modified
- Report token usage and estimated cost

**Backend Comparison**:

| Backend | Wraps | Agent Logic | Runtime | Key Feature |
|---------|-------|-------------|---------|-------------|
| AnthropicSDKBackend | [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) | Custom tool-use loop | HTTP (async client) | Full control, custom tools, AWS Bedrock |
| ClaudeCodeBackend | [`claude-code-sdk`](https://github.com/anthropics/claude-code-sdk-python) | Claude Code | Claude Code CLI | Battle-tested tools, AWS Bedrock, Google Vertex |
| CodexBackend | [`openai-agents`](https://github.com/openai/openai-agents-python) | Codex MCP server | MCP over stdio | OpenAI models, custom MCP tools, sandbox modes |
| OpenHandsBackend | [`openhands-sdk`](https://github.com/All-Hands-AI/OpenHands) | OpenHands agent | litellm | Any model, no sandbox (full access) |
| ACPBackend | [`agent-client-protocol`](https://agentclientprotocol.github.io/python-sdk/) | Any ACP-compatible agent | JSON-RPC over stdio | Interoperability with ACP agents |

**Which backend to use?**

| Use Case and Scenario | Recommended Backend | Config |
|----------|---------|--------|
| Custom tools, full control | `AnthropicSDKBackend` | `backend="anthropic-sdk"` (default) |
| Battle-tested Claude Code tools | `ClaudeCodeBackend` | `backend="claude-code", model="claude-sonnet-4-5-20250929", claude_max_thinking_tokens=8192` |
| Need skills, slash commands, MCP | `ClaudeCodeBackend` | `backend="claude-code"` |
| OpenAI models (o3, GPT-5-Codex) | `CodexBackend` | `backend="codex", model="gpt-5.4", codex_reasoning_effort="high"` |
| Custom MCP tools with OpenAI | `CodexBackend` | `backend="codex", codex_mcp_servers=[...]` |
| Any model via litellm | `OpenHandsBackend` | `backend="openhands", sandbox=False` |
| Any ACP-compatible agent process | `ACPBackend` | `backend="acp", acp_command="codex-acp", acp_args=["-c", "model=\"gpt-5.4\"", "-c", "model_reasoning_effort=\"high\""]` |

> **Note**: All backends default to `sandbox=True`. Use `sandbox=False` for unrestricted access.

---

## Backend SDK Patterns

Each backend SDK presents a different API surface. Here's how they work and how we abstract them:

### Anthropic SDK Backend (Implemented)

```python
# Actual implementation uses anthropic SDK directly
import anthropic

client = anthropic.AsyncAnthropic()

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16384,
    system="You are an autonomous coding agent...",
    tools=[...],  # File tools: read_file, write_file, edit_file
    messages=[{"role": "user", "content": "Fix the bug"}],
)

# Tool use loop handles file operations
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        # Continue conversation with tool result
```

**Key Abstractions**:
- `anthropic.AsyncAnthropic()` → async HTTP client
- Custom tool-use loop manages file operations
- Supports both Anthropic API and AWS Bedrock via boto3
- No subprocess or CLI dependency

### Claude Code SDK Backend (Implemented)

```python
# Uses claude-code-sdk (Claude Code as a library)
from claude_code_sdk import query, ClaudeCodeOptions

# Safe mode with native sandbox (default)
options = ClaudeCodeOptions(
    cwd="/path/to/project",
    model="claude-sonnet-4-5-20250929",
    max_thinking_tokens=8192,
    allowed_tools=["Read", "Edit", "Write", "Bash", "Glob"],
    sandbox={"enabled": True, "autoAllowBashIfSandboxed": True},
    permission_mode="default",
)

async for message in query(prompt="Fix the bug", options=options):
    # message is AssistantMessage, ToolResultMessage, or ResultMessage
    print(message)
```

**Key Abstractions**:
- `query()` → streams messages as `AsyncIterator`
- Built-in tools: Read, Edit, Write, Bash, Glob, Grep, etc.
- Native OS-level sandbox support
- Supports AWS Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`), Google Vertex, Microsoft Foundry
- Same tools that power Claude Code

**When to use**:
- You want production-ready tools maintained by Anthropic
- You need Claude Code features (skills, slash commands, MCP servers)
- You want native OS-level sandboxing

### Codex Backend (Implemented)

```python
# Uses openai-agents SDK MCPServerStdio for MCP communication
from agents.mcp import MCPServerStdio

mcp_server = MCPServerStdio(
    name="codex",
    params={"command": "codex", "args": ["mcp-server"]},
)
await mcp_server.connect()

# Call codex tool to start a session
result = await mcp_server.call_tool("codex", {
    "prompt": "Fix the bug",
    "cwd": "/path/to/project",
    "approval-policy": "never",
    "sandbox": "workspace-write",
    "model": "o3",
    "config": {"model_reasoning_effort": "high"},
})

# Continue with codex-reply for subsequent messages
result = await mcp_server.call_tool("codex-reply", {
    "prompt": "Now add tests",
    "conversationId": result["conversationId"],
})
```

**Key Abstractions**:
- `MCPServerStdio` → manages subprocess lifecycle
- MCP tools: `codex` (start session) and `codex-reply` (continue)
- Configurable sandbox and approval policies
- Custom MCP servers passed via `config` parameter

**When to use**:
- You want to use OpenAI reasoning models (o3, GPT-5-Codex)
- You need specific approval policies (untrusted, on-request, on-failure, never)
- You want to extend Codex with custom MCP tools

### Codex Backend Tool Limitations

The Codex backend runs custom tools in a **subprocess** via MCP. Tools are serialized using `cloudpickle`. 

**Note**: **Tools which capture unpicklable state will fail silently.**

**When tools fail to pickle:**
- Warning logged: `custom_tools_not_picklable`
- Tools are dropped — agent runs without them
- No error raised (silent failure)

**Writing Codex-compatible tools:**

✅ **Do:**
- Use module-level functions (not lambdas or closures)
- Import dependencies inside the function
- Pass dynamic state via environment variables or temp files

❌ **Don't:**
- Capture `self` or instance variables in closures
- Capture async clients, locks, or trace recorders
- Use lambda functions that capture outer scope

**Example — BAD (captures self):**
```python
def _create_tools(self):
    async def wrapper(inputs):
        return await self.some_method(inputs)  # Captures self - NOT picklable!
    return [FunctionTool(func=wrapper)]
```

**Example — GOOD (stateless module-level function):**
```python
# Module level - no closure, fully picklable
async def _stateless_wrapper(inputs: dict) -> str:
    from mymodule import some_function  # Import inside function
    return await some_function(inputs)

def _create_tools(self):
    return [FunctionTool(func=_stateless_wrapper)]
```

**Note**: For tools that need dynamic state (such as `mas_code` that changes per call), use environment variables or temporary files to pass data to the subprocess.

### ACP Backend (Implemented)

```python
agent = AutonomousCodingAgent(
    backend="acp",
    acp_command="codex-acp",
    acp_args=["-c", 'model="gpt-5.4"', "-c", 'model_reasoning_effort="high"'],
    acp_permission_policy="allow",
)
```

**Key Abstractions**:
- Spawns an ACP-compatible agent subprocess over stdio
- Creates an ACP session with the request working directory
- Sends prompts through `session/prompt`
- Serves ACP client file reads and writes inside the request working directory
- Adds Agenter's autonomous backend contract by default and auto-continues once
  if an interactive ACP agent asks for confirmation instead of editing
- Converts ACP session updates into Agenter backend messages
- Tracks modified files by comparing workspace snapshots

**When to use**:
- You want to run an ACP-compatible agent through Agenter's validation and event model
- You need an agent that already speaks ACP but does not have a native Agenter backend

**Security note**: ACP agents are external processes. Agenter observes file changes after execution, but sandbox enforcement depends on the launched ACP agent and its own configuration.

### Abstraction Strategy

The `CodingBackend` protocol normalizes backend differences:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CodingBackend Protocol                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  connect(cwd, allowed_write_paths, resume_session_id, output_type, system_prompt)│
│  execute(prompt: str)     → AsyncIterator[BackendMessage]                       │
│  modified_files()         → ModifiedFiles                                       │
│  usage()                  → Usage (tokens, cost)                                │
│  structured_output()      → BaseModel | None                                    │
│  refusal()                → RefusalMessage | None                               │
│  disconnect()             → Cleanup                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────┬───────────┼───────────┬─────────────┐
    │             │           │           │             │
    ▼             ▼           ▼           ▼             ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────────┐
│ Anthropic │ │ ClaudeAgt │ │  Codex    │ │   OpenHands      │
│ Backend   │ │ Backend   │ │ Backend   │ │   Backend        │
│ messages. │ │ query()   │ │ call_tool │ │ Conversation.run │
│ create()  │ │           │ │           │ │                  │
└───────────┘ └───────────┘ └───────────┘ └──────────────────┘
```

**Protocol Method Mappings**:

| Protocol Method | AnthropicSDKBackend | ClaudeCodeBackend | CodexBackend | OpenHandsBackend |
|-----------------|---------------------|-------------------|--------------|------------------|
| `connect(cwd)` | Set `PathResolver(cwd)` | Set `cwd` option | Set in MCP params | Set in Conversation |
| `execute(prompt)` | `messages.create()` + tool loop | `query(prompt, options)` | `call_tool("codex")` | `Conversation.run()` |
| `modified_files()` | Track via FileOperations | Parse from messages | Parse from result | Extract from events |
| `usage()` | Track tokens + litellm pricing | Extract from SDK | Extract from result | Track via litellm |
| `structured_output()` | Parse from tool call | Parse from tool call | Parse from response text | Parse from response text |
| `refusal()` | Capture Refusal tool | Capture Refusal tool | Capture Refusal tool | Capture Refusal tool |

---

## Data Flow

### Execution Flow

```text
┌──────────────────────────────────────────────────────────────────┐
│                         CodingRequest                            │
│  prompt, cwd, system_prompt, budget, output_type                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AutonomousCodingAgent.execute()               │
│  1. Create CodingSession with config                             │
│  2. Initialize backend                                           │
│  3. Run session.run(request)                                     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CodingSession.run()                         │
│  LOOP (max_iterations):                                          │
│    backend.execute(prompt) → stream BackendMessages              │
│    Collect files modified                                        │
│    Run validators → ValidationResult                             │
│    IF passed THEN COMPLETED                                      │
│    IF budget exceeded THEN BUDGET_EXCEEDED                       │
│    Prepare retry prompt with errors                              │
│  END LOOP → FAILED                                               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                         CodingResult                             │
│  status, files, summary, iterations, metrics                     │
└──────────────────────────────────────────────────────────────────┘
```

### Event Streams

For observability, the session emits events throughout its execution:

| Event Type | When | Emitted Data |
|------------|------|------|
| `SESSION_START` | Session initialized | cwd, model, max_iterations |
| `ITERATION_START` | Beginning iteration N | iteration number |
| `BACKEND_MESSAGE` | Message from backend | message_type, content, tool_name |
| `VALIDATION_START` | About to run validators | validators list, file_count |
| `VALIDATION_RESULT` | Result from one validator | validator name, passed, errors |
| `ITERATION_END` | Iteration complete | iteration, passed, files_modified, tokens, cost |
| `COMPLETED` | Task completed successfully | files, summary, iterations, tokens, cost |
| `FAILED` | Task failed | status, files, summary, iterations, tokens, cost |
| `REFUSED` | LLM declined the request | refusal reason, category |
| `SESSION_END` | Session finished (always emitted) | status, iterations, tokens, cost |

Event lifecycle:
```text
SESSION_START → ITERATION_START → BACKEND_MESSAGE* →
VALIDATION_START → VALIDATION_RESULT* → ITERATION_END →
(repeat iterations) → COMPLETED/FAILED/REFUSED → SESSION_END
```

---

## Validation Framework

### Validator Protocol

Validators run against modified files and return pass/fail with errors.

| Validator | Checks | Blocking | When to Use |
|-----------|--------|----------|-------------|
| SyntaxValidator | Python AST parsing | Yes | Always (default, instant feedback) |
| SecurityValidator | Bandit static analysis | No (advisory) | Security-sensitive code |

**Note:** Validators are configurable via `AutonomousCodingAgent(validators=[...])`. Default is `[SyntaxValidator()]`. `SecurityValidator` uses Bandit to detect vulnerabilities (eval, hardcoded secrets, SQL injection, etc.) and is non-blocking by default. Custom validators can be added by implementing the `Validator` protocol.

### Validation Flow

```text
Files Modified
      │
      ▼
┌─────────────┐
│   Syntax    │
│  Validator  │
└─────────────┘
      │
      ▼
   Errors?
      │
      ▼
ValidationResult
(passed, errors)
```

Validators run sequentially via `ValidatorChain`. Blocking validators (like `SyntaxValidator`) short-circuit on failure.

---

## Budget Enforcement

### Budget Criteria

| Criterion | Unit | Enforcement |
|-----------|------|-------------|
| Tokens | count | Sum across iterations, hard stop |
| Cost | USD | Estimated from token usage, hard stop |
| Time | seconds | Wall clock from session start, hard stop |
| Iterations | count | Loop limit, hard stop |

### Budget Checkpoints

Budget is checked:

1. before each iteration starts
2. after each backend execution completes
3. before retry prompt is sent

If any limit is exceeded, the session returns `BUDGET_EXCEEDED` status with an explanation.

---

## Configuration

### Config Hierarchy

```text
AutonomousCodingAgent
├── Backend Selection
│   └── backend: "anthropic-sdk" | "claude-code" | "codex" | "openhands" | "acp"
├── Security
│   └── sandbox: bool = True  (unified sandbox control)
├── Backend-Specific (anthropic-sdk backend)
│   ├── model: str  (e.g., "claude-sonnet-4-20250514")
│   ├── tools: list[Tool]  (custom tools)
│   └── use_anthropic_tools: bool  (use text_editor_20250728)
├── Backend-Specific (claude-code backend)
│   ├── allowed_tools: ["Read", "Edit", "Write", "Bash", ...]
│   ├── setting_sources: ["project", "user"]
│   └── claude_max_thinking_tokens: int
├── Backend-Specific (codex backend)
│   ├── codex_approval_policy: "never" | "on-request" | "on-failure" | "untrusted"
│   ├── codex_mcp_servers: list[CodexMCPServer]  (custom MCP tools)
│   └── codex_reasoning_effort: "minimal" | "low" | "medium" | "high"
├── Backend-Specific (openhands backend)
│   ├── model: str  (litellm format, e.g., "openai/gpt-4o")
│   └── sandbox: False  (required - no sandbox support)
├── Backend-Specific (acp backend)
│   ├── acp_command: str
│   ├── acp_args: list[str]
│   ├── acp_env: dict[str, str]
│   ├── acp_mcp_servers: list[Any]
│   ├── acp_permission_policy: "deny" | "allow"
│   └── acp_autonomous: bool
├── Safeguards
│   ├── max_iterations: 5
│   ├── max_tokens: 500_000
│   ├── max_cost_usd: 10.0
│   └── max_time_seconds: 3600
└── Validation
    └── validators: ["syntax", "security"]  (security is non-blocking by default)
```

---

## Error Handling

### Exception vs. Status

**Exceptions** are raised from unrecoverable errors that prevent execution:
- `ConfigurationError`: Invalid configuration (wrong backend name, missing keys)
- `BackendError`: Backend connection or execution failure
- `BudgetExceededError`: Only if `raise_on_budget_exceeded=True`

**Status codes** are returned for expected completion states:
- `CodingStatus.COMPLETED`: Task finished successfully within budget
- `CodingStatus.COMPLETED_WITH_LIMIT_EXCEEDED`: Task succeeded but exceeded budget limits
- `CodingStatus.BUDGET_EXCEEDED`: Stopped before completion due to limits (default behavior)
- `CodingStatus.REFUSED`: LLM declined the request
- `CodingStatus.FAILED`: Task couldn't complete (validation never passed)

| Category | Handling | Recovery |
|----------|----------|----------|
| Invalid configuration | Raise `ConfigurationError` | Fix config, retry |
| Backend connection failure | Raise `BackendError` | Check credentials/network |
| Backend execution error | Raise `BackendError` | Check backend health |
| Validation failure | Retry iteration with context | Max iterations limit |
| Budget exceeded | Return `BUDGET_EXCEEDED` status | Caller decides |
| LLM refusal | Return `REFUSED` status | Modify prompt |

---

## Security Considerations

### Unified Sandbox Mode

All backends default to `sandbox=True` for safe operation:

| Backend | `sandbox=True` (default) | `sandbox=False` |
|---------|--------------------------|-----------------|
| AnthropicSDKBackend | PathResolver enforces `allowed_write_paths` | Writes anywhere in cwd |
| ClaudeCodeBackend | Native OS-level sandbox | `bypassPermissions` mode |
| CodexBackend | `workspace-write` mode | `danger-full-access` mode |
| OpenHandsBackend | Not supported (raises error) | Full filesystem access |
| ACPBackend | Depends on launched ACP agent | Depends on launched ACP agent |

### Usage Examples

```python
# Safe mode (default) - all backends sandboxed
agent = AutonomousCodingAgent(backend="claude-code")

# Disable sandbox for full access
agent = AutonomousCodingAgent(backend="claude-code", sandbox=False)
```

### File System Scope

- `sandbox=True`: Operations restricted based on a given backend's sandbox implementation
- `sandbox=False`: Full filesystem access

### API Key Management

- Keys passed via environment variables or config
- Never logged or included in error messages
- Backend-specific key names: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

---

## Framework Adapters

### LangGraph Adapter

`create_coding_node()` returns an async function compatible with `StateGraph.add_node()`.

**Input**: State dict with `prompt` key (and optional `cwd`)

**Output**: State update with `coding_result` dict containing `CodingResult` fields

```python
from agenter.adapters.langgraph import create_coding_node, CodingState

graph = StateGraph(CodingState)
graph.add_node("coder", create_coding_node(cwd="./workspace"))
```

### PydanticAI Adapter

`CodingAgent` provides direct execution with a PydanticAI-like interface without extra LLM layers.

**Input**: Prompt and cwd via `run()` method

**Output**: `CodingResult` with status, files modified, and summary

```python
from agenter.adapters.pydantic_ai import CodingAgent

agent = CodingAgent(backend="anthropic-sdk")
result = await agent.run("Implement the feature", cwd="./workspace")
```
