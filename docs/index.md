# Agenter

**Decode your intent into working code.**

Agenter is a backend-agnostic SDK for orchestrating autonomous AI coding agents. It provides a unified interface for working with multiple agent runtimes like Claude Code, Codex, OpenHands, and ACP-compatible agents.

## Quick Start

```python
import asyncio
from agenter import AutonomousCodingAgent, CodingRequest, Verbosity

async def main():
    agent = AutonomousCodingAgent(model="claude-sonnet-4-20250514")

    result = await agent.execute(
        CodingRequest(
            prompt="Create a FastAPI app with a health check endpoint",
            cwd="./workspace",
        ),
        verbosity=Verbosity.NORMAL
    )

    if result.status == "completed":
        print(f"Success! Modified {len(result.files)} files.")
        print(f"Cost: ${result.total_cost_usd:.4f}")

asyncio.run(main())
```

## Installation

```bash
# Default installation (anthropic-sdk backend)
pip install agenter

# With specific backends
pip install agenter[claude-code]
pip install agenter[codex]
pip install agenter[acp]

# With framework adapters
pip install agenter[adapters]
```

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

OVERVIEW
ARCHITECTURE
NAMING_CONVENTIONS
api
```

## Guides

- [Overview](OVERVIEW.md) - Introduction and core concepts
- [Architecture](ARCHITECTURE.md) - SDK design and backend protocol

## API Reference

- [API Reference](api.rst) - Auto-generated API documentation

## Links

- [GitHub Repository](https://github.com/moonsong-labs/agenter)
- [PyPI Package](https://pypi.org/project/agenter/)
- [Issue Tracker](https://github.com/moonsong-labs/agenter/issues)
