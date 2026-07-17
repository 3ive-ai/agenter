# Contributing to Agenter

Thank you for your interest in contributing to Agenter! We welcome contributions from the community to help make this the standard abstraction layer for autonomous coding agents.

## Development Environment

Agenter requires Python 3.12+ and uses [uv](https://docs.astral.sh/uv/) for dependency management.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/3ive-ai/agenter.git
    cd agenter
    ```

2.  **Install dependencies**
    ```bash
    uv sync --extra dev
    ```
    This installs the package in editable mode along with all development tools (pytest, ruff, mypy).

## Development Workflow

### Code Style

We use `ruff` for both linting and formatting, and `mypy` for static type checking.

*   **Format code:**
    ```bash
    uv run ruff format .
    ```

*   **Lint code:**
    ```bash
    uv run ruff check . --fix
    ```

*   **Type check:**
    ```bash
    uv run mypy agenter
    ```

Please ensure all checks pass before submitting a Pull Request.

### Testing

We use `pytest` for testing.

*   **Run all tests:**
    ```bash
    uv run pytest
    ```

*   **Run specific tests:**
    ```bash
    uv run pytest tests/test_config.py
    ```

*   **Run with coverage:**
    ```bash
    uv run pytest --cov=agenter
    ```

Note: Some integration tests require API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). These are skipped automatically if keys are not present in your environment.

## Project Structure

*   `agenter/`: Main package source code
    *   `coding_agent.py`: Main entry point (`AutonomousCodingAgent`)
    *   `coding_backends/`: Backend implementations (Anthropic, Claude Code, Codex, OpenHands)
    *   `adapters/`: Framework adapters (LangGraph, PydanticAI)
    *   `data_models/`: Pydantic models for types and schemas
    *   `post_validators/`: Security and syntax validation
    *   `runtime/`: Session management, budget tracking, tracing
*   `tests/`: Test suite
*   `docs/`: Documentation

## Submitting a Pull Request

1.  Fork the repository.
2.  Create a new branch for your feature or fix (`git checkout -b feature/amazing-feature`).
3.  Write code and add tests for your changes.
4.  Ensure all tests, linting, and type checks pass.
5.  Commit your changes using clear commit messages.
6.  Push to your fork and submit a Pull Request.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
