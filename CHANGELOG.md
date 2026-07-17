# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2026-07-17

### Added
- Persistent ACP sessions through `AutonomousCodingAgent.open_session()` with serialized follow-ups
- Stable ACP session/request trace identifiers and per-request plus cumulative usage metadata
- Explicit usage availability so an ACP adapter's missing metrics are not mistaken for measured zero usage
- ACP `session/resume` support with legacy `session/load` fallback
- Active ACP prompt cancellation without discarding the persistent session
- Live ACP session updates through a host callback

### Changed
- Split backend connection lifetime from individual `CodingSession` requests while preserving one-shot `execute()` behavior
- Track ACP file changes separately for the latest turn, high-level request, and full session
- Require `agent-client-protocol` 0.10 or later so ACP transport frames are not arbitrarily truncated

### Fixed
- Type checking when the optional ACP dependency is not installed

## [0.1.3] - 2026-05-19

### Added
- Agent Client Protocol backend for ACP-compatible agent subprocesses

### Changed
- Pass model and thinking-token settings through to the Claude Code backend
- Support reasoning-effort configuration in the Codex backend

### Fixed
- Exclude bytecode caches from file-change results for path-only backends

## [0.1.2] - 2026-03-24

### Added
- Configurable default backend via `ACA_DEFAULT_BACKEND` environment variable
- OpenClaw skill integration (`integrations/openclaw/`) with CLI bridge for autonomous coding from any messaging channel
- Codex backend: filesystem diff to catch all modified files, not just event-parsed ones
- Codex backend: automatic `OPENAI_API_KEY` → codex auth sync on connect
- Codex backend: structured output support via prompt injection
- Codex backend: pre-flight check for codex CLI installation

### Fixed
- `AutonomousCodingAgent(backend=...)` now accepts `None` to use env var default instead of hardcoding `"anthropic-sdk"`

### Security
- Added warning about litellm 1.82.7–1.82.8 supply chain compromise in optional dependency comment

## [0.1.0] - 2025-01-22

### Added
- Initial release of Agenter SDK
- Core `Agent` abstraction with unified interface for coding agents
- Support for multiple backends:
  - Anthropic API (direct)
  - AWS Bedrock
  - Claude Code CLI
  - OpenAI Codex
- Budget controls with `BudgetConfig` for cost limits and token tracking
- Security validation framework with configurable validators
- Path validation to restrict file system access
- Streaming support with real-time output callbacks
- Framework adapters for PydanticAI and LangGraph integration
- Comprehensive type hints and Pydantic models
- Structured logging with structlog

[Unreleased]: https://github.com/3ive-ai/agenter/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/3ive-ai/agenter/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/3ive-ai/agenter/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/3ive-ai/agenter/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/3ive-ai/agenter/releases/tag/v0.1.0
