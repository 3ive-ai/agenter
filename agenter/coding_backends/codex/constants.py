"""Constants for Codex backend."""

from typing import Final

# Default model for Codex
CODEX_DEFAULT_MODEL: Final = "gpt-5.4"

# Default approval policy (autonomous operation)
CODEX_DEFAULT_APPROVAL_POLICY: Final = "never"

# Default sandbox mode (workspace-scoped writes)
CODEX_DEFAULT_SANDBOX: Final = "workspace-write"

# Valid reasoning effort levels for Codex/OpenAI reasoning models.
CODEX_REASONING_EFFORTS: Final = frozenset({"minimal", "low", "medium", "high"})

# Valid approval policies
# - untrusted: Requires approval for all commands
# - on-request: Requires approval for certain sensitive operations
# - on-failure: Only requires approval if a command fails
# - never: Auto-approve all tool calls (autonomous mode)
CODEX_APPROVAL_POLICIES: Final = frozenset(
    {
        "untrusted",
        "on-request",
        "on-failure",
        "never",
    }
)

# Valid sandbox modes
# - read-only: No file writes allowed
# - workspace-write: Allow writes only within working directory
# - danger-full-access: Full filesystem access
CODEX_SANDBOX_MODES: Final = frozenset(
    {
        "read-only",
        "workspace-write",
        "danger-full-access",
    }
)

# MCP tool names exposed by Codex MCP server
CODEX_TOOL_START: Final = "codex"
CODEX_TOOL_REPLY: Final = "codex-reply"

# Tool names that modify files (for tracking modified paths)
FILE_MODIFICATION_TOOLS: Final = frozenset({"write_file", "edit_file", "create_file", "Write", "Edit"})

# Possible argument keys for file paths in tool calls
FILE_PATH_ARG_KEYS: Final = frozenset({"path", "file_path", "filepath", "file"})

# Default timeout for MCP session in seconds.
# LLM calls can take a while for complex tasks; 1 hour is reasonable.
# Users can override via CodexBackend(timeout=...).
CODEX_DEFAULT_TIMEOUT_SECONDS: Final = 3600.0  # 1 hour
