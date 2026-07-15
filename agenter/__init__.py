"""Agenter - Decode your intent into working code.

A backend-agnostic SDK for orchestrating autonomous coding agents.

Example:
    from agenter import AutonomousCodingAgent, CodingRequest

    agent = AutonomousCodingAgent()
    result = await agent.execute(
        CodingRequest(
            prompt="Create a hello.py file with a greet() function",
            cwd="/path/to/project"
        )
    )
    print(f"Status: {result.status}")
    print(f"Files modified: {list(result.files.keys())}")
"""

from importlib.metadata import version

__version__ = version("agenter")

from .coding_agent import AutonomousCodingAgent
from .data_models import (
    AgenterError,
    BackendError,
    Budget,
    BudgetExceededError,
    CodingEvent,
    CodingEventType,
    CodingRequest,
    CodingResult,
    CodingStatus,
    ConfigurationError,
    PathSecurityError,
    ToolError,
    ToolErrorCode,
    ToolExecutionError,
    ToolResult,
    ValidationError,
    ValidationResult,
    Verbosity,
)
from .file_system import FileOperations, PathResolver
from .logging import configure_logging
from .post_validators.syntax import SyntaxValidator
from .runtime import BudgetMeter, FileTracer, PersistentCodingSession, Tracer
from .tools import FunctionTool, Tool, tool

__all__ = [
    "AgenterError",
    "AutonomousCodingAgent",
    "BackendError",
    "Budget",
    "BudgetExceededError",
    "BudgetMeter",
    "CodingEvent",
    "CodingEventType",
    "CodingRequest",
    "CodingResult",
    "CodingStatus",
    "ConfigurationError",
    "FileOperations",
    "FileTracer",
    "FunctionTool",
    "PathResolver",
    "PathSecurityError",
    "PersistentCodingSession",
    "SyntaxValidator",
    "Tool",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolResult",
    "Tracer",
    "ValidationError",
    "ValidationResult",
    "Verbosity",
    "__version__",
    "configure_logging",
    "tool",
]
