"""Comprehensive tests for Agenter."""

import time

import pytest

from agenter import (
    AutonomousCodingAgent,
    Budget,
    BudgetMeter,
    CodingRequest,
    SyntaxValidator,
    Verbosity,
)
from agenter.config import DEFAULT_MAX_ITERATIONS
from agenter.data_models import BudgetLimitType, ConfigurationError, UsageDelta


def _has_claude_agent_sdk() -> bool:
    """Check if claude-agent-sdk is installed."""
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# BudgetMeter Tests
# =============================================================================


class TestBudgetMeter:
    """Test BudgetMeter with real budget limits."""

    def test_iteration_limit(self):
        budget = Budget(max_iterations=3)
        tracker = BudgetMeter(budget)

        assert tracker.exceeded() == (False, None)
        tracker.add_iteration()
        assert tracker.exceeded() == (False, None)
        tracker.add_iteration()
        assert tracker.exceeded() == (False, None)
        tracker.add_iteration()
        assert tracker.exceeded() == (True, BudgetLimitType.ITERATIONS)

    def test_token_limit(self):
        budget = Budget(max_tokens=1000, max_iterations=100)
        tracker = BudgetMeter(budget)

        tracker.add_usage(UsageDelta(tokens=500))
        assert tracker.exceeded() == (False, None)
        tracker.add_usage(UsageDelta(tokens=500))
        assert tracker.exceeded() == (True, BudgetLimitType.TOKENS)

    def test_cost_limit(self):
        budget = Budget(max_cost_usd=1.0, max_iterations=100)
        tracker = BudgetMeter(budget)

        tracker.add_usage(UsageDelta(cost_usd=0.50))
        assert tracker.exceeded() == (False, None)
        tracker.add_usage(UsageDelta(cost_usd=0.50))
        assert tracker.exceeded() == (True, BudgetLimitType.COST)

    def test_time_limit(self):
        budget = Budget(max_time_seconds=0.1, max_iterations=100)
        tracker = BudgetMeter(budget)

        assert tracker.exceeded() == (False, None)
        time.sleep(0.15)
        assert tracker.exceeded() == (True, BudgetLimitType.TIME)

    def test_usage(self):
        budget = Budget(max_iterations=10)
        tracker = BudgetMeter(budget)

        tracker.add_iteration()
        tracker.add_iteration()
        tracker.add_usage(UsageDelta(tokens=500, cost_usd=0.5))

        usage = tracker.usage()
        assert usage["iterations"] == 2
        assert usage["tokens_used"] == 500
        assert usage["cost_usd"] == 0.5
        assert usage["elapsed_seconds"] > 0


# =============================================================================
# SyntaxValidator Tests
# =============================================================================


class TestSyntaxValidator:
    """Test SyntaxValidator with real Python code."""

    @pytest.mark.asyncio
    async def test_valid_single_file(self):
        validator = SyntaxValidator()
        files = {"app.py": "def main():\n    print('Hello')\n\nif __name__ == '__main__':\n    main()"}
        result = await validator.validate(files, "/tmp")
        assert result.passed
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_valid_multiple_files(self):
        validator = SyntaxValidator()
        files = {
            "models.py": "class User:\n    def __init__(self, name):\n        self.name = name",
            "utils.py": "def greet(name):\n    return f'Hello, {name}!'",
            "main.py": (
                "from models import User\nfrom utils import greet\n\nuser = User('Alice')\nprint(greet(user.name))"
            ),
        }
        result = await validator.validate(files, "/tmp")
        assert result.passed

    @pytest.mark.asyncio
    async def test_invalid_syntax_missing_colon(self):
        validator = SyntaxValidator()
        files = {"bad.py": "def hello()\n    return 'world'"}
        result = await validator.validate(files, "/tmp")
        assert not result.passed
        assert len(result.errors) == 1
        assert "bad.py" in result.errors[0]

    @pytest.mark.asyncio
    async def test_invalid_syntax_unclosed_paren(self):
        validator = SyntaxValidator()
        files = {"bad.py": "print('hello'"}
        result = await validator.validate(files, "/tmp")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_invalid_syntax_indentation(self):
        validator = SyntaxValidator()
        files = {"bad.py": "def foo():\nreturn 1"}
        result = await validator.validate(files, "/tmp")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_ignores_non_python_files(self):
        validator = SyntaxValidator()
        files = {
            "readme.md": "# This is {{{ not valid python",
            "config.json": '{"key": value}',
            "script.sh": "echo 'hello'",
        }
        result = await validator.validate(files, "/tmp")
        assert result.passed

    @pytest.mark.asyncio
    async def test_mixed_valid_and_non_python(self):
        validator = SyntaxValidator()
        files = {
            "app.py": "def main(): pass",
            "readme.md": "# Documentation",
        }
        result = await validator.validate(files, "/tmp")
        assert result.passed


# =============================================================================
# Budget Tests
# =============================================================================


class TestBudget:
    """Test Budget dataclass."""

    def test_default_max_iterations_matches_config(self):
        """Budget defaults should match config constants."""
        budget = Budget()
        assert budget.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_custom_values_override_defaults(self):
        """Custom budget values should override defaults."""
        budget = Budget(max_tokens=10000, max_cost_usd=1.0, max_time_seconds=60.0, max_iterations=10)
        assert budget.max_tokens == 10000
        assert budget.max_iterations == 10


# =============================================================================
# CodingRequest Tests
# =============================================================================


class TestCodingRequest:
    """Test CodingRequest dataclass."""

    def test_default_max_iterations_matches_config(self):
        """Request defaults should match config constants."""
        req = CodingRequest(prompt="test", cwd="/tmp")
        assert req.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_budget_overrides_max_iterations(self):
        """Budget should take precedence over max_iterations."""
        budget = Budget(max_tokens=5000, max_iterations=10)
        req = CodingRequest(prompt="test", cwd="/tmp", budget=budget)
        assert req.budget.max_iterations == 10


# =============================================================================
# CodingResult Tests
# =============================================================================


class TestUsageDeltaValidation:
    """Test UsageDelta negative value rejection (P0.2 fix)."""

    @pytest.mark.parametrize(("field", "value"), [("tokens", -100), ("cost_usd", -0.50)])
    def test_rejects_negative_values(self, field, value):
        """UsageDelta should reject negative values."""
        with pytest.raises(ValueError, match="cannot be negative"):
            UsageDelta(**{field: value})


# =============================================================================
# Backend Sandbox Tests
# =============================================================================


class TestAnthropicSDKBackendSandbox:
    """Test AnthropicSDKBackend sandbox modes."""

    def test_sandbox_true_is_default(self):
        """AnthropicSDKBackend defaults to sandbox=True."""
        from agenter.coding_backends.anthropic_sdk import AnthropicSDKBackend

        backend = AnthropicSDKBackend()
        assert backend._sandbox is True

    @pytest.mark.asyncio
    async def test_sandbox_true_enforces_write_paths(self):
        """sandbox=True enforces allowed_write_paths via PathResolver."""
        import tempfile

        from agenter.coding_backends.anthropic_sdk import AnthropicSDKBackend

        backend = AnthropicSDKBackend(sandbox=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir, allowed_write_paths=["*.py"])
            assert backend._path_resolver._allowed_write_paths == ["*.py"]

    @pytest.mark.asyncio
    async def test_sandbox_false_ignores_write_paths(self):
        """sandbox=False ignores allowed_write_paths (unrestricted within cwd)."""
        import tempfile

        from agenter.coding_backends.anthropic_sdk import AnthropicSDKBackend

        backend = AnthropicSDKBackend(sandbox=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir, allowed_write_paths=["*.py"])
            # Should ignore write restrictions
            assert backend._path_resolver._allowed_write_paths is None


class TestClaudeCodeBackendSandbox:
    """Test ClaudeCodeBackend sandbox modes."""

    def test_sandbox_true_is_default(self):
        """ClaudeCodeBackend defaults to sandbox=True (safe mode)."""
        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(sandbox=True)
        assert backend._sandbox is True

    def test_sandbox_false_for_unrestricted_access(self):
        """ClaudeCodeBackend(sandbox=False) enables unrestricted access."""
        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(sandbox=False)
        assert backend._sandbox is False

    def test_sandbox_dict_for_custom_config(self):
        """ClaudeCodeBackend accepts custom sandbox config dict."""
        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        custom_config = {"enabled": True, "autoAllowBashIfSandboxed": False}
        backend = ClaudeCodeBackend(sandbox=custom_config)
        assert backend._sandbox == custom_config

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_claude_agent_sdk(), reason="claude-agent-sdk not installed")
    async def test_sandbox_true_passes_config_to_sdk(self):
        """sandbox=True passes sandbox config and permission_mode=default to SDK."""
        import tempfile
        from unittest.mock import patch

        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(sandbox=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir)

            captured_options = None

            async def fake_query(prompt, options):
                nonlocal captured_options
                captured_options = options
                return
                yield  # make it a generator

            with patch("claude_agent_sdk.query", fake_query):
                async for _ in backend.execute("test"):
                    pass

            assert captured_options.permission_mode == "default"
            assert captured_options.sandbox["enabled"] is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_claude_agent_sdk(), reason="claude-agent-sdk not installed")
    async def test_sandbox_false_passes_bypass_to_sdk(self):
        """sandbox=False passes permission_mode=bypassPermissions to SDK."""
        import tempfile
        from unittest.mock import patch

        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(sandbox=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir)

            captured_options = None

            async def fake_query(prompt, options):
                nonlocal captured_options
                captured_options = options
                return
                yield

            with patch("claude_agent_sdk.query", fake_query):
                async for _ in backend.execute("test"):
                    pass

            assert captured_options.permission_mode == "bypassPermissions"


class TestCodexBackendSandbox:
    """Test CodexBackend sandbox modes."""

    def test_sandbox_workspace_write_default(self):
        """CodexBackend defaults to workspace-write sandbox."""
        from agenter.coding_backends.codex import CodexBackend

        backend = CodexBackend()
        assert backend._sandbox == "workspace-write"

    def test_sandbox_accepts_valid_modes(self):
        """CodexBackend accepts valid sandbox modes."""
        from agenter.coding_backends.codex import CodexBackend

        for mode in ["read-only", "workspace-write", "danger-full-access"]:
            backend = CodexBackend(sandbox=mode)
            assert backend._sandbox == mode

    def test_sandbox_rejects_invalid_mode(self):
        """CodexBackend rejects invalid sandbox mode."""
        from agenter.coding_backends.codex import CodexBackend

        with pytest.raises(ConfigurationError, match="sandbox"):
            CodexBackend(sandbox="invalid")


# =============================================================================
# Agent Configuration Tests
# =============================================================================


class TestAgentConfiguration:
    """Test AutonomousCodingAgent configuration."""

    def test_unknown_backend_raises(self):
        """Test that unknown backend raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            AutonomousCodingAgent(backend="unknown")

    def test_agent_default_validators(self):
        """Test agent has SyntaxValidator by default."""
        agent = AutonomousCodingAgent()
        assert len(agent._validators) == 1
        assert isinstance(agent._validators[0], SyntaxValidator)

    def test_agent_custom_validators(self):
        """Test agent can be created with no validators."""
        agent = AutonomousCodingAgent(validators=[])
        assert len(agent._validators) == 0

    def test_quiet_verbosity_reconfigures_agenter_logging(self, monkeypatch):
        """QUIET suppresses structlog even after earlier verbose configuration."""
        calls = []
        agent = AutonomousCodingAgent(validators=[])
        monkeypatch.setattr(agent, "_create_backend", lambda: object())
        monkeypatch.setattr("agenter.coding_agent.configure_logging", lambda **kwargs: calls.append(kwargs))

        agent._setup_session(Verbosity.QUIET, log_dir=None)

        assert calls == [{"quiet": True}]


# =============================================================================
# ModifiedFiles Tests
# =============================================================================


class TestModifiedFiles:
    """Test ContentModifiedFiles and PathsModifiedFiles types."""

    def test_content_modified_files(self):
        """Test ContentModifiedFiles with full content."""
        from agenter.data_models import ContentModifiedFiles

        files = ContentModifiedFiles(files={"a.py": "print(1)", "b.py": "print(2)"})
        assert len(files) == 2
        assert "a.py" in files
        assert "missing.py" not in files
        assert files.paths() == ["a.py", "b.py"]
        assert files.content("a.py") == "print(1)"
        assert not files.paths_only

    def test_paths_modified_files(self):
        """PathsModifiedFiles should return None for content."""
        from agenter.data_models import PathsModifiedFiles

        files = PathsModifiedFiles(file_paths=["test.py", "main.py"])
        assert files.content("test.py") is None
        assert "test.py" in files
        assert files.paths_only
        assert len(files) == 2


# =============================================================================
# ValidatorChain Tests
# =============================================================================


class TestValidatorChain:
    """Test ValidatorChain with multiple validators."""

    @pytest.mark.asyncio
    async def test_chain_aggregates_results_from_validators(self):
        """Chain should aggregate results from all validators."""
        from agenter.post_validators.chain import ValidatorChain

        chain = ValidatorChain([SyntaxValidator()])
        passed_result = await chain.validate({"test.py": "print(1)"}, "/tmp")
        failed_result = await chain.validate({"test.py": "def broken(:"}, "/tmp")

        assert passed_result.passed
        assert not failed_result.passed
        assert len(failed_result.errors) > 0

    @pytest.mark.asyncio
    async def test_empty_chain_passes(self):
        """Empty validator chain should pass all inputs."""
        from agenter.post_validators.chain import ValidatorChain

        chain = ValidatorChain([])
        result = await chain.validate({"test.py": "invalid python (:"}, "/tmp")
        assert result.passed
