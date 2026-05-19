"""Tests for the config module."""

import os
from unittest.mock import patch

import pytest

from agenter.config import (
    default_backend,
    default_model,
    is_bedrock,
    parse_backend_spec,
)
from agenter.data_models import ConfigurationError


class TestIsBedrock:
    """Test is_bedrock function."""

    def test_returns_false_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            assert is_bedrock() is False

    def test_returns_true_when_env_set(self):
        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "some-token"}):
            result = is_bedrock()
            assert result is True


class TestDefaultModel:
    """Test default_model function."""

    def test_returns_different_models_based_on_environment(self):
        """Default model should differ based on Bedrock environment."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            anthropic_model = default_model()

        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "token"}):
            bedrock_model = default_model()

        # Models should be different for different backends
        assert anthropic_model != bedrock_model


class TestDefaultBackend:
    """Test default_backend function."""

    def test_returns_anthropic_sdk_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ACA_DEFAULT_BACKEND", None)
            assert default_backend() == "anthropic-sdk"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"ACA_DEFAULT_BACKEND": "codex"}):
            assert default_backend() == "codex"

    def test_acp_env_var_override(self):
        with patch.dict(os.environ, {"ACA_DEFAULT_BACKEND": "acp"}):
            assert default_backend() == "acp"

    def test_invalid_env_var_raises(self):
        with (
            patch.dict(os.environ, {"ACA_DEFAULT_BACKEND": "invalid"}),
            pytest.raises(ConfigurationError, match="Invalid ACA_DEFAULT_BACKEND"),
        ):
            default_backend()


class TestParseBackendSpec:
    """Test backend spec parsing."""

    def test_accepts_acp_backend(self):
        assert parse_backend_spec("agenter:acp") == ("agenter", "acp", None)
