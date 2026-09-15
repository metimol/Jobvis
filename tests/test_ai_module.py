"""Unit tests for Groq AI module, agent initialization, and process_text handling."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_groq_env():
    """Ensure GROQ_API_KEY is present in environment for module imports."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_fixture_key_12345"}):
        yield


def test_missing_groq_api_key_error():
    """Verify MissingGroqAPIKeyError exception properties and backward-compatible alias."""
    import importlib
    import sys

    if "const" in sys.modules:
        del sys.modules["const"]
    const = importlib.import_module("const")

    err = const.MissingGroqAPIKeyError()
    assert str(err) == "GROQ_API_KEY is not set"
    assert issubclass(const.MissingGroqAPIKeyError, Exception)


def test_const_import_behavior():
    """Verify const module raises MissingGroqAPIKeyError when GROQ_API_KEY is missing."""
    import importlib
    import sys

    with patch.dict(os.environ, {}, clear=True):
        if "const" in sys.modules:
            del sys.modules["const"]

        with pytest.raises(Exception) as excinfo:
            importlib.import_module("const")

        assert "GROQ_API_KEY is not set" in str(excinfo.value)
        assert excinfo.type.__name__ == "MissingGroqAPIKeyError"


def test_ai_config_model_initialization():
    """Verify ai.config initializes ChatGroq with openai/gpt-oss-120b."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_api_key_1234567890"}):
        import sys

        for mod in ["const", "ai.config", "ai"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import ai.config

        assert ai.config.model is not None
        assert ai.config.model.model_name == "openai/gpt-oss-120b"
        assert ai.config.agent is not None


@pytest.mark.asyncio
async def test_ask_agent_string_response():
    """Verify ask_agent correctly returns string content from agent."""
    from ai.process_text import ask_agent

    mock_msg = MagicMock()
    mock_msg.content = "Here are the best jobs for Python developer."
    mock_msg.additional_kwargs = {}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})

    with patch("ai.process_text.agent", fake_agent):
        res = await ask_agent("Find me Python jobs")
        assert res == "Here are the best jobs for Python developer."


@pytest.mark.asyncio
async def test_ask_agent_list_content_response():
    """Verify ask_agent correctly parses list content format from agent."""
    from ai.process_text import ask_agent

    mock_msg = MagicMock()
    mock_msg.content = [{"text": "Found 3 jobs matching your criteria."}]
    mock_msg.additional_kwargs = {}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})

    with patch("ai.process_text.agent", fake_agent):
        res = await ask_agent("Find jobs")
        assert res == "Found 3 jobs matching your criteria."


@pytest.mark.asyncio
async def test_ask_agent_reasoning_content_fallback():
    """Verify ask_agent falls back to reasoning_content if content is empty."""
    from ai.process_text import ask_agent

    mock_msg = MagicMock()
    mock_msg.content = ""
    mock_msg.additional_kwargs = {"reasoning_content": "Reasoning step: recommended 2 roles."}

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})

    with patch("ai.process_text.agent", fake_agent):
        res = await ask_agent("Find jobs")
        assert res == "Reasoning step: recommended 2 roles."


@pytest.mark.asyncio
async def test_ask_agent_exception_handling():
    """Verify ask_agent logs warning and returns ai_error on exception."""
    from ai.process_text import ask_agent

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(side_effect=RuntimeError("Groq API rate limit"))

    with patch("ai.process_text.agent", fake_agent):
        res = await ask_agent("Find jobs")
        assert res == "ai_error"
