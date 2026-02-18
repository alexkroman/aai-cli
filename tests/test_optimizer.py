"""Tests for optimizer module — prompt proposal and output extraction."""

from unittest.mock import MagicMock, patch

from aai_cli.optimizer import _propose_prompt


def _make_response(content: str):
    """Build a mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("aai_cli.optimizer.OpenAI")
def test_propose_prompt_returns_text(mock_openai_cls):
    """Should extract text from OpenAI-compatible response."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_response(
        "  Transcribe all spoken words verbatim.  "
    )

    history = [{"prompt": "baseline", "wer": 0.15, "worst_samples": []}]
    candidate, full_prompt = _propose_prompt(history, "fake-key")

    assert candidate == "Transcribe all spoken words verbatim."
    # Verify it called the AssemblyAI LLM gateway
    mock_openai_cls.assert_called_once_with(
        api_key="fake-key",
        base_url="https://llm-gateway.assemblyai.com/v1",
    )


@patch("aai_cli.optimizer.OpenAI")
def test_propose_prompt_empty_content(mock_openai_cls):
    """Edge case: None content — should return empty string."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_response(None)

    history = [{"prompt": "baseline", "wer": 0.15, "worst_samples": []}]
    candidate, _ = _propose_prompt(history, "fake-key")

    assert candidate == ""
