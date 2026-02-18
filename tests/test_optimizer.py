"""Tests for optimizer module — prompt proposal and output extraction."""

from unittest.mock import MagicMock, patch

from aai_cli.optimizer import REFLECTION_PROMPT, _propose_prompt, _refine_reflection_prompt


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


@patch("aai_cli.optimizer.OpenAI")
def test_refine_reflection_prompt(mock_openai_cls):
    """Should call LLM with meta-prompt and return refined reflection prompt."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    new_template = (
        "You are an expert optimizing ASR prompts.\n\n"
        "{trajectory}\n\n{error_samples}\n\n{stagnation_warning}"
    )
    mock_client.chat.completions.create.return_value = _make_response(f"  {new_template}  ")

    history = [
        {"prompt": "baseline", "wer": 0.20, "worst_samples": []},
        {"prompt": "improved", "wer": 0.15, "worst_samples": []},
        {"prompt": "worse", "wer": 0.25, "worst_samples": []},
    ]

    result = _refine_reflection_prompt(REFLECTION_PROMPT, history, "fake-key")

    assert result == new_template
    # Verify the meta-prompt was sent to the LLM
    call_args = mock_client.chat.completions.create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "Current Reflection Prompt Template" in prompt_content
    assert "IMPROVED WER" in prompt_content
    assert "DID NOT improve WER" in prompt_content


@patch("aai_cli.optimizer.OpenAI")
def test_refine_reflection_prompt_empty_history(mock_openai_cls):
    """With only baseline in history, improved/failed should show (none)."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_response("new prompt")

    history = [{"prompt": "baseline", "wer": 0.20, "worst_samples": []}]

    result = _refine_reflection_prompt(REFLECTION_PROMPT, history, "fake-key")

    assert result == "new prompt"
    call_args = mock_client.chat.completions.create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "(none)" in prompt_content
