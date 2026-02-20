"""Tests for streaming.py — streaming model detection and parameter building."""

from aai_cli.streaming import (
    PROMPT_SUPPORTED_MODELS,
    STREAMING_SPEECH_MODELS,
    _build_streaming_params,
    is_streaming_model,
)

# ---------------------------------------------------------------------------
# is_streaming_model
# ---------------------------------------------------------------------------


def test_is_streaming_model_true():
    assert is_streaming_model("u3-pro") is True


def test_is_streaming_model_false():
    assert is_streaming_model("universal-3-pro") is False


def test_is_streaming_model_all_known():
    for model in STREAMING_SPEECH_MODELS:
        assert is_streaming_model(model) is True


# ---------------------------------------------------------------------------
# _build_streaming_params
# ---------------------------------------------------------------------------


def test_build_streaming_params_with_prompt():
    params = _build_streaming_params("u3-pro", "Transcribe verbatim.")
    assert params.prompt == "Transcribe verbatim."
    assert params.sample_rate == 16000


def test_build_streaming_params_no_prompt_for_unsupported():
    params = _build_streaming_params("universal-streaming-english", "some prompt")
    assert not hasattr(params, "prompt") or params.prompt is None


def test_build_streaming_params_empty_prompt():
    """Empty prompt should not be set even for supported models."""
    params = _build_streaming_params("u3-pro", "")
    assert not hasattr(params, "prompt") or params.prompt is None or params.prompt == ""


def test_build_streaming_params_custom_sample_rate():
    params = _build_streaming_params("u3-pro", "test", sample_rate=8000)
    assert params.sample_rate == 8000


def test_prompt_supported_models_subset():
    """All prompt-supported models should also be streaming models."""
    for model in PROMPT_SUPPORTED_MODELS:
        assert model in STREAMING_SPEECH_MODELS
