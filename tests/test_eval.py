"""Tests for shared evaluation helpers."""

from unittest.mock import MagicMock, patch

import pytest

from aai_cli.eval import LaserResponse, compute_laser_score, compute_wer


def test_compute_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_compute_wer_different():
    result = compute_wer("hello world", "hello earth")
    assert 0.0 < result <= 1.0


def test_compute_wer_empty_reference():
    assert compute_wer("", "anything") == 0.0


def test_compute_wer_empty_hypothesis():
    assert compute_wer("hello world", "") == 1.0


# --- LASER score tests ---


def test_laser_score_empty_both():
    result = compute_laser_score("", "")
    assert result.laser_score == 1.0


def test_laser_score_empty_reference():
    result = compute_laser_score("", "anything")
    assert result.laser_score == 1.0


def test_laser_score_empty_hypothesis():
    result = compute_laser_score("hello world", "")
    assert result.laser_score == 0.0
    assert result.word_count == 2
    assert len(result.major_errors) > 0


@patch("aai_cli.eval.litellm.completion")
def test_laser_score_structured_output(mock_completion):
    """Should parse structured JSON response from LLM."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = LaserResponse(
        word_count=7,
        no_penalty_errors=["gonna vs going to"],
        major_errors=["red vs blue"],
        minor_errors=["recieve vs receive"],
        total_penalty=1.5,
        laser_score=0.786,
    ).model_dump_json()
    mock_completion.return_value = mock_response

    result = compute_laser_score(
        "The red car is going to arrive",
        "The blue car is gonna recieve",
        api_key="test-key",
    )

    assert result.laser_score == 0.786
    assert result.word_count == 7
    assert result.major_errors == ["red vs blue"]
    assert result.minor_errors == ["recieve vs receive"]
    assert result.no_penalty_errors == ["gonna vs going to"]
    assert result.total_penalty == 1.5

    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["response_format"] is LaserResponse
    assert call_kwargs["temperature"] == 0.0


# --- _llm_call_with_retry tests ---


@patch("aai_cli.eval.litellm.completion")
def test_llm_call_with_retry_success_first_try(mock_completion):
    """Should return on first successful call."""
    mock_completion.return_value = "ok"
    from aai_cli.eval import _llm_call_with_retry

    result = _llm_call_with_retry([{"role": "user", "content": "hi"}], "model", "key")
    assert result == "ok"
    assert mock_completion.call_count == 1


@patch("aai_cli.eval.time.sleep")
@patch("aai_cli.eval.litellm.completion")
def test_llm_call_with_retry_retries_then_succeeds(mock_completion, mock_sleep):
    """Should retry on failure and return on eventual success."""
    mock_completion.side_effect = [RuntimeError("fail"), RuntimeError("fail"), "ok"]
    from aai_cli.eval import _llm_call_with_retry

    result = _llm_call_with_retry([{"role": "user", "content": "hi"}], "model", "key")
    assert result == "ok"
    assert mock_completion.call_count == 3
    assert mock_sleep.call_count == 2


@patch("aai_cli.eval.time.sleep")
@patch("aai_cli.eval.litellm.completion")
def test_llm_call_with_retry_all_fail(mock_completion, mock_sleep):
    """Should raise after exhausting all retries."""
    mock_completion.side_effect = RuntimeError("always fail")
    from aai_cli.eval import _llm_call_with_retry

    with pytest.raises(RuntimeError, match="LASER scoring failed after 3 attempts"):
        _llm_call_with_retry([{"role": "user", "content": "hi"}], "model", "key")


# --- _normalize_laser_response tests ---


def test_normalize_laser_response_basic():
    from aai_cli.eval import _normalize_laser_response

    content = LaserResponse(
        word_count=5,
        no_penalty_errors=[],
        major_errors=["x"],
        minor_errors=[],
        total_penalty=1.0,
        laser_score=0.8,
    ).model_dump_json()
    parsed = _normalize_laser_response(content)
    assert parsed.laser_score == 0.8
    assert parsed.major_errors == ["x"]


def test_normalize_laser_response_aliases():
    """Should map field aliases to canonical names."""
    import json

    from aai_cli.eval import _normalize_laser_response

    raw = json.dumps(
        {
            "word_count": 3,
            "non_penalizable_errors": ["ok"],
            "major_penalizable_errors": ["bad"],
            "minor_penalizable_errors": ["meh"],
            "total_penalty": 1.0,
            "score": 0.7,
        }
    )
    parsed = _normalize_laser_response(raw)
    assert parsed.no_penalty_errors == ["ok"]
    assert parsed.major_errors == ["bad"]
    assert parsed.minor_errors == ["meh"]
    assert parsed.laser_score == 0.7


def test_normalize_laser_response_clamps_score():
    import json

    from aai_cli.eval import _normalize_laser_response

    raw = json.dumps(
        {
            "word_count": 2,
            "no_penalty_errors": [],
            "major_errors": [],
            "minor_errors": [],
            "total_penalty": 0,
            "laser_score": 1.5,
        }
    )
    parsed = _normalize_laser_response(raw)
    assert parsed.laser_score == 1.0


def test_normalize_laser_response_string_to_list():
    """Should convert string values in list fields to lists."""
    import json

    from aai_cli.eval import _normalize_laser_response

    raw = json.dumps(
        {
            "word_count": 2,
            "no_penalty_errors": "some error",
            "major_errors": '["real error"]',
            "minor_errors": "",
            "total_penalty": 1.0,
            "laser_score": 0.5,
        }
    )
    parsed = _normalize_laser_response(raw)
    assert parsed.no_penalty_errors == ["some error"]
    assert parsed.major_errors == ["real error"]
    assert parsed.minor_errors == []


def test_normalize_laser_response_none_raises():
    from aai_cli.eval import _normalize_laser_response

    with pytest.raises(RuntimeError, match="empty content"):
        _normalize_laser_response(None)


# --- _laser_response_to_result tests ---


def test_laser_response_to_result():
    from aai_cli.eval import _laser_response_to_result

    resp = LaserResponse(
        word_count=10,
        no_penalty_errors=["a"],
        major_errors=["b"],
        minor_errors=["c"],
        total_penalty=2.0,
        laser_score=0.8,
    )
    result = _laser_response_to_result(resp)
    assert result.laser_score == 0.8
    assert result.word_count == 10
    assert result.no_penalty_errors == ["a"]


@patch("aai_cli.eval.litellm.completion")
def test_laser_score_perfect(mock_completion):
    """Perfect match should score 1.0."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = LaserResponse(
        word_count=2,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=0.0,
        laser_score=1.0,
    ).model_dump_json()
    mock_completion.return_value = mock_response

    result = compute_laser_score("hello world", "hello world", api_key="test-key")
    assert result.laser_score == 1.0
    assert result.total_penalty == 0.0
