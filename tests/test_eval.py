"""Tests for shared evaluation helpers."""

from unittest.mock import MagicMock, patch

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
    assert result["laser_score"] == 1.0


def test_laser_score_empty_reference():
    result = compute_laser_score("", "anything")
    assert result["laser_score"] == 1.0


def test_laser_score_empty_hypothesis():
    result = compute_laser_score("hello world", "")
    assert result["laser_score"] == 0.0
    assert result["word_count"] == 2
    assert len(result["major_errors"]) > 0


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

    assert result["laser_score"] == 0.786
    assert result["word_count"] == 7
    assert result["major_errors"] == ["red vs blue"]
    assert result["minor_errors"] == ["recieve vs receive"]
    assert result["no_penalty_errors"] == ["gonna vs going to"]
    assert result["total_penalty"] == 1.5

    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["response_format"] is LaserResponse
    assert call_kwargs["temperature"] == 0.0


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
    assert result["laser_score"] == 1.0
    assert result["total_penalty"] == 0.0
