"""Tests for shared evaluation helpers."""

from aai_cli.eval import compute_wer


def test_compute_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_compute_wer_different():
    result = compute_wer("hello world", "hello earth")
    assert 0.0 < result <= 1.0


def test_compute_wer_empty_reference():
    assert compute_wer("", "anything") == 0.0


def test_compute_wer_empty_hypothesis():
    assert compute_wer("hello world", "") == 1.0
