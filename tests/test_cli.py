"""Tests for CLI entry point."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from aai_cli.datasets import _apply_hf_dataset_override, load_all_datasets
from aai_cli.repl import _BANNER
from aai_cli.service import (
    get_env_key,
    load_config,
    run_eval,
)
from aai_cli.types import AudioSample, DatasetOptions, LaserResult, TranscriptionResult

# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


def test_banner_renders():
    """Verify banner string contains expected text."""
    assert "What would you like to build?" in _BANNER


def test_get_env_key_returns_value(monkeypatch):
    """Env var present — should return its value."""
    monkeypatch.setenv("TEST_KEY_ABC", "secret123")
    assert get_env_key("TEST_KEY_ABC") == "secret123"


def test_get_env_key_raises_when_missing(monkeypatch):
    """Env var absent — should raise ValueError."""
    monkeypatch.delenv("TEST_KEY_MISSING", raising=False)
    with pytest.raises(ValueError, match="Missing env var"):
        get_env_key("TEST_KEY_MISSING")


def test_default_config_loads():
    """OmegaConf.load should load config.yaml and have the expected structure."""
    cfg = load_config()
    assert "earnings22" in cfg.datasets
    assert "peoples" in cfg.datasets
    assert "ami" in cfg.datasets
    assert "loquacious" in cfg.datasets
    assert "gigaspeech" in cfg.datasets
    assert "tedlium" in cfg.datasets
    assert "commonvoice" in cfg.datasets
    assert "librispeech" in cfg.datasets
    assert "librispeech-other" in cfg.datasets
    assert cfg.optimization.samples == 100
    assert cfg.optimization.iterations == 50
    assert cfg.optimization.num_threads == 12


def test_config_override():
    """load_config override should change optimization.samples."""
    cfg = load_config(overrides={"optimization": {"samples": 50}})
    assert cfg.optimization.samples == 50


# ---------------------------------------------------------------------------
# Eval config tests
# ---------------------------------------------------------------------------


def test_eval_config_defaults():
    """Eval section should have expected defaults."""
    cfg = load_config()
    assert cfg.eval.max_samples == 50
    assert cfg.eval.prompt == "Transcribe verbatim."
    assert cfg.eval.num_threads == 12


def test_eval_config_override():
    """load_config overrides should change eval settings."""
    cfg = load_config(
        overrides={
            "eval": {
                "max_samples": 10,
                "prompt": "Transcribe exactly.",
            },
        }
    )
    assert cfg.eval.max_samples == 10
    assert cfg.eval.prompt == "Transcribe exactly."


# ---------------------------------------------------------------------------
# load_all_datasets with total_samples parameter
# ---------------------------------------------------------------------------


@patch("aai_cli.datasets.load_dataset_samples")
@patch("aai_cli.datasets.hf_login")
def test_load_all_datasets_uses_total_samples(mock_login, mock_load, ds_entry):
    """When total_samples is passed, it should be used instead of optimization.samples."""
    mock_load.return_value = [AudioSample(audio={}, reference="hello")]
    cfg = OmegaConf.create(
        {
            "datasets": {"ds1": ds_entry(), "ds2": ds_entry()},
            "optimization": {"samples": 100},
        }
    )
    load_all_datasets(cfg, "tok", total_samples=6)
    # 6 total / 2 datasets = 3 each
    assert mock_load.call_count == 2
    assert mock_load.call_args_list[0].args[2] == 3
    assert mock_load.call_args_list[1].args[2] == 3


@patch("aai_cli.datasets.load_dataset_samples")
@patch("aai_cli.datasets.hf_login")
def test_load_all_datasets_falls_back_to_optimization_samples(mock_login, mock_load, ds_entry):
    """Without total_samples, it should fall back to optimization.samples."""
    mock_load.return_value = [AudioSample(audio={}, reference="hi")]
    cfg = OmegaConf.create(
        {
            "datasets": {"ds1": ds_entry()},
            "optimization": {"samples": 20},
        }
    )
    load_all_datasets(cfg, "tok")
    assert mock_load.call_args.args[2] == 20


# ---------------------------------------------------------------------------
# run_eval tests
# ---------------------------------------------------------------------------

FAKE_SAMPLES = [
    AudioSample(audio={"bytes": b"x"}, reference="hello world"),
    AudioSample(audio={"bytes": b"y"}, reference="foo bar"),
    AudioSample(audio={"bytes": b"z"}, reference="test sentence"),
]


@patch("aai_cli.service.transcribe")
@patch("aai_cli.service.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_shows_wer(mock_load, mock_transcribe, capsys, eval_cfg):
    """run_eval should display WER and TTFS."""
    mock_transcribe.side_effect = [
        TranscriptionResult(text="hello world", ttfb=None, ttfs=1.0),
        TranscriptionResult(text="foo baz", ttfb=None, ttfs=1.1),
        TranscriptionResult(text="test", ttfb=None, ttfs=0.9),
    ]
    run_eval(eval_cfg(), "key", "token")
    output = capsys.readouterr().out
    assert "WER:" in output
    assert "TTFS:" in output


@patch("aai_cli.service.transcribe")
@patch("aai_cli.service.load_all_datasets")
def test_run_eval_perfect_wer(mock_load, mock_transcribe, capsys, eval_cfg):
    """All-perfect transcriptions should show 0.00% WER."""
    mock_load.return_value = [AudioSample(audio={"bytes": b"x"}, reference="hello world")]
    mock_transcribe.return_value = TranscriptionResult(text="hello world", ttfb=None, ttfs=0.5)
    run_eval(eval_cfg(max_samples=1), "k", "t")
    output = capsys.readouterr().out
    assert "0.00%" in output


MULTI_DS_SAMPLES = [
    AudioSample(audio={"bytes": b"a"}, reference="hello world", dataset="ds1"),
    AudioSample(audio={"bytes": b"b"}, reference="foo bar", dataset="ds1"),
    AudioSample(audio={"bytes": b"c"}, reference="test sentence", dataset="ds2"),
    AudioSample(audio={"bytes": b"d"}, reference="another one", dataset="ds2"),
]


@patch("aai_cli.service.transcribe")
@patch("aai_cli.service.load_all_datasets", return_value=MULTI_DS_SAMPLES)
def test_run_eval_multi_dataset_wer(mock_load, mock_transcribe, capsys, eval_cfg):
    """run_eval should show per-dataset WER when multiple datasets are used."""
    mock_transcribe.side_effect = [
        TranscriptionResult(text="hello world", ttfb=None, ttfs=0.5),
        TranscriptionResult(text="foo bar", ttfb=None, ttfs=0.6),
        TranscriptionResult(text="test sentence", ttfb=None, ttfs=0.4),
        TranscriptionResult(text="wrong text", ttfb=None, ttfs=0.7),
    ]
    run_eval(eval_cfg(max_samples=4), "key", "token")
    output = capsys.readouterr().out
    assert "WER (ds1):" in output
    assert "WER (ds2):" in output
    assert "WER:" in output


@patch("aai_cli.service.compute_laser_score")
@patch("aai_cli.service.transcribe")
@patch("aai_cli.service.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_laser(mock_load, mock_transcribe, mock_laser, capsys, eval_cfg):
    """run_eval with laser=True should display LASER scores."""
    mock_transcribe.side_effect = [
        TranscriptionResult(text="hello world", ttfb=None, ttfs=1.0),
        TranscriptionResult(text="foo baz", ttfb=None, ttfs=1.1),
        TranscriptionResult(text="test", ttfb=None, ttfs=0.9),
    ]
    mock_laser.return_value = LaserResult(
        laser_score=0.95,
        word_count=2,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=0.1,
    )
    run_eval(eval_cfg(), "key", "token", laser=True)
    output = capsys.readouterr().out
    assert "LASER:" in output
    assert "WER:" in output


@patch("aai_cli.service.compute_laser_score")
@patch("aai_cli.service.transcribe")
@patch("aai_cli.service.load_all_datasets", return_value=MULTI_DS_SAMPLES)
def test_run_eval_laser_multi_dataset(mock_load, mock_transcribe, mock_laser, capsys, eval_cfg):
    """run_eval with laser=True should show per-dataset LASER when multiple datasets."""
    mock_transcribe.side_effect = [
        TranscriptionResult(text="hello world", ttfb=None, ttfs=0.5),
        TranscriptionResult(text="foo bar", ttfb=None, ttfs=0.6),
        TranscriptionResult(text="test sentence", ttfb=None, ttfs=0.4),
        TranscriptionResult(text="wrong text", ttfb=None, ttfs=0.7),
    ]
    mock_laser.return_value = LaserResult(
        laser_score=0.90,
        word_count=2,
        no_penalty_errors=[],
        major_errors=["wrong word"],
        minor_errors=[],
        total_penalty=1.0,
    )
    run_eval(eval_cfg(max_samples=4), "key", "token", laser=True)
    output = capsys.readouterr().out
    assert "LASER:" in output
    assert "WER (ds1):" in output
    assert "WER (ds2):" in output
    assert "Major:" in output


# ---------------------------------------------------------------------------
# _apply_hf_dataset_override tests
# ---------------------------------------------------------------------------


def test_apply_hf_dataset_override_replaces_datasets(ds_entry):
    """_apply_hf_dataset_override should replace cfg.datasets with a single ad-hoc entry."""
    cfg = OmegaConf.create(
        {
            "datasets": {
                "earnings22": ds_entry(
                    path="old", config="c", audio_column="a", text_column="t", split="s"
                ),
            },
            "optimization": {"samples": 50},
        }
    )
    opts = DatasetOptions(
        hf_dataset="mozilla-foundation/common_voice_11_0",
        hf_config="en",
        audio_column="audio",
        text_column="sentence",
        split="test",
    )
    result = _apply_hf_dataset_override(cfg, opts)
    assert list(result.datasets.keys()) == ["custom"]
    assert result.datasets.custom.path == "mozilla-foundation/common_voice_11_0"
    assert result.datasets.custom.config == "en"
    assert result.datasets.custom.audio_column == "audio"
    assert result.datasets.custom.text_column == "sentence"
    assert result.datasets.custom.split == "test"
    # Original config sections are preserved
    assert result.optimization.samples == 50


def test_apply_hf_dataset_override_defaults():
    """_apply_hf_dataset_override should use sensible defaults."""
    cfg = OmegaConf.create({"datasets": {}, "eval": {"max_samples": 10}})
    result = _apply_hf_dataset_override(cfg, DatasetOptions(hf_dataset="some/dataset"))
    assert result.datasets.custom.config == "default"
    assert result.datasets.custom.audio_column == "audio"
    assert result.datasets.custom.text_column == "text"
    assert result.datasets.custom.split == "test"
