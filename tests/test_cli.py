"""Tests for CLI entry point."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aai_cli.cli import (
    _apply_hf_dataset_override,
    get_env_key,
    load_all_datasets,
    print_banner,
    run_eval,
)

# Absolute path to the config directory
CONFIG_DIR = str((Path(__file__).parent / ".." / "aai_cli" / "conf").resolve())


def test_banner_renders(capsys):
    """Verify banner prints without error."""
    print_banner()
    captured = capsys.readouterr()
    assert "Welcome to the AssemblyAI agent" in captured.out


def test_get_env_key_returns_value(monkeypatch):
    """Env var present — should return its value."""
    monkeypatch.setenv("TEST_KEY_ABC", "secret123")
    assert get_env_key("TEST_KEY_ABC") == "secret123"


def test_get_env_key_exits_when_missing(monkeypatch):
    """Env var absent — should sys.exit(1)."""
    monkeypatch.delenv("TEST_KEY_MISSING", raising=False)
    with pytest.raises(SystemExit, match="1"):
        get_env_key("TEST_KEY_MISSING")


def test_default_config_loads():
    """Hydra compose should load config.yaml and have the expected structure."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
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
    """Hydra override should change optimization.samples."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["optimization.samples=50"])
        assert cfg.optimization.samples == 50


# ---------------------------------------------------------------------------
# Eval config tests
# ---------------------------------------------------------------------------


def test_eval_config_defaults():
    """Eval section should have expected defaults."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
        assert cfg.eval.max_samples == 50
        assert cfg.eval.prompt == "Transcribe verbatim."
        assert cfg.eval.num_threads == 12


def test_eval_config_override():
    """Hydra overrides should change eval settings."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "eval.max_samples=10",
                "eval.prompt=Transcribe exactly.",
            ],
        )
        assert cfg.eval.max_samples == 10
        assert cfg.eval.prompt == "Transcribe exactly."


# ---------------------------------------------------------------------------
# load_all_datasets with total_samples parameter
# ---------------------------------------------------------------------------


@patch("aai_cli.cli.load_dataset_samples")
def test_load_all_datasets_uses_total_samples(mock_load):
    """When total_samples is passed, it should be used instead of optimization.samples."""
    mock_load.return_value = [{"audio": {}, "reference": "hello"}]
    cfg = OmegaConf.create(
        {
            "datasets": {
                "ds1": {
                    "path": "p",
                    "config": "c",
                    "audio_column": "a",
                    "text_column": "t",
                    "split": "test",
                },
                "ds2": {
                    "path": "p",
                    "config": "c",
                    "audio_column": "a",
                    "text_column": "t",
                    "split": "test",
                },
            },
            "optimization": {"samples": 100},
        }
    )
    load_all_datasets(cfg, "tok", total_samples=6)
    # 6 total / 2 datasets = 3 each
    assert mock_load.call_count == 2
    assert mock_load.call_args_list[0].args[2] == 3
    assert mock_load.call_args_list[1].args[2] == 3


@patch("aai_cli.cli.load_dataset_samples")
def test_load_all_datasets_falls_back_to_optimization_samples(mock_load):
    """Without total_samples, it should fall back to optimization.samples."""
    mock_load.return_value = [{"audio": {}, "reference": "hi"}]
    cfg = OmegaConf.create(
        {
            "datasets": {
                "ds1": {
                    "path": "p",
                    "config": "c",
                    "audio_column": "a",
                    "text_column": "t",
                    "split": "test",
                },
            },
            "optimization": {"samples": 20},
        }
    )
    load_all_datasets(cfg, "tok")
    assert mock_load.call_args.args[2] == 20


# ---------------------------------------------------------------------------
# run_eval tests
# ---------------------------------------------------------------------------

FAKE_SAMPLES = [
    {"audio": {"bytes": b"x"}, "reference": "hello world"},
    {"audio": {"bytes": b"y"}, "reference": "foo bar"},
    {"audio": {"bytes": b"z"}, "reference": "test sentence"},
]


@patch("aai_cli.cli.transcribe_assemblyai")
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_shows_wer(mock_load, mock_transcribe, capsys):
    """run_eval should display WER and TTFS."""
    mock_transcribe.side_effect = [
        {"text": "hello world", "ttfb": None, "ttfs": 1.0},
        {"text": "foo baz", "ttfb": None, "ttfs": 1.1},
        {"text": "test", "ttfb": None, "ttfs": 0.9},
    ]
    cfg = OmegaConf.create(
        {
            "eval": {
                "max_samples": 3,
                "prompt": "Transcribe.",
                "num_threads": 1,
                "speech_model": "universal-3-pro",
            },
            "datasets": {},
        }
    )
    run_eval(cfg, "key", "token")
    output = capsys.readouterr().out
    assert "WER:" in output
    assert "TTFS:" in output


@patch("aai_cli.cli.transcribe_assemblyai")
@patch("aai_cli.cli.load_all_datasets")
def test_run_eval_perfect_wer(mock_load, mock_transcribe, capsys):
    """All-perfect transcriptions should show 0.00% WER."""
    mock_load.return_value = [{"audio": {"bytes": b"x"}, "reference": "hello world"}]
    mock_transcribe.return_value = {"text": "hello world", "ttfb": None, "ttfs": 0.5}
    cfg = OmegaConf.create(
        {
            "eval": {
                "max_samples": 1,
                "prompt": "Transcribe.",
                "num_threads": 1,
                "speech_model": "universal-3-pro",
            },
            "datasets": {},
        }
    )
    run_eval(cfg, "k", "t")
    output = capsys.readouterr().out
    assert "0.00%" in output


MULTI_DS_SAMPLES = [
    {"audio": {"bytes": b"a"}, "reference": "hello world", "dataset": "ds1"},
    {"audio": {"bytes": b"b"}, "reference": "foo bar", "dataset": "ds1"},
    {"audio": {"bytes": b"c"}, "reference": "test sentence", "dataset": "ds2"},
    {"audio": {"bytes": b"d"}, "reference": "another one", "dataset": "ds2"},
]


@patch("aai_cli.cli.transcribe_assemblyai")
@patch("aai_cli.cli.load_all_datasets", return_value=MULTI_DS_SAMPLES)
def test_run_eval_multi_dataset_wer(mock_load, mock_transcribe, capsys):
    """run_eval should show per-dataset WER when multiple datasets are used."""
    mock_transcribe.side_effect = [
        {"text": "hello world", "ttfb": None, "ttfs": 0.5},
        {"text": "foo bar", "ttfb": None, "ttfs": 0.6},
        {"text": "test sentence", "ttfb": None, "ttfs": 0.4},
        {"text": "wrong text", "ttfb": None, "ttfs": 0.7},
    ]
    cfg = OmegaConf.create(
        {
            "eval": {
                "max_samples": 4,
                "prompt": "Transcribe.",
                "num_threads": 1,
                "speech_model": "universal-3-pro",
            },
            "datasets": {},
        }
    )
    run_eval(cfg, "key", "token")
    output = capsys.readouterr().out
    assert "WER (ds1):" in output
    assert "WER (ds2):" in output
    assert "WER:" in output


# ---------------------------------------------------------------------------
# _apply_hf_dataset_override tests
# ---------------------------------------------------------------------------


def test_apply_hf_dataset_override_replaces_datasets():
    """_apply_hf_dataset_override should replace cfg.datasets with a single ad-hoc entry."""
    cfg = OmegaConf.create(
        {
            "datasets": {
                "earnings22": {
                    "path": "old",
                    "config": "c",
                    "audio_column": "a",
                    "text_column": "t",
                    "split": "s",
                },
            },
            "optimization": {"samples": 50},
        }
    )
    result = _apply_hf_dataset_override(
        cfg,
        hf_dataset="mozilla-foundation/common_voice_11_0",
        hf_config="en",
        audio_column="audio",
        text_column="sentence",
        split="test",
    )
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
    result = _apply_hf_dataset_override(cfg, hf_dataset="some/dataset")
    assert result.datasets.custom.config == "default"
    assert result.datasets.custom.audio_column == "audio"
    assert result.datasets.custom.text_column == "text"
    assert result.datasets.custom.split == "test"
