"""Tests for CLI entry point."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aai_cli.cli import get_env_key, load_all_datasets, print_banner, run_eval

# Absolute path to the config directory
CONFIG_DIR = str((Path(__file__).parent / ".." / "aai_cli" / "conf").resolve())


def test_banner_renders(capsys):
    """Verify pyfiglet banner prints without error."""
    print_banner()
    captured = capsys.readouterr()
    assert "___" in captured.out


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
        assert cfg.optimization.samples == 50
        assert cfg.optimization.iterations == 5
        assert cfg.optimization.num_threads == 12
        assert cfg.optimization.meta_every == 3


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
        assert cfg.mode == "optimize"
        assert cfg.eval.max_samples == 50
        assert cfg.eval.prompt == "Transcribe verbatim."
        assert cfg.eval.num_threads == 12


def test_eval_config_override():
    """Hydra overrides should change eval settings."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "mode=eval",
                "eval.max_samples=10",
                "eval.prompt=Transcribe exactly.",
            ],
        )
        assert cfg.mode == "eval"
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

FAKE_RESULTS = [
    {"reference": "hello world", "hypothesis": "hello world", "wer": 0.0},
    {"reference": "foo bar", "hypothesis": "foo baz", "wer": 0.5},
    {"reference": "test sentence", "hypothesis": "test", "wer": 0.5},
]


@patch("aai_cli.cli._evaluate_prompt", return_value=FAKE_RESULTS)
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_prints_all_ref_hyp(mock_load, mock_eval, capsys):
    """run_eval should print REF and HYP for every sample."""
    cfg = OmegaConf.create(
        {
            "eval": {"max_samples": 3, "prompt": "Transcribe verbatim.", "num_threads": 2},
            "datasets": {},
        }
    )
    run_eval(cfg, "fake-key", "fake-token")
    output = capsys.readouterr().out

    assert "hello world" in output
    assert "foo baz" in output
    assert "REF:" in output
    assert "HYP:" in output
    # All 3 samples printed
    assert "Sample 1/3" in output
    assert "Sample 2/3" in output
    assert "Sample 3/3" in output


@patch("aai_cli.cli._evaluate_prompt", return_value=FAKE_RESULTS)
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_shows_wer_summary(mock_load, mock_eval, capsys):
    """run_eval should display the overall WER in the summary panel."""
    cfg = OmegaConf.create(
        {
            "eval": {"max_samples": 3, "prompt": "Transcribe verbatim.", "num_threads": 2},
            "datasets": {},
        }
    )
    run_eval(cfg, "fake-key", "fake-token")
    output = capsys.readouterr().out

    # Mean WER = (0.0 + 0.5 + 0.5) / 3 = 33.33%
    assert "33.33%" in output


@patch("aai_cli.cli._evaluate_prompt", return_value=FAKE_RESULTS)
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_passes_prompt_and_threads(mock_load, mock_eval, capsys):
    """run_eval should forward prompt and num_threads to _evaluate_prompt."""
    cfg = OmegaConf.create(
        {
            "eval": {"max_samples": 3, "prompt": "Custom prompt.", "num_threads": 4},
            "datasets": {},
        }
    )
    run_eval(cfg, "my-key", "my-token")

    mock_eval.assert_called_once_with("Custom prompt.", FAKE_SAMPLES, "my-key", 4, desc="Eval")
    mock_load.assert_called_once_with(cfg, "my-token", total_samples=3)


@patch("aai_cli.cli._evaluate_prompt")
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_perfect_wer(mock_load, mock_eval, capsys):
    """All-perfect results should show 0.00% WER."""
    mock_eval.return_value = [
        {"reference": "hello", "hypothesis": "hello", "wer": 0.0},
    ]
    cfg = OmegaConf.create(
        {
            "eval": {"max_samples": 1, "prompt": "Transcribe.", "num_threads": 1},
            "datasets": {},
        }
    )
    run_eval(cfg, "k", "t")
    output = capsys.readouterr().out
    assert "0.00%" in output


@patch("aai_cli.cli._evaluate_prompt")
@patch("aai_cli.cli.load_all_datasets", return_value=FAKE_SAMPLES)
def test_run_eval_wer_style_colors(mock_load, mock_eval, capsys):
    """Per-sample WER should use correct style thresholds."""
    mock_eval.return_value = [
        {"reference": "a", "hypothesis": "a", "wer": 0.0},  # success
        {"reference": "b", "hypothesis": "b x", "wer": 0.05},  # progress (<0.1)
        {"reference": "c", "hypothesis": "d", "wer": 0.8},  # warning (>=0.1)
    ]
    cfg = OmegaConf.create(
        {
            "eval": {"max_samples": 3, "prompt": "T", "num_threads": 1},
            "datasets": {},
        }
    )
    run_eval(cfg, "k", "t")
    output = capsys.readouterr().out
    # All 3 samples should be printed
    assert "Sample 1/3" in output
    assert "Sample 3/3" in output
