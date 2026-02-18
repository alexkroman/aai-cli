"""Tests for CLI entry point."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from aai_cli.cli import get_env_key, print_banner

# Absolute path to the config directory
CONFIG_DIR = str((Path(__file__).parent / ".." / "aai_cli" / "conf").resolve())


def test_banner_renders(capsys):
    """Verify pyfiglet banner prints without error."""
    print_banner()
    captured = capsys.readouterr()
    assert "Prompt Optimizer" in captured.out


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
        assert cfg.optimization.num_threads == 24


def test_config_override():
    """Hydra override should change optimization.samples."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["optimization.samples=50"])
        assert cfg.optimization.samples == 50
