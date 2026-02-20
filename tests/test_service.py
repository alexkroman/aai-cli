"""Tests for service.py — config, overrides, and eval orchestration."""

import logging

import pytest

from aai_cli.service import _build_overrides, _suppress_loggers, get_env_key, setup_config
from aai_cli.types import DatasetOptions

# ---------------------------------------------------------------------------
# _build_overrides
# ---------------------------------------------------------------------------


def test_build_overrides_basic():
    result = _build_overrides("eval", prompt="hello", max_samples=10)
    assert result == {"eval": {"prompt": "hello", "max_samples": 10}}


def test_build_overrides_filters_none():
    result = _build_overrides("eval", prompt="hello", max_samples=None)
    assert result == {"eval": {"prompt": "hello"}}


def test_build_overrides_empty():
    result = _build_overrides("eval", a=None, b=None)
    assert result == {}


# ---------------------------------------------------------------------------
# _suppress_loggers
# ---------------------------------------------------------------------------


def test_suppress_loggers():
    _suppress_loggers()
    for name in ("datasets.info", "huggingface_hub", "httpx"):
        assert logging.getLogger(name).level == logging.CRITICAL


# ---------------------------------------------------------------------------
# setup_config
# ---------------------------------------------------------------------------


def test_setup_config_basic():
    cfg = setup_config("eval", prompt="test prompt", max_samples=5)
    assert cfg.eval.prompt == "test prompt"
    assert cfg.eval.max_samples == 5


def test_setup_config_with_dataset_filter():
    cfg = setup_config("eval", ds_opts=DatasetOptions(dataset="earnings22"))
    assert "earnings22" in cfg.datasets
    assert len(list(cfg.datasets.keys())) == 1


def test_setup_config_with_hf_override():
    opts = DatasetOptions(hf_dataset="some/dataset", hf_config="en")
    cfg = setup_config("eval", ds_opts=opts)
    assert "custom" in cfg.datasets
    assert cfg.datasets.custom.path == "some/dataset"


# ---------------------------------------------------------------------------
# get_env_key
# ---------------------------------------------------------------------------


def test_get_env_key_missing(monkeypatch):
    monkeypatch.delenv("__TEST_MISSING__", raising=False)
    with pytest.raises(ValueError, match="Missing env var"):
        get_env_key("__TEST_MISSING__")


def test_get_env_key_present(monkeypatch):
    monkeypatch.setenv("__TEST_PRESENT__", "val")
    assert get_env_key("__TEST_PRESENT__") == "val"
