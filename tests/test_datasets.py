"""Tests for dataset loading and configuration helpers."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from aai_cli.datasets import (
    _filter_datasets,
    _resolve_dataset_config,
    _validate_dataset_args,
    load_dataset_samples,
)
from aai_cli.types import DatasetOptions


def _base_cfg(**extra):
    d = {
        "datasets": {
            "ds1": {
                "path": "p1",
                "config": "c1",
                "audio_column": "audio",
                "text_column": "text",
                "split": "test",
            },
            "ds2": {
                "path": "p2",
                "config": "c2",
                "audio_column": "audio",
                "text_column": "text",
                "split": "test",
            },
        },
    }
    d.update(extra)
    return OmegaConf.create(d)


_DS_CFG = OmegaConf.create(
    {"path": "p", "config": "c", "audio_column": "audio", "text_column": "text", "split": "test"}
)


# ---------------------------------------------------------------------------
# _filter_datasets
# ---------------------------------------------------------------------------


def test_filter_datasets_all():
    cfg = _base_cfg()
    result = _filter_datasets(cfg, "all")
    assert "ds1" in result.datasets
    assert "ds2" in result.datasets


def test_filter_datasets_none():
    cfg = _base_cfg()
    result = _filter_datasets(cfg, None)
    assert "ds1" in result.datasets
    assert "ds2" in result.datasets


def test_filter_datasets_specific():
    cfg = _base_cfg()
    result = _filter_datasets(cfg, "ds1")
    assert "ds1" in result.datasets
    assert "ds2" not in result.datasets


def test_filter_datasets_unknown():
    cfg = _base_cfg()
    with pytest.raises(KeyError, match="Unknown dataset"):
        _filter_datasets(cfg, "nonexistent")


# ---------------------------------------------------------------------------
# _resolve_dataset_config
# ---------------------------------------------------------------------------


def test_resolve_with_hf_dataset():
    cfg = _base_cfg()
    opts = DatasetOptions(hf_dataset="some/dataset", hf_config="en")
    result = _resolve_dataset_config(cfg, opts)
    assert "custom" in result.datasets
    assert result.datasets.custom.path == "some/dataset"


def test_resolve_without_hf_dataset():
    cfg = _base_cfg()
    opts = DatasetOptions(dataset="ds1")
    result = _resolve_dataset_config(cfg, opts)
    assert "ds1" in result.datasets
    assert "ds2" not in result.datasets


# ---------------------------------------------------------------------------
# _validate_dataset_args
# ---------------------------------------------------------------------------


def test_validate_both_raises():
    from click.exceptions import Exit

    with pytest.raises(Exit):
        _validate_dataset_args("hf/ds", "ds1")


def test_validate_only_hf():
    _validate_dataset_args("hf/ds", None)  # should not raise


def test_validate_only_dataset():
    _validate_dataset_args(None, "ds1")  # should not raise


def test_validate_neither():
    _validate_dataset_args(None, None)  # should not raise


# ---------------------------------------------------------------------------
# load_dataset_samples
# ---------------------------------------------------------------------------


@patch("aai_cli.datasets.load_dataset")
def test_load_dataset_samples_basic(mock_load, mock_hf_dataset):
    mock_load.return_value = mock_hf_dataset(
        [
            {"audio": {"bytes": b"a"}, "text": "hello"},
            {"audio": {"bytes": b"b"}, "text": "world"},
            {"audio": {"bytes": b"c"}, "text": "third"},
        ]
    )
    samples = load_dataset_samples("test_ds", _DS_CFG, 2, "token")
    assert len(samples) == 2
    assert samples[0].reference == "hello"
    assert samples[1].reference == "world"


@patch("aai_cli.datasets.load_dataset")
def test_load_dataset_samples_skips_ignore(mock_load, mock_hf_dataset):
    mock_load.return_value = mock_hf_dataset(
        [
            {"audio": {"bytes": b"a"}, "text": "ignore_time_segment_in_scoring"},
            {"audio": {"bytes": b"b"}, "text": "valid"},
        ]
    )
    samples = load_dataset_samples("test_ds", _DS_CFG, 10, "token")
    assert len(samples) == 1
    assert samples[0].reference == "valid"


@patch("aai_cli.datasets.load_dataset")
def test_load_dataset_samples_skips_inaudible(mock_load, mock_hf_dataset):
    mock_load.return_value = mock_hf_dataset(
        [
            {"audio": {"bytes": b"a"}, "text": "word [inaudible] word"},
            {"audio": {"bytes": b"b"}, "text": "clean text"},
        ]
    )
    samples = load_dataset_samples("test_ds", _DS_CFG, 10, "token")
    assert len(samples) == 1
    assert samples[0].reference == "clean text"
