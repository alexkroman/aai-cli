"""Shared fixtures for aai-cli tests."""

import io
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from rich.console import Console

from aai_cli.tools import _ctx


@pytest.fixture()
def tools_ctx(tmp_path):
    """Temporarily set _ctx.cwd to tmp_path. Restores original on teardown."""
    original_cwd = _ctx.cwd
    original_console = _ctx.console
    _ctx.cwd = str(tmp_path)
    yield tmp_path
    _ctx.cwd = original_cwd
    _ctx.console = original_console


@pytest.fixture()
def rich_console():
    """Return (Console, StringIO) pair for capturing rich output."""
    buf = io.StringIO()
    return Console(file=buf, no_color=True, width=200), buf


@pytest.fixture()
def eval_cfg():
    """Factory for minimal eval config dicts."""

    def _create(
        max_samples=3,
        prompt="Transcribe.",
        num_threads=1,
        speech_model="universal-3-pro",
    ):
        return OmegaConf.create(
            {
                "eval": {
                    "max_samples": max_samples,
                    "prompt": prompt,
                    "num_threads": num_threads,
                    "speech_model": speech_model,
                },
                "datasets": {},
            }
        )

    return _create


@pytest.fixture()
def ds_entry():
    """Factory for dataset config entries."""

    def _create(path="p", config="c", audio_column="a", text_column="t", split="test"):
        return {
            "path": path,
            "config": config,
            "audio_column": audio_column,
            "text_column": text_column,
            "split": split,
        }

    return _create


@pytest.fixture()
def mock_hf_dataset():
    """Factory that creates a mock HF dataset with configurable rows."""

    def _create(items):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(items))
        mock_ds.cast_column.return_value = mock_ds
        return mock_ds

    return _create
