"""Tests for tools_eval.py — eval_prompt and optimize_prompt agent tools."""

from unittest.mock import patch

from aai_cli.tools_eval import eval_prompt, optimize_prompt

# ---------------------------------------------------------------------------
# optimize_prompt validation
# ---------------------------------------------------------------------------


def test_optimize_prompt_too_few_samples():
    result = optimize_prompt("Transcribe.", samples=3)
    assert "Error" in result
    assert "at least 5" in result


def test_optimize_prompt_zero_iterations():
    result = optimize_prompt("Transcribe.", iterations=0)
    assert "Error" in result
    assert "at least 1" in result


def test_optimize_prompt_empty_prompt():
    result = optimize_prompt("   ")
    assert "Error" in result
    assert "empty" in result


# ---------------------------------------------------------------------------
# eval_prompt wiring
# ---------------------------------------------------------------------------


@patch("aai_cli.service.run_eval")
@patch("aai_cli.service.setup_config")
def test_eval_prompt_calls_run_eval(mock_setup, mock_run_eval):
    """eval_prompt should wire through to setup_config + run_eval."""
    from omegaconf import OmegaConf

    mock_setup.return_value = OmegaConf.create({"eval": {"prompt": "test"}})
    mock_run_eval.return_value = None

    result = eval_prompt("test prompt", max_samples=5)
    mock_setup.assert_called_once()
    mock_run_eval.assert_called_once()
    # With no output captured, should return "(no output)"
    assert result == "(no output)"


# ---------------------------------------------------------------------------
# optimize_prompt wiring
# ---------------------------------------------------------------------------


@patch("aai_cli.optimizer.run_optimization")
@patch("aai_cli.datasets.load_all_datasets")
@patch("aai_cli.service.setup_config")
def test_optimize_prompt_calls_run_optimization(mock_setup, mock_load, mock_run):
    """optimize_prompt should wire through to setup_config + load_all_datasets + run_optimization."""
    from omegaconf import OmegaConf

    from aai_cli.types import OptimizationResult

    mock_setup.return_value = OmegaConf.create(
        {
            "optimization": {
                "starting_prompt": "Transcribe.",
                "iterations": 1,
                "llm_model": "claude-sonnet-4-6",
                "num_threads": 1,
            }
        }
    )
    mock_load.return_value = []
    mock_run.return_value = OptimizationResult(
        best_prompt="optimized",
        best_score=0.95,
        metric="LASER",
        starting_prompt="Transcribe.",
        model="assemblyai/universal-3-pro",
        total_eval_samples=5,
        optimizer="DSPy-GEPA",
        num_trials=5,
        llm_model="claude-sonnet-4-6",
        timestamp="2024-01-01",
    )

    result = optimize_prompt("Transcribe.", samples=5, iterations=1)
    mock_setup.assert_called_once()
    mock_load.assert_called_once()
    mock_run.assert_called_once()
    assert "optimized" in result
    assert "0.95" in result
