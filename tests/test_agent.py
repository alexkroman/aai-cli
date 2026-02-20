"""Tests for agent.py — callbacks, helpers, and agent creation."""

import io
from unittest.mock import MagicMock, patch

from rich.console import Console
from smolagents import ActionStep
from smolagents.memory import PlanningStep

from aai_cli.agent import (
    _check_answer,
    _configure_litellm,
    _make_plan_log,
    _make_step_log,
    _prune_old_steps,
)

# ---------------------------------------------------------------------------
# _check_answer
# ---------------------------------------------------------------------------


def test_check_answer_valid():
    assert _check_answer("This is an answer") is True


def test_check_answer_empty():
    assert _check_answer("") is False


def test_check_answer_none():
    assert _check_answer(None) is False


def test_check_answer_whitespace():
    assert _check_answer("   ") is False


# ---------------------------------------------------------------------------
# _make_plan_log
# ---------------------------------------------------------------------------


def test_make_plan_log_prints_plan(rich_console):
    con, buf = rich_console
    log = _make_plan_log(con)
    step = MagicMock(spec=PlanningStep)
    step.plan = "Step 1: Do something\nStep 2: Do another thing"
    log(step)
    output = buf.getvalue()
    assert "Step 1" in output
    assert "Step 2" in output


def test_make_plan_log_skips_empty(rich_console):
    con, buf = rich_console
    log = _make_plan_log(con)
    step = MagicMock(spec=PlanningStep)
    step.plan = ""
    log(step)
    output = buf.getvalue()
    assert output.strip() == ""


def test_make_plan_log_no_plan_attr(rich_console):
    con, buf = rich_console
    log = _make_plan_log(con)
    step = MagicMock(spec=[])  # no attributes
    log(step)
    output = buf.getvalue()
    assert output.strip() == ""


# ---------------------------------------------------------------------------
# _make_step_log
# ---------------------------------------------------------------------------


def test_make_step_log_prints_observations(rich_console):
    con, buf = rich_console
    log = _make_step_log(con)
    step = MagicMock(spec=ActionStep)
    step.step_number = 1
    step.observations = "Output line 1\nOutput line 2"
    log(step)
    output = buf.getvalue()
    assert "Output line 1" in output


def test_make_step_log_truncates(rich_console):
    con, buf = rich_console
    log = _make_step_log(con)
    step = MagicMock(spec=ActionStep)
    step.step_number = 1
    step.observations = "\n".join(f"Line {i}" for i in range(10))
    log(step)
    output = buf.getvalue()
    assert "+7 lines" in output


def test_make_step_log_no_step_number(rich_console):
    con, buf = rich_console
    log = _make_step_log(con)
    step = MagicMock(spec=[])  # no step_number
    log(step)
    assert buf.getvalue().strip() == ""


# ---------------------------------------------------------------------------
# _prune_old_steps
# ---------------------------------------------------------------------------


def test_prune_old_steps_trims_observations():
    agent = MagicMock()
    old_step = MagicMock(spec=ActionStep)
    old_step.step_number = 1
    old_step.observations = "x" * 500
    old_step.observations_images = ["img"]

    current = MagicMock(spec=ActionStep)
    current.step_number = 10

    agent.memory.steps = [old_step]
    _prune_old_steps(current, agent)

    assert len(old_step.observations) < 500
    assert "[pruned]" in old_step.observations
    assert old_step.observations_images is None


def test_prune_old_steps_keeps_recent():
    agent = MagicMock()
    recent_step = MagicMock(spec=ActionStep)
    recent_step.step_number = 8
    recent_step.observations = "x" * 500

    current = MagicMock(spec=ActionStep)
    current.step_number = 10

    agent.memory.steps = [recent_step]
    _prune_old_steps(current, agent)

    # Step 8 is within 5 of step 10, so it shouldn't be pruned
    assert len(recent_step.observations) == 500


def test_prune_old_steps_no_agent():
    step = MagicMock(spec=ActionStep)
    step.step_number = 1
    # Should not raise
    _prune_old_steps(step, None)


# ---------------------------------------------------------------------------
# _configure_litellm
# ---------------------------------------------------------------------------


def test_configure_litellm():
    # Clear the lru_cache so we can test
    _configure_litellm.cache_clear()
    with (
        patch("litellm.suppress_debug_info", False),
        patch("litellm.drop_params", False),
        patch("litellm.modify_params", False),
    ):
        _configure_litellm()
        import litellm

        assert litellm.num_retries == 2
        assert litellm.request_timeout == 120
    _configure_litellm.cache_clear()


# ---------------------------------------------------------------------------
# create_agent
# ---------------------------------------------------------------------------


@patch("aai_cli.agent.LiteLLMModel")
@patch("aai_cli.agent.CodeAgent")
def test_create_agent_returns_agent_and_tools(mock_code_agent, _mock_model, tmp_path):
    _configure_litellm.cache_clear()

    prompts = tmp_path / "prompts.yaml"
    prompts.write_text("system_prompt: test\n")
    sys_prompt = tmp_path / "system_prompt.txt"
    sys_prompt.write_text("You are a helpful agent working in {cwd}.")

    con = Console(file=io.StringIO())
    from aai_cli.agent import create_agent

    mock_code_agent.return_value = MagicMock()

    _agent, tool_names = create_agent(
        "fake-key", con, "/tmp", prompts_path=prompts, system_prompt_path=sys_prompt
    )
    assert mock_code_agent.called
    assert "final_answer" in tool_names
    assert len(tool_names) > 5
