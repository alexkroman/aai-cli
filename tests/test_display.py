"""Tests for display helpers."""

from aai_cli.display import _latency_stats, _print_eval_summary, _print_sample_result
from aai_cli.types import EvalSampleResult, LaserResult

# ---------------------------------------------------------------------------
# _latency_stats
# ---------------------------------------------------------------------------


def test_latency_stats_single_value():
    result = _latency_stats([1.0])
    assert "avg 1.00s" in result
    assert "P95 1.00s" in result


def test_latency_stats_multiple_values():
    result = _latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "avg 3.00s" in result
    assert "Min 1.00s" in result
    assert "Max 5.00s" in result


# ---------------------------------------------------------------------------
# _print_sample_result
# ---------------------------------------------------------------------------


def test_print_sample_result_basic(rich_console):
    con, buf = rich_console
    r = EvalSampleResult(text="hello world", wer=0.0, reference="hello world", ttfs=1.5)
    _print_sample_result(r, 1, 5, laser=False, console=con)
    output = buf.getvalue()
    assert "WER: 0.0%" in output
    assert "REF:" in output
    assert "HYP:" in output
    assert "TTFS 1.50s" in output


def test_print_sample_result_with_laser(rich_console):
    con, buf = rich_console
    laser = LaserResult(
        laser_score=0.9,
        word_count=5,
        no_penalty_errors=["ok"],
        major_errors=["bad"],
        minor_errors=["meh"],
        total_penalty=1.5,
    )
    r = EvalSampleResult(text="hyp", wer=0.1, reference="ref", laser=laser)
    _print_sample_result(r, 2, 10, laser=True, console=con)
    output = buf.getvalue()
    assert "LASER:" in output
    assert "Major:" in output
    assert "Minor:" in output
    assert "No penalty:" in output


def test_print_sample_result_with_dataset_tag(rich_console):
    con, buf = rich_console
    r = EvalSampleResult(text="hyp", wer=0.5, reference="ref", dataset="earnings22")
    _print_sample_result(r, 1, 1, laser=False, console=con)
    output = buf.getvalue()
    assert "(earnings22)" in output


def test_print_sample_result_ttfb_and_ttfs(rich_console):
    con, buf = rich_console
    r = EvalSampleResult(text="h", wer=0.0, reference="h", ttfb=0.25, ttfs=1.0)
    _print_sample_result(r, 1, 1, laser=False, console=con)
    output = buf.getvalue()
    assert "TTFB 0.25s" in output
    assert "TTFS 1.00s" in output


# ---------------------------------------------------------------------------
# _print_eval_summary
# ---------------------------------------------------------------------------


def test_print_eval_summary_basic(rich_console):
    con, buf = rich_console
    results = [
        EvalSampleResult(text="hello world", wer=0.0, reference="hello world", ttfs=1.0),
        EvalSampleResult(text="foo baz", wer=0.5, reference="foo bar", ttfs=2.0),
    ]
    _print_eval_summary(results, 2, "universal-3-pro", "batch", laser=False, console=con)
    output = buf.getvalue()
    assert "WER:" in output
    assert "TTFS:" in output
    assert "Samples: 2" in output


def test_print_eval_summary_multi_dataset(rich_console):
    con, buf = rich_console
    results = [
        EvalSampleResult(text="a", wer=0.0, reference="a", dataset="ds1"),
        EvalSampleResult(text="b", wer=0.5, reference="c", dataset="ds2"),
    ]
    _print_eval_summary(results, 2, "u3-pro", "streaming", laser=False, console=con)
    output = buf.getvalue()
    assert "WER (ds1):" in output
    assert "WER (ds2):" in output


def test_print_eval_summary_laser(rich_console):
    con, buf = rich_console
    laser = LaserResult(
        laser_score=0.9,
        word_count=5,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=0.5,
    )
    results = [EvalSampleResult(text="h", wer=0.1, reference="r", laser=laser)]
    _print_eval_summary(results, 1, "universal-3-pro", "batch", laser=True, console=con)
    output = buf.getvalue()
    assert "LASER:" in output
