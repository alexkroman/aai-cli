"""Tests for tools_scaffold.py — Gradio and Pipecat scaffolding tools."""

from aai_cli.tools_scaffold import _merge_requirements, create_gradio_asr_demo, create_voice_agent

# ---------------------------------------------------------------------------
# _merge_requirements
# ---------------------------------------------------------------------------


def test_merge_requirements_new_file(tmp_path):
    req_path = tmp_path / "requirements.txt"
    _merge_requirements(req_path, ["numpy", "requests"])
    lines = set(req_path.read_text().strip().splitlines())
    assert "numpy" in lines
    assert "requests" in lines


def test_merge_requirements_existing_file(tmp_path):
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("numpy\npandas\n")
    _merge_requirements(req_path, ["requests", "numpy"])
    lines = set(req_path.read_text().strip().splitlines())
    assert "numpy" in lines
    assert "pandas" in lines
    assert "requests" in lines


def test_merge_requirements_dedup(tmp_path):
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("assemblyai>=0.30\n")
    _merge_requirements(req_path, ["assemblyai>=0.30", "gradio>=5.0"])
    content = req_path.read_text()
    assert content.count("assemblyai>=0.30") == 1


# ---------------------------------------------------------------------------
# create_gradio_asr_demo
# ---------------------------------------------------------------------------


def test_create_gradio_demo(tools_ctx):
    result = create_gradio_asr_demo(
        title="Test App",
        description="A test",
        prompt="Transcribe.",
        output_path="app.py",
    )
    assert "Wrote" in result
    assert (tools_ctx / "app.py").is_file()
    assert (tools_ctx / "requirements.txt").is_file()
    content = (tools_ctx / "app.py").read_text()
    assert "Test App" in content


def test_create_gradio_demo_path_traversal(tools_ctx):
    result = create_gradio_asr_demo(output_path="../../hack.py")
    assert "Error" in result


# ---------------------------------------------------------------------------
# create_voice_agent
# ---------------------------------------------------------------------------


def test_create_voice_agent(tools_ctx):
    result = create_voice_agent(
        title="Voice Bot",
        description="A test bot",
        output_path="bot.py",
    )
    assert "Wrote" in result
    assert (tools_ctx / "bot.py").is_file()
    assert (tools_ctx / "requirements.txt").is_file()
    req_content = (tools_ctx / "requirements.txt").read_text()
    assert "pipecat" in req_content


def test_create_voice_agent_path_traversal(tools_ctx):
    result = create_voice_agent(output_path="../../hack.py")
    assert "Error" in result
