"""Tool functions exposed to the coding agent."""

import os
import shlex
import subprocess
from pathlib import Path

from rich.console import Console
from smolagents import tool

# Module-level state shared by tools — call init_tools() before use.
_console: Console | None = None
_cwd: str = "."


def init_tools(console: Console, cwd: str) -> None:
    """Set the shared console and working directory used by tool functions."""
    global _console, _cwd
    _console = console
    _cwd = cwd


def _run_aai_cmd(cmd: str, timeout: int = 600) -> str:
    """Run an `aai` CLI command, stream output live, and return it."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=_cwd,
        env=env,
    )
    lines: list[str] = []
    try:
        # Use readline() instead of iterator to avoid read-ahead buffering deadlocks
        # with tqdm \r output filling the pipe buffer.
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            if line:
                line = line.rstrip("\n")
                lines.append(line)
                if _console:
                    _console.print(f"    {line}", style="dim")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        lines.append(f"[timed out after {timeout}s]")

    output = "\n".join(lines)
    if proc.returncode and proc.returncode != 0:
        output += f"\n[exit code: {proc.returncode}]"
    return output or "(no output)"


def _hf_dataset_args(
    hf_dataset: str, hf_config: str, audio_column: str, text_column: str, split: str, dataset: str
) -> list[str]:
    """Build CLI flags for HF dataset overrides shared by eval/optimize tools."""
    if hf_dataset:
        return [
            f"--hf-dataset {shlex.quote(hf_dataset)}",
            f"--hf-config {shlex.quote(hf_config)}",
            f"--audio-column {shlex.quote(audio_column)}",
            f"--text-column {shlex.quote(text_column)}",
            f"--split {shlex.quote(split)}",
        ]
    if dataset != "all":
        return [f"--dataset {dataset}"]
    return []


@tool
def eval_prompt(
    prompt: str,
    max_samples: int = 10,
    num_threads: int = 12,
    dataset: str = "all",
    hf_dataset: str = "",
    hf_config: str = "default",
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "test",
) -> str:
    """Evaluate an AssemblyAI transcription prompt and measure Word Error Rate (WER).

    Sends audio samples to AssemblyAI for transcription using the given prompt,
    then compares the output against ground-truth references. Reports per-sample
    results and overall WER. Lower WER = better transcription quality.

    Use this when a user asks to evaluate, test, or benchmark AssemblyAI transcription,
    measure WER, or compare prompt performance.

    Args:
        prompt: The transcription prompt to evaluate (e.g. "Transcribe verbatim.").
        max_samples: Number of audio samples to transcribe (default: 10 for quick test, use 50+ for reliable results).
        num_threads: Parallel transcription threads (default: 12).
        dataset: Preconfigured dataset shortcut. One of "earnings22", "peoples", "ami", or "all" (default). Ignored when hf_dataset is set.
        hf_dataset: Any HF audio dataset path (e.g. "mozilla-foundation/common_voice_11_0"). Overrides dataset.
        hf_config: HF dataset config/subset name (default: "default").
        audio_column: Name of the audio column in the dataset (default: "audio").
        text_column: Name of the text/reference column in the dataset (default: "text").
        split: Dataset split to use (default: "test").
    """
    parts = [
        "aai eval",
        f"--prompt {shlex.quote(prompt)}",
        f"--max-samples {max_samples}",
        f"--num-threads {num_threads}",
        *_hf_dataset_args(hf_dataset, hf_config, audio_column, text_column, split, dataset),
    ]

    return _run_aai_cmd(" ".join(parts), timeout=600)


@tool
def optimize_prompt(
    starting_prompt: str,
    iterations: int = 5,
    samples: int = 50,
    candidates_per_step: int = 1,
    num_threads: int = 12,
    llm_model: str = "",
    dataset: str = "all",
    hf_dataset: str = "",
    hf_config: str = "default",
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "test",
) -> str:
    """Optimize an AssemblyAI transcription prompt using OPRO (LLM-guided optimization).

    Iteratively proposes improved transcription prompts by analyzing transcription
    errors and using an LLM to suggest fixes. Each iteration: generates candidate
    prompts, evaluates them against audio datasets, and keeps the best.

    This is long-running — use fewer iterations/samples for quick experiments.

    Use this when a user asks to optimize, improve, or tune a transcription prompt.

    Args:
        starting_prompt: The seed prompt to begin optimizing from.
        iterations: Number of optimization rounds (default: 5).
        samples: Total audio samples to evaluate per candidate (default: 50, minimum: 5, 50+ recommended).
        candidates_per_step: Number of candidate prompts generated per iteration (default: 1).
        num_threads: Parallel transcription threads (default: 12).
        llm_model: Model used to generate candidate prompts. Leave empty to use config default.
        dataset: Preconfigured dataset shortcut. One of "earnings22", "peoples", "ami", or "all" (default). Ignored when hf_dataset is set.
        hf_dataset: Any HF audio dataset path (e.g. "mozilla-foundation/common_voice_11_0"). Overrides dataset.
        hf_config: HF dataset config/subset name (default: "default").
        audio_column: Name of the audio column in the dataset (default: "audio").
        text_column: Name of the text/reference column in the dataset (default: "text").
        split: Dataset split to use (default: "test").
    """
    # --- Validation ---
    if samples < 5:
        return "Error: samples must be at least 5 (50+ recommended for reliable results)."
    if iterations < 1:
        return "Error: iterations must be at least 1."
    if candidates_per_step < 1:
        return "Error: candidates_per_step must be at least 1."
    if not starting_prompt.strip():
        return "Error: starting_prompt cannot be empty."

    parts = [
        "aai optimize",
        f"--starting-prompt {shlex.quote(starting_prompt)}",
        f"--iterations {iterations}",
        f"--samples {samples}",
        f"--candidates-per-step {candidates_per_step}",
        f"--num-threads {num_threads}",
        *([f"--llm-model {shlex.quote(llm_model)}"] if llm_model else []),
        *_hf_dataset_args(hf_dataset, hf_config, audio_column, text_column, split, dataset),
    ]

    return _run_aai_cmd(" ".join(parts), timeout=3600)


@tool
def search_audio_datasets(
    query: str,
    task: str = "automatic-speech-recognition",
    language: str = "",
    limit: int = 10,
) -> str:
    """Search Hugging Face Hub for audio datasets relevant to speech recognition.

    Use this to discover datasets for evaluating or optimizing transcription prompts.
    Results are sorted by downloads (most popular first) and include dataset ID,
    download count, description, languages, and available configs.

    Args:
        query: Free-text search term (e.g. "earnings", "medical", "french").
        task: Task category filter (default: "automatic-speech-recognition").
        language: Optional language filter (e.g. "en", "fr"). Leave empty for all languages.
        limit: Maximum number of results to return (default: 10).
    """
    from huggingface_hub import HfApi

    api = HfApi()

    kwargs: dict = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    }
    if task:
        kwargs["task_categories"] = task
    if language:
        kwargs["language"] = language

    datasets = list(api.list_datasets(**kwargs))

    if not datasets:
        return f"No datasets found for query='{query}', task='{task}', language='{language}'."

    lines: list[str] = []
    for ds in datasets:
        name = ds.id
        downloads = getattr(ds, "downloads", "?")
        desc = getattr(ds, "description", "") or getattr(ds, "card_data", None) and ""
        pretty = getattr(ds, "pretty_name", "") or ""
        langs = getattr(ds, "languages", None) or []
        if not desc and pretty:
            desc = pretty
        if desc and len(desc) > 120:
            desc = desc[:120] + "…"
        lang_str = ", ".join(langs) if langs else "n/a"
        lines.append(f"- **{name}**  ({downloads:,} downloads, langs: {lang_str})")
        if desc:
            lines.append(f"  {desc}")

    header = f"Found {len(datasets)} dataset(s) for query='{query}':\n"
    return header + "\n".join(lines)


@tool
def get_dataset_info(dataset_id: str) -> str:
    """Get detailed metadata for a Hugging Face dataset: configs, splits, columns, and sizes.

    Use this BEFORE calling eval_prompt or optimize_prompt with an hf_dataset to discover
    the correct config name, split, audio column, and text column.

    Args:
        dataset_id: The HF dataset path (e.g. "Hani89/medical_asr_recording_dataset").
    """
    from datasets import get_dataset_config_names, get_dataset_infos

    lines: list[str] = [f"**{dataset_id}**\n"]

    configs = get_dataset_config_names(dataset_id)
    lines.append(f"Configs: {', '.join(configs)}\n")

    infos = get_dataset_infos(dataset_id)
    for cfg_name, cfg_info in infos.items():
        lines.append(f"### Config: `{cfg_name}`")
        if cfg_info.features:
            cols = list(cfg_info.features.keys())
            lines.append(f"  Columns: {', '.join(cols)}")
        if cfg_info.splits:
            for split_name, split_info in cfg_info.splits.items():
                lines.append(f"  Split `{split_name}`: {split_info.num_examples:,} examples")
        lines.append("")

    return "\n".join(lines)


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "gradio_app.py.tmpl"


@tool
def create_gradio_asr_demo(
    title: str = "AssemblyAI Transcription Demo",
    description: str = "Upload audio or record from your microphone to transcribe with AssemblyAI.",
    prompt: str = "",
    output_path: str = "app.py",
) -> str:
    """Build a Gradio web app for AssemblyAI speech-to-text and write it to disk.

    Generates a complete, ready-to-run Python file that launches a Gradio UI for
    transcribing audio via AssemblyAI. Supports microphone recording and file upload.

    **Use this tool first** whenever the user asks to:
    - Build, create, or make a transcription app, demo, or prototype
    - Build a demo for speech-to-text, ASR, or audio transcription
    - Build a voice app, speech-to-text app, or ASR app
    - Create a UI or web interface for audio transcription
    - Make a Gradio demo or prototype for speech recognition
    - Build something with AssemblyAI features (speaker labels, sentiment, etc.)
    - Create a podcast transcriber, meeting transcriber, or call analyzer
    - Demo AssemblyAI capabilities or showcase transcription

    After generating the app, you can further customize the written file if needed.

    Args:
        title: Title shown at the top of the Gradio app.
        description: Subtitle / description shown below the title.
        prompt: AssemblyAI transcription prompt (e.g. "Transcribe verbatim."). Leave empty for default.
        output_path: File path to write the app to (default: "app.py").
    """
    prompt_default = prompt if prompt else ""
    template = _TEMPLATE_PATH.read_text()
    app_code = template.format(
        prompt_default=prompt_default,
        title=title,
        description=description,
    )

    out = Path(_cwd) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(app_code)

    req_path = out.parent / "requirements.txt"
    req_path.write_text("assemblyai>=0.30\ngradio>=5.0\n")

    return (
        f"Wrote Gradio ASR demo to {out}\n"
        f"Wrote {req_path}\n\n"
        f"Run it with:\n"
        f"  pip install -r requirements.txt\n"
        f"  ASSEMBLYAI_API_KEY=your-key python {output_path}"
    )


_OPENAPI_SPEC_URL = (
    "https://raw.githubusercontent.com/AssemblyAI/assemblyai-api-spec/main/openapi.yml"
)
_spec_cache: str | None = None


def _fetch_openapi_spec() -> str:
    """Fetch the AssemblyAI OpenAPI spec, caching for the session."""
    global _spec_cache
    if _spec_cache is None:
        import urllib.request

        with urllib.request.urlopen(_OPENAPI_SPEC_URL, timeout=15) as resp:
            _spec_cache = resp.read().decode()
    return _spec_cache


@tool
def search_assemblyai_api(query: str) -> str:
    """Search the AssemblyAI OpenAPI spec for endpoints, parameters, and schemas.

    Fetches the latest spec from GitHub. Use this when the user asks about specific
    API endpoints, request/response formats, parameters, or data models.

    Use simple single-word queries for best results (e.g. "transcribe", "speaker_labels").
    Multi-word queries match lines containing ANY of the words, ranked by relevance.

    Args:
        query: Search term (e.g. "transcribe", "speaker_labels", "streaming", "redact_pii").
    """
    try:
        spec = _fetch_openapi_spec()
    except Exception as e:
        return f"Error fetching OpenAPI spec: {e}"

    lines = spec.splitlines()
    words = query.lower().split()
    if not words:
        return "No query provided. Try a keyword like 'transcribe' or 'speaker_labels'."

    # Score each line by how many query words it contains
    scored: list[tuple[int, int]] = []  # (score, line_index)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for w in words if w in line_lower)
        if score > 0:
            scored.append((score, i))

    # Sort by score descending, then by position ascending
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Deduplicate overlapping context windows and collect top matches
    seen_ranges: set[int] = set()
    matches: list[str] = []
    for _score, i in scored:
        if i in seen_ranges:
            continue
        start = max(0, i - 2)
        end = min(len(lines), i + 10)
        seen_ranges.update(range(start, end))
        matches.append("\n".join(lines[start:end]))
        if len(matches) >= 15:
            break

    if not matches:
        return f"No matches for '{query}' in the OpenAPI spec."
    return f"Found {len(matches)} match(es) for '{query}':\n\n" + "\n\n---\n\n".join(matches)
