"""Agent tools for searching datasets and API specs."""

import functools

from smolagents import tool

_OPENAPI_SPEC_URL = (
    "https://raw.githubusercontent.com/AssemblyAI/assemblyai-api-spec/main/openapi.yml"
)


@functools.lru_cache(maxsize=1)
def _fetch_openapi_spec() -> str:
    """Fetch the AssemblyAI OpenAPI spec, caching for the session."""
    import urllib.request

    with urllib.request.urlopen(_OPENAPI_SPEC_URL, timeout=15) as resp:
        return resp.read().decode()


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
        card = getattr(ds, "card_data", None)
        desc = getattr(ds, "description", "") or (card.get("description", "") if card else "")
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
