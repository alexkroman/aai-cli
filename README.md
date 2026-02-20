# aai-cli

A voice AI coding agent for building applications with AssemblyAI. Includes tools for prompt evaluation, optimization, dataset discovery, and app scaffolding.

## Install

```bash
poetry install
```

To make the `aai` command available globally:

```bash
pipx install .
```

## Environment Variables

```bash
export ANTHROPIC_API_KEY=your-anthropic-key    # powers the coding agent (Claude)
export ASSEMBLYAI_API_KEY=your-assemblyai-key  # speech transcription
export HF_TOKEN=your-huggingface-token         # Hugging Face dataset access
```

## Usage

### Interactive Agent (default)

```bash
aai
```

This launches an interactive coding agent that can:
- Build Gradio transcription apps with `create_gradio_asr_demo`
- Evaluate transcription prompts with `eval_prompt`
- Optimize prompts with `optimize_prompt`
- Search Hugging Face for audio datasets
- Look up AssemblyAI API features

Type `/ideas` for inspiration, `/help` for commands.

### Evaluate a Prompt

```bash
aai eval --prompt "Transcribe verbatim." --max-samples 50
aai eval --dataset earnings22 --max-samples 20
aai eval --hf-dataset mozilla-foundation/common_voice_11_0 --hf-config en
```

### Optimize a Prompt

```bash
aai optimize --starting-prompt "Transcribe verbatim." --iterations 5 --samples 50
aai optimize --llm-model claude-sonnet-4-6 --output results.json
```

## Configuration

Edit `aai_cli/conf/config.yaml` to configure datasets and defaults:

```yaml
eval:
  max_samples: 50
  prompt: "Transcribe verbatim."
  num_threads: 12

datasets:
  earnings22:
    path: sanchit-gandhi/earnings22_robust_split
    config: default
    audio_column: audio
    text_column: sentence
    split: test

optimization:
  samples: 100
  iterations: 250
  starting_prompt: >-
    Transcribe verbatim.
  num_threads: 12
  llm_model: claude-sonnet-4-6
```

## Development

```bash
poetry install
poetry run poe check  # format, lint, typecheck, test
```

## License

MIT
