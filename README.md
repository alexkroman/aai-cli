# aai-cli

Optimize AssemblyAI universal-3-pro prompts using OPRO-style optimization.

## Install

```bash
poetry install
```

## Environment Variables

Set these before running:

```bash
export ASSEMBLYAI_API_KEY=your-assemblyai-key
export HF_TOKEN=your-huggingface-token
```

The AssemblyAI API key is used for both speech transcription and the [LLM Gateway](https://www.assemblyai.com/docs/guides/llm-gateway) (which proxies requests to Claude, GPT, Gemini, etc.).

## Configuration

Edit `aai_cli/conf/config.yaml` to configure datasets and optimization parameters:

```yaml
datasets:
  earnings22:
    path: sanchit-gandhi/earnings22_robust_split
    config: default
    audio_column: audio
    text_column: sentence
    split: test

optimization:
  samples: 100              # total samples split across datasets
  iterations: 10            # optimization iterations
  candidates_per_step: 5    # candidate prompts generated per iteration
  trajectory_size: 5        # number of history entries shown to optimizer
  seed: 42                  # seed for reproducible eval shuffle
  num_threads: 12           # parallel transcription threads
  llm_model: claude-sonnet-4-5-20250929  # model for prompt generation
  starting_prompt: >-
    Transcribe verbatim.
```

### Available LLM Models

Any model supported by the AssemblyAI LLM Gateway can be used:

| Model | Parameter |
|-------|-----------|
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` |
| GPT-5.1 | `gpt-5.1` |
| GPT-5 | `gpt-5` |
| GPT-4.1 | `gpt-4.1` |
| Gemini 2.5 Pro | `gemini-2.5-pro` |
| Gemini 2.5 Flash | `gemini-2.5-flash` |

## Usage

```bash
# Run with defaults from config.yaml
poetry run aai

# Override parameters via CLI
poetry run aai optimization.samples=50
poetry run aai optimization.iterations=10 optimization.candidates_per_step=4
poetry run aai optimization.llm_model=gemini-2.5-pro
```

The optimizer saves state to `outputs/optimization_state.json` after each iteration. Rerunning will automatically resume from the saved state.

## Development

```bash
poetry install
poetry run poe check  # format, lint, typecheck, test
```

## License

MIT
