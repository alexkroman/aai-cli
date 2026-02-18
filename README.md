# aai-cli

Optimize AssemblyAI universal-3-pro prompts using OPRO-style optimization with TextGrad-inspired gradient feedback.

## Install

```bash
poetry install
```

## Environment Variables

Set these before running:

```bash
export ASSEMBLYAI_API_KEY=your-assemblyai-key
export ANTHROPIC_API_KEY=your-anthropic-key
export HF_TOKEN=your-huggingface-token
```

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
  samples: 50              # total samples split across datasets
  iterations: 5            # optimization iterations
  candidates_per_step: 8   # candidate prompts generated per iteration
  trajectory_size: 5       # number of history entries shown to optimizer
  seed: 42                 # seed for reproducible eval shuffle
  num_threads: 24          # parallel transcription threads
  starting_prompt: >-
    Transcribe every spoken word...
```

## Usage

```bash
# Run with defaults from config.yaml
poetry run aai

# Override parameters via CLI
poetry run aai optimization.samples=50
poetry run aai optimization.iterations=10 optimization.candidates_per_step=4

# Remove a dataset
poetry run aai '~datasets.ami'
```

The optimizer saves state to `outputs/optimization_state.json` after each iteration. Rerunning will automatically resume from the saved state.

## Development

```bash
poetry install
poetry run poe check  # format, lint, typecheck, test
```

## License

MIT
