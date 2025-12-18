# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dia is a 1.6B parameter text-to-speech (TTS) model by Nari Labs that generates realistic dialogue from text transcripts. It supports multi-speaker dialogue via `[S1]`/`[S2]` tags, voice cloning via audio prompts, and non-verbal generation (laughter, coughing, etc.).

## Common Commands

```bash
# Install package in development mode
pip install -e .

# Run with uv (recommended)
uv run python example/simple.py

# Lint check
ruff check

# Format check
ruff format --check --diff

# Run Gradio web UI
python app.py

# Run CLI
python cli.py --text "[S1] Hello world."
```

### Example Scripts

```bash
python example/simple.py           # Basic generation
python example/voice_clone.py      # Voice cloning
python example/simple_batch.py     # Batch generation
python example/benchmark.py        # Performance benchmarking
python example/simple-mac.py       # Mac-specific (uses float32)
python example/simple-cpu.py       # CPU-only
```

## Architecture

### Core Module Structure (`/dia/`)

- **model.py**: Main `Dia` class - handles model loading, text encoding, audio generation
- **config.py**: Pydantic configuration classes (`DiaConfig`, `EncoderConfig`, `DecoderConfig`)
- **layers.py**: Neural network layers (transformer blocks, attention, MLP)
- **audio.py**: Audio delay pattern utilities for multi-channel generation
- **state.py**: Inference state management (KV caches, encoder/decoder states)

### Encoder-Decoder Transformer

**Encoder** (12 layers, 1024 hidden, 16 heads):
- Processes byte-level text tokens (vocab size 256)
- Max 1024 position embeddings

**Decoder** (18 layers, 2048 hidden, 16 heads with 4 KV heads):
- Generates 9-channel audio tokens (vocab size 1028)
- Uses cross-attention to encoder outputs
- Max 3072 position embeddings

### Audio Processing

- Uses Descript Audio Codec (DAC) for tokenization/detokenization
- Sample rate: 44100 Hz
- Delay pattern applied across 9 audio channels for causal generation
- Key functions: `apply_audio_delay()`, `revert_audio_delay()` in `audio.py`

### Generation Flow

1. Text → byte tokens (with speaker tags)
2. Encoder pass → context embeddings
3. Optional audio prompt → DAC encode → delay pattern
4. Autoregressive decoder generation with CFG
5. DAC decode → waveform output

### Key Generation Parameters

- `cfg_scale`: Classifier-Free Guidance scale (default 3.0)
- `temperature`: Sampling temperature (default 1.8)
- `top_p`: Nucleus sampling (default 0.90)
- `top_k`: Top-K filtering
- `seed`: For reproducibility

## Device Support

- **CUDA**: Optimized with float16/bfloat16
- **MPS** (Apple Silicon): Uses float32
- **CPU**: Supported, uses float32

Model auto-selects device in `Dia.from_pretrained()`.

## Code Style

- Line length: 119 characters
- Linter: Ruff
- Type hints used throughout
- Pydantic for configuration validation
