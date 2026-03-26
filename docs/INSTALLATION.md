# Installation Guide

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repo
git clone https://github.com/zigzagGmbH/finetune-moonshine-asr.git
cd finetune-moonshine-asr

# Install all dependencies
uv sync
```

That's it. `uv sync` reads `pyproject.toml` and installs everything into a local `.venv`.

## Running Scripts

Always prefix commands with `uv run` to use the project's virtual environment:

```bash
uv run python train.py --config configs/mls_cv_german_no_curriculum.yaml
uv run python scripts/inference.py --model ./model --audio sample.wav
uv run python scripts/evaluate.py --model ./model --dataset ./data --split test
```

## Pinned Dependencies (Critical)

These version constraints are tested and required — do not loosen them:

| Package | Constraint | Why |
|---------|-----------|-----|
| `datasets` | `< 4.0.0` | 4.x switches audio decoding to torchcodec, breaks Moonshine |
| `transformers` | `>= 4.35.0, < 4.50.0` | 4.50+ removes `group_by_length`, `fp16_full_eval` params |
| `torch` | `>= 2.0.0` | Tested with 2.11.0+cu130 on RTX 5090 |

See `pyproject.toml` for the full dependency list.

## Optional: Live Transcription

For real-time microphone transcription:

```bash
uv add sounddevice
```

Silero VAD is downloaded automatically via `torch.hub` on first use.

## Optional: ONNX Export

For model conversion and fast inference:

```bash
uv add onnx onnxruntime optimum
```

These are already included in `pyproject.toml`.

## GPU Support

### Verify CUDA

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Known GPU Issues

- **Dual GPU / DataParallel is broken** for Moonshine — always use `CUDA_VISIBLE_DEVICES=0`
- **gradient_checkpointing is broken** in transformers 4.49 for Moonshine — set to `false` in config
- **bf16 is preferred** on RTX 30xx/40xx/50xx; use fp16 on older cards

## Verification

```bash
# Test core imports
uv run python -c "import torch, transformers, datasets; print('Core imports OK')"

# Test audio processing
uv run python -c "import librosa, soundfile, torchaudio; print('Audio processing OK')"

# Test scripts
uv run python scripts/inference.py --help
uv run python scripts/evaluate.py --help
```

## System Dependencies (Linux)

If audio loading fails:

```bash
sudo apt-get install libsndfile1 portaudio19-dev
```

## Note on Pierre's Original

This is a fork of [pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr). The original used `requirements.txt` and `pip`. We migrated to `pyproject.toml` + `uv` for better dependency resolution and reproducibility.
