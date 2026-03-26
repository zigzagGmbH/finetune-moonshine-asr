# Moonshine-Tiny-DE: German ASR Fine-Tuning

Fine-tuning [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) (27M params) for **German automatic speech recognition**, built on [Pierre Chéneau's fine-tuning toolkit](https://github.com/pierre-cheneau/finetune-moonshine-asr).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-moonshine--tiny--de-yellow)](https://huggingface.co/dattazigzag/moonshine-tiny-de)

## Overview

This is a fork of [pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr) adapted for **German language** fine-tuning on an NVIDIA RTX 5090. The original toolkit provides curriculum learning, schedule-free optimization, and production-ready inference scripts for Moonshine ASR.

**Our contribution:** a German fine-tuning pipeline using MLS German data, RTX 5090-optimized config, and patches for bf16 training + dual-GPU compatibility issues.

### Results

| Metric | Value |
|--------|-------|
| **WER** | 36.7% on MLS German test set |
| **Model size** | 27M parameters (~200 MB) |
| **Training data** | MLS German — 469,942 samples (~1,967 hours) |
| **Training time** | ~9.7 hours on single RTX 5090 |
| **Training steps** | 10,000 (schedule-free AdamW, bf16) |

For reference: Pierre achieved 21.8% WER on French with 60k samples at 8k steps using curriculum learning. Our first run used 8× more data but no curriculum — there's room to improve.

## Pre-trained German Model

**[dattazigzag/moonshine-tiny-de](https://huggingface.co/dattazigzag/moonshine-tiny-de)** — ready to use:

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="dattazigzag/moonshine-tiny-de")
result = transcriber("german_audio.wav")
print(result["text"])
```

## Quick Start: Fine-Tune Your Own

```bash
# Clone
git clone https://github.com/zigzagGmbH/finetune-moonshine-asr.git
cd finetune-moonshine-asr

# Install dependencies (uses uv)
uv sync

# Prepare dataset (MLS German, ~1.1 TB decoded)
uv run python scripts/prepare_german_dataset.py \
    --output /path/to/german_combined \
    --skip-cv

# Train (single GPU, bf16)
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/mls_cv_german_no_curriculum.yaml
```

## What Changed from Pierre's Original

### New files
- `scripts/prepare_german_dataset.py` — MLS German data preparation pipeline
- `configs/mls_cv_german_no_curriculum.yaml` — RTX 5090 training config (bf16, batch 16)
- `contexts/moonshine_de_context.md` — full project context, gotchas, and roadmap

### Patches to `train.py`
- Safe config key access: `train_config.get("fp16", False)` / `train_config.get("bf16", False)` (Pierre's original hardcoded `train_config['fp16']`)
- bf16 training support alongside fp16

### Key gotchas documented
- **`gradient_checkpointing` is broken** for Moonshine in transformers 4.49 (`bool` vs dict `is_updated` error) — set to `false`
- **Dual GPU / DataParallel is broken** for Moonshine KV cache — always use `CUDA_VISIBLE_DEVICES=0`
- **`datasets >= 4.0` breaks audio decoding** (switches to torchcodec) — pinned `< 4.0`
- **`transformers >= 4.50` removes training params** — pinned `< 4.50`
- **Common Voice pulled from HuggingFace** (Oct 2025) — use `--skip-cv` flag

Full details in [`contexts/moonshine_de_context.md`](contexts/moonshine_de_context.md).

## Pinned Dependencies

These versions are tested and working. Do not upgrade without testing:

```
datasets >= 2.14.0, < 4.0.0
transformers >= 4.35.0, < 4.50.0
torch >= 2.0.0  (tested with 2.11.0+cu130)
```

See `pyproject.toml` for the full list.

## Repository Structure

```
finetune-moonshine-asr/
├── train.py                              # Main training script (patched for bf16)
├── configs/
│   ├── mls_cv_german_no_curriculum.yaml  # German RTX 5090 config
│   └── example_config.yaml              # Reference config with all options
├── scripts/
│   ├── prepare_german_dataset.py         # German data prep (MLS + optional CV)
│   ├── inference.py                      # Single-file and live inference
│   ├── evaluate.py                       # WER/CER evaluation
│   ├── convert_for_deployment.py         # ONNX export pipeline
│   ├── intelligent_segmentation.py       # Whisper V3 forced-alignment segmentation
│   ├── extract_samples.py               # Extract test samples
│   └── checkpoint_to_dataset.py          # Create datasets from checkpoints
├── moonshine_ft/                         # Core fine-tuning library
│   ├── data_loader.py
│   ├── curriculum.py
│   └── utils/
├── contexts/
│   └── moonshine_de_context.md           # Full project context & roadmap
├── docs/                                 # Pierre's original guides (still useful)
│   ├── TRAINING_GUIDE.md
│   ├── INFERENCE_GUIDE.md
│   ├── DATASET_PREPARATION.md
│   ├── LIVE_MODE_GUIDE.md
│   └── ONNX_MODE_GUIDE.md
└── examples/
    └── fine_tune_moonshine_curriculum.ipynb
```

## Next Steps

- [ ] Improve WER: resume training (20k steps), add curriculum learning
- [ ] Add more data sources: SWC (~386h), VoxPopuli DE (~282h), Bundestag (~600h)
- [ ] ONNX/ORT export for native `moonshine-voice` CLI integration
- [ ] Test with real-world German audio (conversational, accented, noisy)
- [ ] Post results to [moonshine#141](https://github.com/moonshine-ai/moonshine/issues/141)

## Acknowledgments

- **[Pierre Chéneau](https://github.com/pierre-cheneau/finetune-moonshine-asr)** — original fine-tuning toolkit and French model ([moonshine-tiny-fr](https://huggingface.co/Cornebidouil/moonshine-tiny-fr), 21.8% WER)
- **[Moonshine AI / Useful Sensors](https://github.com/moonshine-ai/moonshine)** — base model and architecture
- **[German language support community](https://github.com/moonshine-ai/moonshine/issues/141)** — dataset recommendations and discussion

## License

MIT — same as Pierre's original. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{datta2026moonshine-tiny-de,
  author = {Saurabh Datta},
  title = {Moonshine-Tiny-DE: Fine-tuned German Speech Recognition},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/dattazigzag/moonshine-tiny-de}
}
```

Based on:
```bibtex
@misc{cheneau2026moonshine-finetune,
  author = {Pierre Chéneau (Cornebidouil)},
  title = {Moonshine ASR Fine-Tuning Guide},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/pierre-cheneau/finetune-moonshine-asr}
}
```
