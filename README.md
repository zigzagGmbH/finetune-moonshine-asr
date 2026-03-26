# Moonshine-Tiny-DE: German ASR Fine-Tuning

Fine-tuning [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) (27M params) for **German automatic speech recognition**, built on [Pierre Chéneau's fine-tuning toolkit](https://github.com/pierre-cheneau/finetune-moonshine-asr).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-moonshine--tiny--de-yellow)](https://huggingface.co/dattazigzag/moonshine-tiny-de)

## Results

| Metric | Value |
|--------|-------|
| **WER** | 36.7% on MLS German test set |
| **Model size** | 27M parameters (~200 MB) |
| **Training data** | MLS German — 469,942 samples (~1,967 hours) |
| **Training time** | ~9.7 hours on single RTX 5090 |
| **Training steps** | 10,000 (schedule-free AdamW, bf16) |

For reference: Pierre achieved 21.8% WER on French with 60k samples at 8k steps using curriculum learning. Our first run used 8× more data but no curriculum — there's room to improve.

## Use the Pre-Trained German Model

### Install

```bash
pip install transformers torch torchaudio
```

### Transcribe a file

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="dattazigzag/moonshine-tiny-de")
result = transcriber("german_audio.wav")
print(result["text"])
```

### Transcribe from code

```python
from transformers import AutoProcessor, MoonshineForConditionalGeneration
import torch, torchaudio

model = MoonshineForConditionalGeneration.from_pretrained("dattazigzag/moonshine-tiny-de")
processor = AutoProcessor.from_pretrained("dattazigzag/moonshine-tiny-de")
model.eval()

waveform, sr = torchaudio.load("german_audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=80)
print(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
```

### Inference with the repo scripts

```bash
git clone https://github.com/zigzagGmbH/finetune-moonshine-asr.git
cd finetune-moonshine-asr && uv sync

# Single file
CUDA_VISIBLE_DEVICES=0 uv run python scripts/inference.py \
    --model dattazigzag/moonshine-tiny-de \
    --audio german_audio.wav

# Batch (directory of WAVs)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/inference.py \
    --model dattazigzag/moonshine-tiny-de \
    --audio ./test_audio/ \
    --output results.json

# Live microphone transcription
uv run python scripts/inference.py \
    --model dattazigzag/moonshine-tiny-de \
    --live

# Evaluate WER on a dataset
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
    --model dattazigzag/moonshine-tiny-de \
    --dataset /path/to/german_dataset \
    --split test
```

## Fine-Tune Your Own

### 1. Prepare data

```bash
uv run python scripts/prepare_german_dataset.py \
    --output /path/to/german_combined \
    --skip-cv
```

### 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/mls_cv_german_no_curriculum.yaml
```

### 3. Resume / train longer

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/mls_cv_german_no_curriculum.yaml \
    --resume /path/to/model/final \
    --max-steps 20000
```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for the full guide (curriculum learning, hyperparameters, troubleshooting).

## Scripts Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `train.py` | Main training script (patched for bf16) | ✅ Tested |
| `scripts/prepare_german_dataset.py` | MLS German data prep | ✅ Tested |
| `scripts/inference.py` | Single-file, batch, and live inference | ✅ Tested |
| `scripts/evaluate.py` | WER/CER evaluation on test sets | ✅ Tested |
| `scripts/convert_for_deployment.py` | ONNX/ORT export pipeline | Planned |
| `scripts/extract_samples.py` | Extract audio samples from datasets | Available |
| `scripts/checkpoint_to_dataset.py` | Convert checkpoints to HF datasets | Available |

## Documentation

| Doc | Content |
|-----|---------|
| [INSTALLATION.md](docs/INSTALLATION.md) | Setup with `uv`, pinned deps, GPU notes |
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Full training guide, curriculum learning, troubleshooting |
| [DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md) | Data formats, quality checks, preprocessing |
| [INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) | Inference, evaluation, and deployment scripts |
| [LIVE_MODE_GUIDE.md](docs/LIVE_MODE_GUIDE.md) | Real-time microphone transcription with VAD |
| [ONNX_MODE_GUIDE.md](docs/ONNX_MODE_GUIDE.md) | ONNX/ORT conversion for fast deployment |
| [moonshine_de_context.md](contexts/moonshine_de_context.md) | Full project context, all gotchas, roadmap |

## What Changed from Pierre's Original

### New files
- `scripts/prepare_german_dataset.py` — MLS German data preparation
- `configs/mls_cv_german_no_curriculum.yaml` — RTX 5090 config (bf16, batch 16)
- `contexts/moonshine_de_context.md` — full project context and roadmap

### Patches to `train.py`
- Safe config key access: `.get("fp16", False)` / `.get("bf16", False)`
- bf16 training support alongside fp16

### Key gotchas documented
- **`gradient_checkpointing` broken** for Moonshine in transformers 4.49
- **Dual GPU / DataParallel broken** for Moonshine KV cache — use `CUDA_VISIBLE_DEVICES=0`
- **`datasets >= 4.0` breaks audio decoding** — pinned `< 4.0`
- **`transformers >= 4.50` removes training params** — pinned `< 4.50`
- **Common Voice pulled from HuggingFace** (Oct 2025) — use `--skip-cv`

Full details in [`contexts/moonshine_de_context.md`](contexts/moonshine_de_context.md).

## Repository Structure

```
finetune-moonshine-asr/
├── train.py                              # Main training script (patched for bf16)
├── configs/
│   ├── mls_cv_german_no_curriculum.yaml  # German RTX 5090 config
│   └── example_config.yaml              # Reference config with all options
├── scripts/
│   ├── prepare_german_dataset.py         # German data prep (MLS + optional CV)
│   ├── inference.py                      # Single-file, batch, and live inference
│   ├── evaluate.py                       # WER/CER evaluation
│   ├── convert_for_deployment.py         # ONNX export pipeline
│   ├── extract_samples.py               # Extract test samples
│   └── checkpoint_to_dataset.py          # Create datasets from checkpoints
├── moonshine_ft/                         # Core fine-tuning library
│   ├── data_loader.py                    # Dataset loading (MLS/CV/local/CSV)
│   ├── curriculum.py                     # Curriculum learning scheduler
│   └── utils/                            # Metrics + audio preprocessing
├── contexts/
│   └── moonshine_de_context.md           # Full project context & roadmap
└── docs/                                 # Guides (installation, training, inference, etc.)
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
