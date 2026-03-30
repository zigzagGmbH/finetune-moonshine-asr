---
license: mit
language:
- de
tags:
- automatic-speech-recognition
- moonshine
- german
- asr
- speech
datasets:
- facebook/multilingual_librispeech
metrics:
- wer
base_model: UsefulSensors/moonshine-tiny
model-index:
- name: moonshine-tiny-de
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: MLS German (test split)
      type: facebook/multilingual_librispeech
      args: german
    metrics:
    - name: WER
      type: wer
      value: 36.7
---

# Moonshine-Tiny-DE: Fine-tuned German Speech Recognition

Fine-tuned [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) for German automatic speech recognition.

## Model Details

- **Base model:** UsefulSensors/moonshine-tiny (27M parameters)
- **Language:** German (de)
- **Training data:** MLS German — 469,942 samples (~1,967 hours of audiobook speech)
- **WER:** 36.7% on MLS German test set (3,394 samples)
- **Training:** 10,000 steps, schedule-free AdamW, bf16, effective batch size 64
- **Hardware:** Single NVIDIA RTX 5090 (32 GB), ~9.7 hours

## Usage

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="dattazigzag/moonshine-tiny-de")
result = transcriber("german_audio.wav")
print(result["text"])
```

### Batch processing

```python
from pathlib import Path

audio_files = Path("./audio").glob("*.wav")
for audio in audio_files:
    result = transcriber(str(audio))
    print(f"{audio.name}: {result['text']}")
```

### With explicit model loading

```python
from transformers import AutoProcessor, MoonshineForConditionalGeneration
import torch

model = MoonshineForConditionalGeneration.from_pretrained("dattazigzag/moonshine-tiny-de")
processor = AutoProcessor.from_pretrained("dattazigzag/moonshine-tiny-de")
model.eval()

# Process audio (16kHz mono WAV)
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=80)
text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## Training Details

### Approach

This is **not** trained from scratch. We fine-tuned the English-only moonshine-tiny model to understand German. The pre-trained model already knew audio feature extraction, attention patterns, and tokenization — we adapted it to German phonetics and vocabulary.

### Configuration

| Setting | Value |
|---------|-------|
| Optimizer | schedule-free AdamW |
| Learning rate | 3e-4 (constant after 300-step warmup) |
| Precision | bf16 |
| Batch size | 16 per device × 4 accumulation = 64 effective |
| Audio duration | 4–20 seconds |
| Gradient checkpointing | Disabled (broken with Moonshine in transformers 4.49) |
| Curriculum learning | Disabled (simple first run) |

### Training curve

| Step | Loss | WER |
|------|------|-----|
| 500 | 2.37 | — |
| 1,000 | 2.04 | 46.5% |
| 5,000 | ~1.65 | ~39% |
| 10,000 | 1.61 | **36.7%** |

### Error patterns

- Phonetically similar confusions: b/p, d/t, ck/x (classic German ASR challenges)
- Compound word splitting errors: "herzaubern" → "herr sauben"
- Longer sequences degrade more than shorter ones
- Audiobook speech only — no conversational speech exposure

## Limitations

- **Audiobook speech only** — trained on MLS (read speech). May underperform on conversational, noisy, or accented German.
- **First training run** — WER can likely be improved with curriculum learning, more training steps, or additional data sources (SWC, VoxPopuli, Bundestag).
- **No Common Voice data** — Mozilla pulled it from HuggingFace in Oct 2025, so we lack speaker diversity.
- **HuggingFace transformers only** — produces safetensors format, not the `.ort` format for the native `moonshine-voice` CLI. ONNX conversion is a planned next step.

## Fine-tuning toolkit

Trained using a fork of [Pierre Chéneau's finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr) with German-specific adaptations:

- [Training config](https://github.com/zigzagGmbH/finetune-moonshine-asr/blob/main/configs/mls_cv_german_no_curriculum.yaml)
- [Data preparation script](https://github.com/zigzagGmbH/finetune-moonshine-asr/blob/main/scripts/prepare_german_dataset.py)
- [Full context & gotchas](https://github.com/zigzagGmbH/finetune-moonshine-asr/blob/main/contexts/moonshine_de_context.md)

## Acknowledgments

- [Moonshine AI / Useful Sensors](https://github.com/moonshine-ai/moonshine) for the base model
- [Pierre Chéneau](https://github.com/pierre-cheneau/finetune-moonshine-asr) for the fine-tuning toolkit and [moonshine-tiny-fr](https://huggingface.co/Cornebidouil/moonshine-tiny-fr) (21.8% WER French reference)
- [German language support community (issue #141)](https://github.com/moonshine-ai/moonshine/issues/141)

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
