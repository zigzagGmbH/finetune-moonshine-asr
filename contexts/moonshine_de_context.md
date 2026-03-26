# Moonshine German (DE) Fine-Tuning — Context & Next Steps

**Last updated:** 2026-03-26
**Author:** Saurabh Datta (dattazigzag) + Claude
**Status:** First model trained (36.7% WER). Multiple follow-up steps pending.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Architecture & Approach](#2-architecture--approach)
3. [Server & Storage Layout](#3-server--storage-layout)
4. [All the Gotchas (Dependency Hell Log)](#4-all-the-gotchas)
5. [Current Model Results](#5-current-model-results)
6. [NEXT: Push Model to HuggingFace](#6-next-push-model-to-huggingface)
7. [NEXT: Git & Repo Strategy](#7-next-git--repo-strategy)
8. [NEXT: ONNX Export for moonshine-voice CLI](#8-next-onnx-export-for-moonshine-voice-cli)
9. [NEXT: Improve WER (More Training)](#9-next-improve-wer)
10. [NEXT: Test with Real-World German Audio](#10-next-test-with-real-world-audio)
11. [NEXT: Cleanup](#11-next-cleanup)
12. [Key File Locations](#12-key-file-locations)
13. [Exact Commands That Work](#13-exact-commands-that-work)
14. [Pinned Dependencies](#14-pinned-dependencies)
15. [References](#15-references)

---

## 1. What We Built

A **German ASR model** fine-tuned from `UsefulSensors/moonshine-tiny` (27M params, English-only) using 469,942 samples (~1,967 hours) of MLS German audiobook data.

**This is NOT trained from scratch.** We fine-tuned an existing English speech-to-text model to understand German. The pre-trained model already knew audio feature extraction, attention patterns, and tokenization — we adapted it to German phonetics and vocabulary.

**Results after 10,000 steps (~9.7 hours on RTX 5090):**
- Final train loss: 1.61
- Final eval WER: **36.7%**
- Model size: ~200 MB at `/data/results-moonshine-de/final/`

For comparison: Pierre Chéneau got **21.8% WER** on French with 60k samples at 8k steps. We used 8× more data but only 10k steps. The loss plateaued around 1.61, suggesting more training or hyperparameter tuning is needed.

---

## 2. Architecture & Approach

### What is Moonshine?
- Fast, edge-optimized ASR by Moonshine AI (formerly Useful Sensors)
- Encoder-decoder transformer with RoPE embeddings
- Supports variable-length input (no 30s padding like Whisper)
- "Flavors of Moonshine" paper: monolingual tiny models outperform multilingual for 27M params

### What we fine-tuned
- **Base model:** `UsefulSensors/moonshine-tiny` (HuggingFace safetensors, 27M params)
- **Framework:** HuggingFace `Seq2SeqTrainer` via Pierre Chéneau's `finetune-moonshine-asr` toolkit
- **Optimizer:** Schedule-free AdamW (same as Moonshine paper)
- **Precision:** bf16 (RTX 5090 native)
- **No curriculum learning** (simple first run)

### What this produces
- HuggingFace `transformers`-compatible model (safetensors format)
- **NOT** the `.ort` format that the native `moonshine-voice` CLI uses
- Inference via `transformers.pipeline("automatic-speech-recognition")` or Pierre's `scripts/inference.py`
- ONNX/ORT conversion needed for native `moonshine-voice` integration (see section 8)

### Dataset
- **MLS German** (`facebook/multilingual_librispeech`, `"german"`)
- 469,942 train samples, 3,394 test samples
- Duration: 10.0s - 20.0s (all within Moonshine paper's [4, 30]s recommendation)
- Clean read-speech audiobooks, verified transcripts, CC BY 4.0
- Common Voice was planned but Mozilla pulled it from HuggingFace (Oct 2025)

### German-specific considerations (from issue #141)
- German subtitle data paraphrases rather than transcribes verbatim (RobFlo98's warning)
- Boilerplate like "Untertitelung. BR" can leak into pseudo-labeled data
- Stick to verified transcript sources: MLS, SWC, M-AILABS, Bundestag

---

## 3. Server & Storage Layout

**Server:** ziggie.is — dual RTX 5090 (32 GB each), Ryzen Threadripper PRO (48 cores), 125 GB RAM

```
/           295 GB  — OS, Docker, ~/.cache/huggingface/hub/ (49 GB model cache)
/data       1.5 TB  — AI models, training outputs, service databases
/storage    1.8 TB  — Bulk datasets, archives
```

### Current disk usage for this project

| Path | Size | What | Deletable? |
|------|------|------|------------|
| `/storage/datasets/german_combined/` | 1.1 TB | Decoded MLS German (float32 Arrow arrays) | Yes, after training done |
| `/data/results-moonshine-de_encoded/` | ~426 GB | Tokenized training cache | Yes, after training done |
| `/data/results-moonshine-de/final/` | ~200 MB | **TRAINED MODEL — KEEP** | NO |
| `/data/results-moonshine-de/checkpoint-*/` | ~500 MB each | Intermediate checkpoints | Yes, keep best only |
| `~/.cache/huggingface/hub/` | 49 GB | Downloaded model weights (on `/`) | Yes, re-downloadable |
| `/data/logs/moonshine-de-no-curriculum/` | ~few MB | TensorBoard logs | Keep for reference |

### Important storage rules
- HF datasets cache was symlinked: `~/.cache/huggingface/datasets/` → `/storage/datasets/huggingface_cache/` (currently deleted to save space, recreate if needed)
- Never point large outputs to `/` — always use `/data/` or `/storage/`
- MLS German: 30 GB compressed → **1.1 TB** as decoded Arrow arrays. Plan accordingly.

---

## 4. All the Gotchas

### 4.1 Dependency Versions (CRITICAL)
Pierre's toolkit was built for older versions. We hit breaking changes with newer ones.

```
# WHAT WORKS (pinned):
datasets >= 2.14.0, < 4.0.0    # 4.x requires torchcodec, breaks audio decoding
transformers >= 4.35.0, < 4.50.0  # 5.x removes Seq2SeqTrainingArguments params
torch 2.11.0+cu130              # latest, works fine
```

**Do NOT upgrade datasets to 4.x** — it switches audio decoding from soundfile/librosa to torchcodec, which requires FFmpeg system libs and has CUDA compatibility issues.

**Do NOT upgrade transformers to 5.x** — it removes `group_by_length`, `fp16_full_eval`, and other params that `train.py` uses.

### 4.2 Config Adaptations for RTX 5090

Pierre's config was for RTX 2060 (6 GB). Key changes for 5090 (32 GB):

| Setting | Pierre (2060) | Ours (5090) | Why |
|---------|--------------|-------------|-----|
| `fp16` | `true` | `false` | 5090 uses bf16 natively |
| `bf16` | not present | `true` | Better precision than fp16 |
| `batch_size` | 4 | 16 | More VRAM |
| `gradient_checkpointing` | `true` | **`false`** | Broken with Moonshine in transformers 4.49 |

### 4.3 gradient_checkpointing is BROKEN for Moonshine
`modeling_moonshine.py` line 235 expects `past_key_value.is_updated` to be dict-like, but gradient checkpointing passes it as a plain `bool`. Error:
```
AttributeError: 'bool' object has no attribute 'is_updated'
```
**Fix:** Set `gradient_checkpointing: false`. Not needed anyway — 27M model fits easily in 32 GB.

### 4.4 Dual GPU / DataParallel is BROKEN for Moonshine
PyTorch auto-detects both 5090s and wraps in `DataParallel`, which breaks KV cache handling (same `is_updated` error).
**Fix:** Always use `CUDA_VISIBLE_DEVICES=0` to force single GPU.

### 4.5 train.py expects `fp16` key even when using bf16
Pierre's `train.py` hardcodes `train_config['fp16']`. When config only has `bf16`, it throws `KeyError`.
**Fix:** Add both to config + patch train.py to use `.get()`:
```python
fp16=train_config.get("fp16", False),
bf16=train_config.get("bf16", False),
```

### 4.6 Common Voice pulled from HuggingFace
As of October 2025, all `mozilla-foundation/common_voice_*` datasets return `EmptyDatasetError`. Mozilla moved them to Mozilla Data Collective (requires separate account + download).

### 4.7 `trust_remote_code` deprecated
`datasets >= 4.0.0` dropped this parameter. Remove it from any `load_dataset()` calls. (We pinned to < 4.0 so this is moot, but note for future.)

### 4.8 Final eval dtype mismatch (cosmetic)
After training completes, the final evaluation crashes with `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same`. This is a known issue with bf16 training — **the model is saved correctly before this error**. Use `scripts/evaluate.py` separately to get final metrics.

### 4.9 HF cache explodes disk space
Every `.map()` call creates a new Arrow cache. Multiple failed runs = hundreds of GB of orphaned caches. Monitor with `du -sh ~/.cache/huggingface/datasets/` and clean regularly.

### 4.10 `num_proc` > 8 causes OOM
With 125 GB RAM, 16 workers decoding audio simultaneously caused OOM kills. 8 workers is the sweet spot.

---

## 5. Current Model Results

### Training Curve

| Step | Loss | WER | Epoch |
|------|------|-----|-------|
| 50 | 5.17 | — | 0.01 |
| 100 | 4.49 | — | 0.01 |
| 300 | 2.73 | — | 0.04 |
| 500 | 2.37 | — | 0.06 |
| 1000 | 2.04 | 46.5% | 0.14 |
| 5000 | ~1.65 | ~39% | 0.68 |
| 9000 | 1.61 | 37.0% | 1.23 |
| 10000 | 1.61 | **36.7%** | 1.36 |

### Inference Examples (from test set)

```
Sample 1:
  Predicted:  degen sie soeben weilten meine gedanken bei ihnen in allele...
  Reference:  denken sie soeben weilten meine gedanken bei ihnen in adelaide...
  → Minor phonetic confusions (d/t, compound words)

Sample 4 (near perfect):
  Predicted:  freund mich freut mich eine edle treue seele und ein gelungen mensch
  Reference:  freut mich freut mich eine edle treue seele und ein gelungener mensch
```

### Error Patterns
- Phonetically similar confusions: b/p, d/t, ck/x (classic German ASR challenges)
- Compound word splitting: "herzaubern" → "herr sauben"
- Longer sequences degrade more than shorter ones
- Audiobook style only (no conversational speech exposure)

---

## 6. NEXT: Push Model to HuggingFace

### Step 1: Create a HuggingFace model repo

Go to https://huggingface.co/new and create a new model. Suggested naming: `dattazigzag/moonshine-tiny-de`

### Step 2: Create a model card

On ziggie, create `README.md` inside the model directory:

```bash
cat > /data/results-moonshine-de/final/README.md << 'EOF'
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
- **Training data:** MLS German (~1,967 hours, 469,942 samples)
- **WER:** 36.7% on MLS German test set
- **Training:** 10,000 steps, schedule-free AdamW, bf16, batch size 64

## Usage

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="dattazigzag/moonshine-tiny-de")
result = transcriber("german_audio.wav")
print(result["text"])
```

## Training

Fine-tuned using [finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr) toolkit on a single NVIDIA RTX 5090 (32 GB). Training took ~9.7 hours.

## Limitations

- Trained on audiobook speech only (MLS) — may underperform on conversational, noisy, or accented speech
- First training run — WER can likely be improved with curriculum learning or more steps
- Does not include Common Voice data (diverse speakers/accents)

## Acknowledgments

- [Moonshine AI / Useful Sensors](https://github.com/moonshine-ai/moonshine) for the base model
- [Pierre Chéneau](https://github.com/pierre-cheneau/finetune-moonshine-asr) for the fine-tuning toolkit
- [German Language Support community (issue #141)](https://github.com/moonshine-ai/moonshine/issues/141)
EOF
```

### Step 3: Upload to HuggingFace

```bash
# On ziggie
cd /data/results-moonshine-de/final/
uvx hf auth login  # if not already logged in

# Upload all files
uvx hf upload dattazigzag/moonshine-tiny-de . --repo-type model
```

Or using Python:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/data/results-moonshine-de/final",
    repo_id="dattazigzag/moonshine-tiny-de",
    repo_type="model",
)
```

### Step 4: Test that the upload works

```python
from transformers import pipeline
transcriber = pipeline("automatic-speech-recognition", model="dattazigzag/moonshine-tiny-de")
# Should download and load successfully
```

---

## 7. NEXT: Git & Repo Strategy

### Recommendation: Fork + own repo

**Don't PR back to Pierre's repo** — our changes are German-specific with ziggie-specific configs and a different data prep pipeline. Pierre's repo is a French fine-tuning guide; ours diverges significantly.

### Recommended approach:

1. **Fork Pierre's repo** to your GitHub (`dattasaurabh82/finetune-moonshine-asr` or `dattazigzag/moonshine-de`)
2. Keep your existing branch work
3. Add German-specific files:
   - `scripts/prepare_german_dataset.py`
   - `configs/mls_cv_german_no_curriculum.yaml`
   - `docs/GERMAN_FINETUNE_GUIDE.md` (based on this context doc)
   - This context file as `contexts/moonshine_de_context.md`
4. Clean up: remove French-specific configs you don't need, update README

### On ziggie:

```bash
cd ~/Projects/finetune-moonshine-asr

# If you haven't already forked on GitHub, do that first via web UI
# Then update your remote:
git remote set-url origin git@github-dattazigzag:dattazigzag/moonshine-de.git
# (using your SSH alias)

# Commit your work
git add scripts/prepare_german_dataset.py configs/mls_cv_german_no_curriculum.yaml
git commit -m "feat: German fine-tuning pipeline — MLS dataset, RTX 5090 config"

# Push
git push origin your-branch-name
```

### What to credit Pierre:
- Keep his license (MIT)
- Acknowledge in your README that this is based on his toolkit
- Link back to his repo and the French model

---

## 8. NEXT: ONNX Export for moonshine-voice CLI

This is the path to using your model with the native `moonshine-voice` CLI (`mic_transcriber`, `intent_recognizer`, etc.).

### The pipeline:

```
HuggingFace safetensors → ONNX → ORT (flatbuffer) → moonshine-voice CLI
```

### Step 1: Export to ONNX using Pierre's script

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/convert_for_deployment.py \
    --model /data/results-moonshine-de/final \
    --output /data/results-moonshine-de/onnx
```

### Step 2: Convert ONNX to ORT format

The native `moonshine-voice` library uses ORT flatbuffer format (`.ort` files). Moonshine's quantization approach (from their README):
- 8-bit weights across the board
- 8-bit calculations for MatMul
- Frontend convolution layers kept at B16 float precision

Their conversion scripts are at `scripts/quantize-streaming-model.sh` in the moonshine repo. This step may require additional tooling from the Moonshine team.

### Step 3: Test with moonshine-voice CLI

```bash
# In your moonshine (not finetune) repo
cd ~/Projects/moonshine
uv run -m moonshine_voice.mic_transcriber --model-path /path/to/ort/model --model-arch <arch_number>
```

### ⚠️ Important caveat
The Moonshine v2 native library expects a specific ORT model structure (encoder, decoder, tokenizer). The fine-tuned HuggingFace model may not directly convert to the exact format the CLI expects. This might require:
- Studying the Moonshine model download format (`encoder_model.ort`, `decoder_model_merged.ort`, `tokenizer.bin`)
- Potentially asking the Moonshine team for guidance (Pete Warden commented on issue #141)
- The `optimum` library's ONNX export might get you part of the way

### Step 4: Inform the community

Post on [GitHub issue #141](https://github.com/moonshine-ai/moonshine/issues/141):
- Share your WER results
- Link to the HuggingFace model
- Share the training config and approach
- Ask for guidance on ORT conversion if needed
- Pete Warden has expressed interest — he might help with native integration

---

## 9. NEXT: Improve WER

Current: **36.7% WER**. Target: **< 25% WER** (Pierre got 21.8% on French).

### Option A: More training steps (easiest)

The loss was still slowly dropping at step 10,000. Simply resume training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/mls_cv_german_no_curriculum.yaml \
    --resume /data/results-moonshine-de/final \
    --max-steps 20000
```

This continues from where we left off. The encoded dataset cache is already on `/data/` so it starts training immediately.

**Expected improvement:** Loss at 1.61 was still slowly dropping. 20k steps (~1.36 → ~2.7 epochs) should bring WER to ~30-32%.

### Option B: Curriculum learning (recommended for best results)

Create a curriculum config that trains in phases:
1. Phase 1 (4-10s, short utterances): 4,000 steps — learn basic German phonetics
2. Phase 2 (10-20s, medium): 6,000 steps — learn longer sequences
3. Phase 3 (4-20s, full range): 5,000 steps — refine on complete distribution

```bash
# Create curriculum config (TODO: write this config)
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/german_curriculum.yaml \
    --phase 1
```

### Option C: Learning rate tuning

The loss plateau at 1.61 might benefit from:
- Lower learning rate (1e-4 instead of 3e-4) for the second half of training
- Cosine decay schedule instead of constant
- Larger effective batch size (increase gradient_accumulation_steps)

### Option D: More diverse data

MLS is audiobook-only. Adding these would improve real-world performance:
- **Spoken Wikipedia Corpus (SWC):** ~386h, CC BY-SA 4.0 — different speaking style
- **German Bundestag:** ~600h, parliamentary speech — formal but natural
- **VoxPopuli (de):** ~282h, EU Parliament, CC0 — natural speech
- **M-AILABS:** ~237h, audiobooks — similar to MLS but different recordings

Common Voice (~370h, diverse speakers) requires manual download from Mozilla Data Collective.

### Option E: Combine approaches

Best results: resume from current model + add curriculum + more data sources.

### ⚠️ Storage requirement
**Don't delete `/storage/datasets/german_combined/` (1.1 TB) yet** if you plan to train more. The encoded cache on `/data/` can be regenerated, but the decoded dataset takes 2+ hours to recreate.

---

## 10. NEXT: Test with Real-World Audio

### Quick test with test set (already done)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -c "
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from datasets import load_from_disk
import torch
model = MoonshineForConditionalGeneration.from_pretrained('/data/results-moonshine-de/final')
processor = AutoProcessor.from_pretrained('/data/results-moonshine-de/final')
model.eval(); model.to('cuda')
ds = load_from_disk('/storage/datasets/german_combined')
test = ds['test'].select(range(5))
for i in range(5):
    audio = test[i]['audio']
    inputs = processor(audio['array'], sampling_rate=audio['sampling_rate'], return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=80)
    print(f'Predicted: {processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)}')
    print(f'Reference: {test[i][\"sentence\"]}')
    print()
"
```

### Test with a real WAV file
Record German speech on your Mac, scp it to ziggie, then:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/inference.py \
    --model /data/results-moonshine-de/final \
    --audio /path/to/german_speech.wav
```

### Test with FLEURS German (small benchmark dataset)
```python
from datasets import load_dataset
fleurs = load_dataset("google/fleurs", "de_de", split="test", streaming=True)
# Process first 10 samples through the model
```

### Formal evaluation with WER
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
    --model /data/results-moonshine-de/final \
    --dataset /storage/datasets/german_combined \
    --split test
```

---

## 11. NEXT: Cleanup

### After all training and evaluation is done:

```bash
# FREE ~1.8 TB
rm -rf /storage/datasets/german_combined          # 1.1 TB — decoded dataset
rm -rf /data/results-moonshine-de_encoded          # ~426 GB — tokenized cache
rm -rf /data/results-moonshine-de/checkpoint-*/    # intermediate checkpoints
rm -rf ~/.cache/huggingface/hub/                   # 49 GB — model cache on root

# KEEP
/data/results-moonshine-de/final/                  # ~200 MB — YOUR MODEL
/data/logs/moonshine-de-no-curriculum/             # TensorBoard logs
~/Projects/finetune-moonshine-asr/                 # Repo + configs + scripts
```

### If you plan to train more (don't delete yet):
- Keep `/storage/datasets/german_combined/` (1.1 TB) — takes 2+ hours to regenerate
- Keep `/data/results-moonshine-de/final/` — resume training from this checkpoint
- Can delete `/data/results-moonshine-de_encoded/` — regenerates in ~15 min

---

## 12. Key File Locations

| What | Path |
|------|------|
| **Trained model** | `/data/results-moonshine-de/final/` |
| Training repo | `~/Projects/finetune-moonshine-asr/` |
| Training config | `configs/mls_cv_german_no_curriculum.yaml` |
| Data prep script | `scripts/prepare_german_dataset.py` |
| Decoded dataset (1.1 TB) | `/storage/datasets/german_combined/` |
| Encoded/tokenized cache | `/data/results-moonshine-de_encoded/` |
| TensorBoard logs | `/data/logs/moonshine-de-no-curriculum/` |
| All checkpoints | `/data/results-moonshine-de/checkpoint-*/` |
| HF model cache | `~/.cache/huggingface/hub/` |
| Moonshine repo (for CLI) | `~/Projects/moonshine/` |
| This context doc | Copy to repo as `contexts/moonshine_de_context.md` |

---

## 13. Exact Commands That Work

### Start training (single GPU, bf16, no gradient checkpointing)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config configs/mls_cv_german_no_curriculum.yaml
```

### Resume training from checkpoint
```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/mls_cv_german_no_curriculum.yaml \
    --resume /data/results-moonshine-de/final \
    --max-steps 20000
```

### Run inference on test set
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/inference.py \
    --model /data/results-moonshine-de/final \
    --audio some_german.wav
```

### TensorBoard
```bash
uv run tensorboard --logdir /data/logs/moonshine-de-no-curriculum --host 0.0.0.0 --port 6006
# Open http://192.168.178.160:6006
```

### Data prep (if needed again)
```bash
uv run python scripts/prepare_german_dataset.py \
    --output /storage/datasets/german_combined \
    --skip-cv
```

### Upload model to HuggingFace
```bash
cd /data/results-moonshine-de/final/
uvx hf upload dattazigzag/moonshine-tiny-de . --repo-type model
```

### Monitor during training
```bash
# GPU: nvtop
# CPU/RAM: btop
# Disk: watch -n 30 duf
```

---

## 14. Pinned Dependencies

**These versions are tested and working. Do not upgrade without testing.**

```
datasets >= 2.14.0, < 4.0.0
transformers >= 4.35.0, < 4.50.0
torch >= 2.0.0  (tested with 2.11.0+cu130)
torchaudio >= 2.0.0
accelerate >= 0.24.0
evaluate >= 0.4.0
jiwer >= 3.0.0
schedulefree >= 1.4.0
librosa >= 0.10.0
soundfile >= 0.12.0
```

**Do NOT install:** `torchcodec` (not needed with datasets < 4.0)

---

## 15. References

### Papers
- [Moonshine v2 (arXiv:2602.12241)](https://arxiv.org/abs/2602.12241) — streaming encoder architecture
- [Flavors of Moonshine (arXiv:2509.02523)](https://arxiv.org/abs/2509.02523) — monolingual > multilingual for small models
- [Original Moonshine (arXiv:2410.15608)](https://arxiv.org/abs/2410.15608) — base architecture, training recommendations

### Repos & Models
- [Moonshine GitHub](https://github.com/moonshine-ai/moonshine) — main repo, native library
- [German Support Issue #141](https://github.com/moonshine-ai/moonshine/issues/141) — community tracking
- [finetune-moonshine-asr (Pierre)](https://github.com/pierre-cheneau/finetune-moonshine-asr) — base toolkit
- [moonshine-tiny-fr (Pierre's French model)](https://huggingface.co/Cornebidouil/moonshine-tiny-fr) — reference for model card
- [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) — base model

### Datasets
- [MLS German](https://huggingface.co/datasets/facebook/multilingual_librispeech) — our training data
- [fosple's German dataset list (issue #141)](https://github.com/moonshine-ai/moonshine/issues/141) — comprehensive collection
- [Mozilla Data Collective](https://datacollective.mozillafoundation.org) — Common Voice (no longer on HF)

### Our Resources
- [ziggie.is storage setup guide](~/Documents/Projects/ziggie_setup_assistance/guides/01_STORAGE_SETUP.md)
- TensorBoard: `http://192.168.178.160:6006`
