# Moonshine German (DE) Fine-Tuning — Session Report

**Date:** 2026-03-25 / 2026-03-26
**Author:** Saurabh Datta (dattazigzag) + Claude (Anthropic)
**Server:** ziggie.is — dual RTX 5090 (32 GB VRAM each), Ryzen Threadripper PRO (48 cores), 125 GB RAM
**Goal:** Fine-tune Moonshine Tiny (27M params) for German ASR, following the community approach from [finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)

---

## 1. Context & Motivation

Moonshine v2 is a fast, edge-optimized ASR model by Moonshine AI (formerly Useful Sensors). It currently supports English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, and Arabic — but **not German**.

GitHub issue [#141](https://github.com/moonshine-ai/moonshine/issues/141) is a community tracking thread for German support. Pete Warden (Moonshine contributor) has expressed interest. A community member (fosple) compiled an extensive list of German speech datasets. Another member (RobFlo98) flagged a critical German-specific concern: German subtitle data often paraphrases rather than transcribing verbatim, and boilerplate like "Untertitelung. BR" can leak in.

Pierre Chéneau published [finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr), a complete fine-tuning toolkit that achieved 21.8% WER on French using MLS French (~60k samples) on an RTX 2060 (6 GB VRAM). His approach: fine-tune the HuggingFace safetensors version of `UsefulSensors/moonshine-tiny` using HuggingFace `Trainer` with schedule-free AdamW.

Our plan: replicate Pierre's pipeline for German, using ziggie's significantly more powerful hardware (RTX 5090 32GB vs RTX 2060 6GB).

---

## 2. What We Did

### 2.1 Repository Setup (on ziggie)

```bash
git clone git@github.com:pierre-cheneau/finetune-moonshine-asr.git
cd finetune-moonshine-asr
# Created new branch for DE work
uv init --no-package -p 3.12
uv add torch torchaudio transformers datasets accelerate evaluate jiwer \
     tensorboard schedulefree librosa soundfile pyyaml tqdm numpy pandas \
     onnx onnxruntime optimum sounddevice gradio
uv add --dev pytest black isort
```

### 2.2 Research Phase

We analyzed three primary sources:

1. **Moonshine README** — architecture, supported languages, available models, the "Flavors of Moonshine" paper showing mono-lingual models outperform multilingual for small (27M) models
2. **Pierre's finetune-moonshine-asr** — `train.py`, `data_loader.py`, `curriculum.py`, training config for French
3. **GitHub issue #141** — dataset recommendations (fosple's table: MLS ~1100h, Common Voice ~370h, Bundestag ~600h, SWC ~386h, Emilia-YODAS ~5600h), German subtitle contamination warning (RobFlo98)

### 2.3 Dataset Choice

We chose **MLS German** (`facebook/multilingual_librispeech`, `"german"`) as the primary dataset because:
- Pierre's `data_loader.py` already has native `load_mls()` support
- ~1100 hours of clean read-speech audiobooks with verified transcripts
- No subtitle contamination risk (per RobFlo98's warning)
- CC BY 4.0 license

We also planned to add **Common Voice German** for speaker diversity, but discovered that as of October 2025, Mozilla pulled all Common Voice datasets from HuggingFace — they are now only available through Mozilla Data Collective. We proceeded with MLS-only.

### 2.4 Data Preparation Script

We wrote `scripts/prepare_german_dataset.py` to:
- Download MLS German from HuggingFace
- Compute audio durations for all samples
- Filter to [4.0s, 20.0s] per Moonshine paper recommendations
- Save as a local `DatasetDict` for training

### 2.5 Training Configuration

We wrote `configs/mls_cv_german_no_curriculum.yaml` adapted from Pierre's French config:
- `bf16: true` instead of `fp16: true` (RTX 5090 has native bf16)
- `per_device_train_batch_size: 16` instead of 4 (32 GB vs 6 GB VRAM)
- `gradient_accumulation_steps: 4` (effective batch size 64, same as Pierre)
- `max_steps: 10000` (~1.5 epochs over 470k samples)
- `optim: "schedule_free_adamw"` (same as Pierre)
- `gradient_checkpointing: true` → later changed to `false` (see issues below)
- No curriculum learning for first run

### 2.6 Storage Layout

ziggie has three mount points:
- `/` (root): 295 GB — OS, Docker images, home directories
- `/data`: 1.5 TB — AI models, service databases, Docker volumes
- `/storage`: 1.8 TB — bulk storage, datasets, outputs

HuggingFace cache was initially on `/` (`~/.cache/huggingface/`). MLS German parquet downloads (~30 GB) + Arrow cache expansion (~120 GB) quickly filled root. We:
1. Moved HF datasets cache to `/storage` via symlink
2. Pointed prepared dataset output to `/storage/datasets/german_combined`
3. Pointed training outputs (checkpoints, encoded cache) to `/data/`

---

## 3. Issues Encountered (Chronological)

### 3.1 `trust_remote_code` Deprecation Warning
**What:** `datasets >= 4.0.0` dropped `trust_remote_code` parameter. Our script included it.
**Fix:** Removed the parameter with `sed -i '/trust_remote_code=True,/d' scripts/prepare_german_dataset.py`
**Impact:** Warning only, not a blocker.

### 3.2 Disk Full — Root Partition (`/`)
**What:** HuggingFace downloads + Arrow cache expansion filled the 295 GB root partition. MLS German raw data is ~30 GB compressed but expands to ~120+ GB as decoded Arrow arrays.
**Fix:** Symlinked `~/.cache/huggingface/datasets/` to `/storage/datasets/huggingface_cache/`
**Impact:** Lost one full data prep run. Had to restart.

### 3.3 Common Voice Pulled from HuggingFace
**What:** `EmptyDatasetError` when loading `mozilla-foundation/common_voice_17_0`. Mozilla moved all CV datasets to Mozilla Data Collective as of October 2025.
**Fix:** Added `--skip-cv` flag, proceeded with MLS-only.
**Impact:** Lost speaker diversity. MLS is clean audiobook speech only (single recording style).

### 3.4 `torchcodec` Required by `datasets >= 4.0.0`
**What:** New `datasets` 4.x changed audio decoding backend from `soundfile`/`librosa` to `torchcodec`. Installing `torchcodec` then failed because it required FFmpeg system libraries + had CUDA lib mismatch (`libnppicc.so.13`).
**Fix:** Pinned `datasets < 4.0.0` — Pierre's entire codebase was written for datasets 3.x.
**Impact:** Clean fix, no side effects.

### 3.5 Slow Duration Computation (~2+ hours)
**What:** Computing durations for 470k audio samples required decoding each compressed audio file. CPU-bound.
**Fix:** Increased `num_proc` from 4 to 8 (16 caused OOM with 125 GB RAM). Still took ~2.1 hours.
**Impact:** Time cost only. Final result: 469,942 train samples, 1,966.5 hours, avg 15.1s per clip.

### 3.6 Disk Full — `/storage` Partition
**What:** After multiple failed runs, accumulated data filled `/storage` (1.8 TB):
  - `german_combined`: 1.1 TB (decoded audio as float32 Arrow arrays)
  - `results-moonshine-de_encoded`: 426 GB (tokenized training cache from failed run)
  - `huggingface_cache`: 290 GB (raw downloads)
**Fix:** Deleted encoded cache and HF cache (~716 GB freed). Redirected training outputs to `/data/` (1.2 TB free).
**Impact:** Multiple hours lost to re-running data prep.

### 3.7 `KeyError: 'fp16'` in train.py
**What:** Pierre's `train.py` hardcodes `fp16=train_config['fp16']`, but our config uses `bf16` keys.
**Fix:** Added both keys to config (`fp16: false`, `bf16: true`) and patched `train.py` with `.get()` defaults for both fp16 and bf16 parameters.
**Impact:** Quick fix.

### 3.8 `transformers` 5.3.0 Incompatibility
**What:** We initially installed `transformers >= 5.3.0` which removed/renamed multiple `Seq2SeqTrainingArguments` parameters (e.g., `group_by_length`). Pierre's code was written for `transformers ~4.35-4.49`.
**Fix:** Pinned `transformers >= 4.35.0, < 4.50.0`. Resolved to 4.49.0.
**Impact:** Required re-running preprocessing (cache invalidated by version change).

### 3.9 `DataParallel` Crash — Dual GPU Auto-detection
**What:** PyTorch auto-detected both RTX 5090s and wrapped the model in `DataParallel`, which broke Moonshine's KV cache handling: `AttributeError: 'bool' object has no attribute 'is_updated'`
**Fix:** `CUDA_VISIBLE_DEVICES=0` to force single-GPU training.
**Impact:** Quick fix.

### 3.10 `gradient_checkpointing` Crash ← CURRENT BLOCKER
**What:** Even with single GPU, the same `is_updated` error occurs during gradient checkpointing. The Moonshine model's attention code (line 235 in `modeling_moonshine.py`) expects `past_key_value.is_updated` to be a dict-like object, but gradient checkpointing passes it as a plain `bool`.
**Error trace:**
```
File ".../modeling_moonshine.py", line 235, in forward
    is_updated = past_key_value.is_updated.get(self.layer_idx)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'bool' object has no attribute 'is_updated'
```
**Fix applied:** `gradient_checkpointing: false` in config. Currently re-running.
**Impact:** Not a functional concern — 27M param model fits trivially in 33.7 GB VRAM without checkpointing.

---

## 4. Current State (as of writing)

### 4.1 What's Running

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config configs/mls_cv_german_no_curriculum.yaml
```

With `gradient_checkpointing: false`. The encoded dataset cache (from the last successful `.map()` + save to `/data/`) should be reused. Currently saving 907 shards to `/data/results-moonshine-de_encoded`.

### 4.2 Disk State

| Mount | Size | Used | Free |
|-------|------|------|------|
| `/` (root) | 295 GB | 242 GB | 40 GB |
| `/data` | 1.5 TB | ~420 GB | ~1.0 TB |
| `/storage` | 1.8 TB | 1.4 TB | ~290 GB |

### 4.3 Key File Locations

| What | Path |
|------|------|
| Repository | `~/Projects/finetune-moonshine-asr/` |
| Prepared dataset (1.1 TB) | `/storage/datasets/german_combined/` |
| Encoded/tokenized cache | `/data/results-moonshine-de_encoded/` |
| Training checkpoints (pending) | `/data/results-moonshine-de/` |
| TensorBoard logs (pending) | `/data/logs/moonshine-de-no-curriculum/` |
| HF model cache | `~/.cache/huggingface/hub/` (49 GB, on `/`) |
| Training config | `configs/mls_cv_german_no_curriculum.yaml` |
| Data prep script | `scripts/prepare_german_dataset.py` |

### 4.4 Pinned Dependencies

```
datasets >= 2.14.0, < 4.0.0
transformers >= 4.35.0, < 4.50.0
torch (latest, 2.11.0+cu130)
```

---

## 5. What Hasn't Been Done Yet

- Actual GPU training (pending current run succeeding)
- WER evaluation on test set
- Inference testing with German audio
- ONNX export for use with native `moonshine-voice` CLI
- Curriculum learning (planned as follow-up if baseline works)
- Adding Common Voice data (requires manual download from Mozilla Data Collective)
- Multi-GPU training (deferred — single 5090 is sufficient for 27M param model)

---

## 6. Cleanup Plan (After Successful Training)

```bash
# Delete intermediate data (~1.8 TB freed)
rm -rf /storage/datasets/german_combined          # 1.1 TB
rm -rf /data/results-moonshine-de_encoded          # ~426 GB
rm -rf /data/results-moonshine-de/checkpoint-*/    # intermediate checkpoints
rm -rf ~/.cache/huggingface/hub/                   # 49 GB

# Keep
/data/results-moonshine-de/final/                  # ~200-500 MB trained model
/data/logs/moonshine-de-no-curriculum/             # TensorBoard logs
~/Projects/finetune-moonshine-asr/                 # repo + configs
```

---

## 7. Key Lessons

1. **MLS German expands massively** — 30 GB compressed → 1.1 TB as decoded float32 Arrow arrays. Plan storage accordingly.
2. **Pin your dependencies** — `datasets` 4.x and `transformers` 5.x have breaking changes vs. Pierre's codebase. Pin to `datasets < 4` and `transformers < 4.50`.
3. **HuggingFace caching is aggressive** — every `.map()` call creates a new cache. Failed runs accumulate hundreds of GB. Point `~/.cache/huggingface/` to bulk storage early.
4. **Common Voice is no longer on HuggingFace** — as of October 2025, Mozilla Data Collective is the only source.
5. **Gradient checkpointing is broken for Moonshine** in transformers 4.49 — disable it. The model is small enough (27M params) that it's not needed with modern GPUs.
6. **Force single GPU** with `CUDA_VISIBLE_DEVICES=0` — PyTorch's DataParallel wrapper breaks Moonshine's KV cache.
7. **German subtitle contamination** — avoid pseudo-labeled web data for ASR training. Stick to verified transcript sources (MLS, SWC, M-AILABS).

---

## 8. References

- [Moonshine v2 paper (arXiv:2602.12241)](https://arxiv.org/abs/2602.12241)
- [Flavors of Moonshine paper (arXiv:2509.02523)](https://arxiv.org/abs/2509.02523)
- [Original Moonshine paper (arXiv:2410.15608)](https://arxiv.org/abs/2410.15608)
- [Moonshine GitHub](https://github.com/moonshine-ai/moonshine)
- [German Language Support Issue #141](https://github.com/moonshine-ai/moonshine/issues/141)
- [finetune-moonshine-asr (Pierre Chéneau)](https://github.com/pierre-cheneau/finetune-moonshine-asr)
- [moonshine-tiny-fr HuggingFace model card](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)
- [MLS dataset](https://huggingface.co/datasets/facebook/multilingual_librispeech)
- [ziggie.is storage setup guide](~/Documents/Projects/ziggie_setup_assistance/guides/01_STORAGE_SETUP.md)
