# Moonshine German Fine-Tuning: Status

**Date:** 2026-03-30
**Status:** PAUSED — waiting for community German model

---

## Decision

Pausing active training. fosple (moonshine-ai member) achieved **13.16% WER** on German
using moonshine-streaming-small (123M params, v2 architecture) — see issue #141.
Waiting for them to share the model/config before investing more compute time.

## What we learned (Phase 1 research, completed)

- **Tokenizer:** German needs 2.08 tokens/word vs English 1.11 (1.62x overhead).
  Not the primary bottleneck — DE/FR ratio is only 1.13x.
- **Architecture:** moonshine-tiny (27M) plateaued at 35% WER after 6 epochs/44k steps.
  Model capacity is the bottleneck, not training duration.
- **v2 streaming-small (123M):** Proven path to 13% WER on German.
  Requires transformers >= 5.0 (incompatible with current env).
- **v1 base (61M):** Never tested. Configs ready (`german_base_60k.yaml`, `german_base_100k.yaml`).

## Ziggie state (clean)

- `/storage` — 1.7TB free (dataset deleted)
- `/data` — 1.2TB free (old results deleted)
- No active training runs
- Repo at `~/Projects/finetune-moonshine-asr/` is current

## To resume: full dataset recreation steps

### How storage works for this project

```
/storage/datasets/german_combined/     — decoded MLS German dataset (~1.1TB for full 470k)
/storage/datasets/german_60k/          — 60k stratified subset (~140GB est.)
/data/results-moonshine-de-base-60k/   — training output (model, checkpoints)
/data/results-moonshine-de-base-60k_encoded/  — tokenized dataset cache (~50-80GB for 60k)
```

`train.py` line 414 auto-creates the encoded cache at `{output_dir}_encoded` — always on `/data/`.
But `.map()` writes **temporary** Arrow files next to the source dataset during processing.
This is what filled `/storage` last time: 1.1TB dataset + growing `.map() temp cache > 1.8TB.
**Fix: use the 60k subset (~140GB + ~80GB temp = ~220GB — fits easily on /storage).**

Previous HF cache symlink (now deleted):
`~/.cache/huggingface/datasets/` → `/storage/datasets/huggingface_cache/`
Recreate if needed for HF download caching.

### Step 1: Recreate the full MLS German dataset (~2-3 hours, ~1.1TB on /storage)

```bash
cd ~/Projects/finetune-moonshine-asr

uv run python scripts/prepare_german_dataset.py \
    --output /storage/datasets/german_combined \
    --skip-cv
```

This downloads MLS German from HuggingFace (~30GB compressed), decodes to float32
Arrow arrays (~1.1TB), filters to 4-20s duration, saves as DatasetDict.

### Step 2: Create 60k subset (~5 min, ~140GB on /storage)

```bash
uv run python scripts/create_subset.py \
    --source /storage/datasets/german_combined \
    --output /storage/datasets/german_60k \
    --size 60000 \
    --stratify-by-duration
```

### Step 3: Delete the full dataset to free space (optional but recommended)

```bash
rm -rf /storage/datasets/german_combined    # frees ~1.1TB
```

The 60k subset is self-contained. You don't need the full dataset for training.

### Step 4: Update config to point at subset

Edit `configs/german_base_60k.yaml` — change:
```yaml
dataset:
  type: "local"
  path: "/storage/datasets/german_60k"    # NOT german_combined
```

### Step 5: Train

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/german_base_60k.yaml' \
    > /data/logs/moonshine-de-base-60k-run.log 2>&1 &

echo $last_pid > /data/logs/moonshine-de-base-60k-run.pid
echo "PID: $(cat /data/logs/moonshine-de-base-60k-run.pid)"
```

### Step 6: Verify

```bash
sleep 30
tail -30 /data/logs/moonshine-de-base-60k-run.log
```

### Step 7: Monitor

```bash
# WER checks
grep "eval_wer" /data/logs/moonshine-de-base-60k-run.log

# TensorBoard
uv run tensorboard --logdir /data/logs/moonshine-de-base-60k --host 0.0.0.0 --port 6006
```

### Space budget (60k subset path)

| What | Where | Size |
|------|-------|------|
| 60k subset | /storage/datasets/german_60k/ | ~140 GB |
| .map() temp cache (during preprocessing) | /storage (next to dataset) | ~80 GB |
| Encoded/tokenized cache | /data/results-*_encoded/ | ~50-80 GB |
| Training checkpoints | /data/results-*/ | ~500 MB each |
| HF model cache | ~/.cache/huggingface/hub/ (on /) | ~49 GB (already there) |
| **Total /storage** | | **~220 GB of 1.7TB** ✅ |
| **Total /data** | | **~80 GB of 1.2TB** ✅ |

---

## Why the base 60k run crashed (2026-03-30)

Config pointed at `/storage/datasets/german_combined` (the full 470k dataset, 1.1TB).
`.map()` preprocessing writes temporary Arrow cache next to the source dataset.
1.1TB dataset + growing cache > 1.8TB `/storage` drive → `OSError: No space left on device`
at 70% through preprocessing.

**The config must point at the 60k subset, not the full dataset.**

---

## Assets preserved

- `scripts/prepare_german_dataset.py` — downloads + filters MLS German
- `scripts/create_subset.py` — creates stratified subsets (60k, 100k, etc.)
- `scripts/tokenizer_analysis.py` — pure Python tokenizer analysis tool
- `contexts/tokenizer_analysis_and_architecture.md` — full analysis
- `contexts/issue141_findings_and_streaming_model.md` — fosple's result, dataset list, v2 details
- `configs/german_base_60k.yaml` / `german_base_100k.yaml` — ready to use
- `contexts/moonshine_de_context.md` — all gotchas from v1 tiny training

## Links

- Issue #141: https://github.com/moonshine-ai/moonshine/issues/141
- Our HF model (tiny, 35% WER): https://huggingface.co/dattazigzag/moonshine-tiny-de
- Pierre's French model: https://huggingface.co/Cornebidouil/moonshine-tiny-fr
