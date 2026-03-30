# Moonshine German: Reboot Strategy (First Principles)

**Date:** 2026-03-30
**Status:** Ready to execute

---

## The Problem

moonshine-tiny (27M) plateaued at 35% WER on German after 6 epochs (44k steps) on 470k MLS samples. Pierre's French reference: 21.8% WER with the same model on 60k samples.

## What We Know

| Finding | Source |
|---------|--------|
| tiny (27M) → 35% WER on 470k samples | Our training runs |
| French tiny → 21.8% WER on 60k samples | Pierre's result |
| streaming-small (123M) → 13% WER on German | fosple in issue #141 |
| German needs 2.08 tokens/word vs 1.11 English, 1.64 French | Our tokenizer analysis |
| Tokenizer has good German coverage (ä:117, ö:96, ü:96 tokens) | Our tokenizer analysis |
| Overfitting pattern: train loss dropped, eval WER stalled at epoch 4 | Our TensorBoard |
| `max_new_tokens: 80` clips German output (125 tokens needed for 20s audio) | Our tokenizer analysis |
| v1 base (61M) was never tested | — |
| v1 tiny on 60k subset was never tested | — |

## Why We Plateaued: Root Causes (ranked by likelihood)

1. **Model too small for German** — 27M params generating 2.08 tokens/word can't maintain coherence across compound words. The overfitting pattern (train loss drops, eval WER stalls) is a classic capacity bottleneck.

2. **Too much data for the model size** — 470k samples overwhelm 27M params. Pierre's 60k was a better match for tiny's capacity. The model can't learn the diversity in 470k with only 27M params.

3. **max_new_tokens clipping eval** — Set to 80, but German 20s audio needs ~125 tokens. This inflates measured WER by truncating longer predictions. Not a training issue, but it makes our results look worse than they are.

4. **Tokenizer overhead** — 1.62× more tokens than English per sentence. Not the primary bottleneck (DE/FR ratio is only 1.13×), but it amplifies the capacity problem.

5. **Hyperparameters never tuned** — Pierre's LR/batch/schedule were used verbatim. May not be optimal for German or for a different model size.

## The Three Candidate Models

| Model | Params | Env | Effort to try | VRAM (batch 16, bf16) |
|-------|:------:|-----|:-------------:|:---------------------:|
| v1 tiny | 27M | Current | Already done | ~8 GB |
| **v1 base** | **61M** | **Current** | **1 line change** | **~18-22 GB** |
| v2 streaming-small | 123M | transformers 5.x (NEW) | New env + code changes | ~28-35 GB |

**Why start with v1 base, not streaming-small:**
- Same training pipeline — zero infrastructure risk
- Isolates model size as a variable — answers "is 27M too small?"
- Fast experiment (~10 hours on 100k subset)
- If base breaks 28%, we avoid the entire transformers 5.x migration
- If base doesn't break 28%, we learn that v1 architecture is the ceiling, and v2 streaming is the only path (worth the investment)

**Why not streaming-small first:**
- Requires transformers >= 5.0 (breaking our pinned deps)
- All dependency gotchas (datasets, gradient_checkpointing, etc.) need re-evaluation
- 123M params → may not fit batch 16 on 32GB → slower iteration
- fosple hasn't shared training config — we'd be guessing
- We'd never know if v1 base could have been "good enough"

## The Experiment Plan

### Experiment A: v1 base on 100k subset (DO FIRST)

**Config:** `configs/german_base_100k.yaml` (written, ready to go)

| Parameter | Value | Why |
|-----------|-------|-----|
| Model | `UsefulSensors/moonshine-base` (61M) | 2.3× tiny, tests capacity hypothesis |
| Dataset | 100k subset of MLS German | Proportional to model capacity (Pierre: 60k/27M ≈ 2.2k/M; us: 100k/61M ≈ 1.6k/M) |
| Effective batch | 16 × 4 = 64 | Same as proven Pierre setup |
| Learning rate | 2e-4 | Lower than tiny's 3e-4 — larger models need gentler LR |
| Steps | 6,500 (~4 epochs) | Tiny plateaued at epoch 4; if base keeps improving, extend |
| max_new_tokens | **150** (was 80) | Fixes German truncation bug |
| length_penalty | 1.0 (was 1.2) | Neutral — don't bias length |
| Warmup | 200 steps | ~13% of first epoch |
| Eval every | 1000 steps | 6-7 data points to see the WER curve |

**Time estimate:** ~10 hours on RTX 5090
**Decision criteria:**
- WER < 28% → **SUCCESS.** Model size was the bottleneck. Scale to 470k, target <20%.
- WER 28-32% → **Promising.** More data or tuning might help. Try 470k and/or lower LR.
- WER > 32% → **v1 ceiling reached.** Pivot to streaming-small (Experiment C).

### Experiment B: v1 tiny on 60k subset (OPTIONAL, parallel)

Matches Pierre's exact data scale. Answers: "Would tiny have done better with less data?"
- If tiny on 60k → 25-28%: confirms too-much-data was the problem for tiny
- If tiny on 60k → 33%+: confirms 27M is genuinely too small for German

Run this only if ziggie GPU 1 is free and you want the data point.

### Experiment C: v2 streaming-small (ONLY IF A FAILS)

If base plateaus at 32%+:
1. Set up fresh uv project on ziggie with transformers >= 5.0
2. Adapt train.py: swap `MoonshineForConditionalGeneration` → `MoonshineStreamingForConditionalGeneration`
3. Re-evaluate all dependency constraints
4. Test inference first, then training loop
5. Ask fosple for their config (monitor issue #141)

## Key Config Changes from Previous Runs

| Parameter | Old (tiny 470k) | New (base 100k) | Reason |
|-----------|:----------------:|:----------------:|--------|
| model.name | moonshine-tiny | **moonshine-base** | Test capacity hypothesis |
| dataset size | 470,000 | **100,000** | Proportional to model size |
| learning_rate | 3e-4 | **2e-4** | Larger model → lower LR |
| max_steps | 44,000 | **6,500** | 4 epochs on 100k (fast iteration) |
| max_new_tokens | 80 | **150** | **BUG FIX** — was clipping German output |
| length_penalty | 1.2 | **1.0** | Neutral — no length bias |
| eval_batch_size | 16 | **8** | Base uses more VRAM during generate() |
| warmup_steps | 300 | **200** | Proportional to shorter training |
| eval_steps | 2000 | **1000** | More data points in shorter run |

## Before Training Checklist (ziggie)

```bash
# 1. Create 100k subset
cd ~/Projects/finetune-moonshine-asr
CUDA_VISIBLE_DEVICES=0 uv run python scripts/create_subset.py \
    --source /storage/datasets/german_combined \
    --output /storage/datasets/german_100k \
    --size 100000 --stratify-by-duration

# 2. Verify subset
uv run python -c "from datasets import load_from_disk; d = load_from_disk('/storage/datasets/german_100k'); print(f'Train: {len(d[\"train\"]):,}, Test: {len(d[\"test\"]):,}')"

# 3. Start training
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
    --config configs/german_base_100k.yaml --no-curriculum

# 4. Monitor
# In another terminal:
# tensorboard --logdir /data/logs/moonshine-de-base-100k
```

## What Success Looks Like

- **Minimum viable:** base on 100k → WER < 28% → proves size matters, path to <20% is clear
- **Stretch goal:** base on 470k → WER < 22% → competitive with fosple's streaming result for most use cases
- **Ideal endgame:** streaming-small with mixed data → WER < 15% → best-in-class German edge ASR
