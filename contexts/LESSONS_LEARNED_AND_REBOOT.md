# Moonshine German Fine-Tuning: Lessons Learned & First-Principles Reboot

**Date:** 2026-03-29 (updated 2026-03-30)
**Author:** Saurabh Datta + Claude
**Status:** Completed 3 training runs, plateaued at ~35% WER. Resetting approach.

---

## What We Did

Fine-tuned `UsefulSensors/moonshine-tiny` (27M params, English-only) for German using Pierre Chéneau's `finetune-moonshine-asr` toolkit. Three training runs on RTX 5090:

| Run | Steps | Epochs | Time | Final WER | Notes |
|-----|:-----:|:------:|:----:|:---------:|-------|
| v1 | 10,000 | 1.36 | 9.7h | 36.7% | Loss still dropping — undertrained |
| v2 (resume) | 12,700 | ~1.7 new | 13.6h | 35.1% | Cancelled early to start best run |
| best | 44,000 | 5.99 | 40.2h | **34.99%** | Plateaued at epoch 4, overfitting after |

**Pierre's French reference:** 60k samples, 8k steps (~8.5 epochs) → **21.8% WER**

---

## Why We Plateaued at 35%

The WER curve tells the story clearly:

```
Epoch 0.3 → 42.7%  (rapid learning)
Epoch 1.1 → 37.3%  (slowing down)
Epoch 2.2 → 35.6%  (diminishing returns)
Epoch 4.1 → 34.99% (best — plateau reached)
Epoch 5.2 → 35.1%  (overfitting — eval loss rising, WER going back up)
```

**More training does not help.** We ran 6 full epochs (44k steps, 40 hours) and the model stopped improving at epoch 4. The architecture or data is the bottleneck, not training duration.

---

## Root Cause Analysis: Why Pierre Got 21.8% and We Got 35%

### 1. Tokenizer mismatch (LIKELY THE BIGGEST FACTOR)

Moonshine's tokenizer was trained on **English text**. German has:
- **Compound words:** "Geschwindigkeitsbeschränkung" (speed limit) — a single word the English tokenizer will shred into many subword tokens
- **Umlauts:** ä, ö, ü, ß — may not have efficient single-token representations
- **Different word frequency distribution** — common German function words may be split inefficiently

**Impact:** If the tokenizer needs 3-4 tokens for what should be 1, the decoder has to predict 3-4x more tokens per word, compounding errors. French is much closer to English in tokenization (Latin alphabet, similar word lengths, shared vocabulary roots).

**TO INVESTIGATE:** Compare tokenizer efficiency — how many tokens does the Moonshine tokenizer produce per German word vs English word vs French word? If German is significantly worse, this explains the WER gap.

### 2. Model capacity: moonshine-tiny (27M) may be too small for German

We only tried `moonshine-tiny` (27M params) because Pierre used it for French. But German may need more capacity:

- **moonshine-tiny:** 27M params — worked for French (simpler morphology, closer to English)
- **moonshine-base:** 61M params — 2.3× more parameters, more decoder capacity for longer token sequences

**Why this matters for German specifically:**
- German compound words produce longer token sequences (see tokenizer issue above). The decoder needs to generate more tokens per word, requiring more sequential prediction steps. A larger decoder with more layers/attention heads can maintain coherence over longer output sequences.
- German has a larger effective vocabulary than French. More embedding parameters help represent more distinct subword patterns.
- The Moonshine "Flavors" paper (arXiv:2509.02523) showed that monolingual tiny models outperform multilingual models — but this was for English. German's complexity may push beyond what 27M params can handle.
- The overfitting pattern we saw (train loss dropping but eval WER stalling) is a classic sign of **capacity bottleneck** — the model memorizes training patterns but can't generalize because it lacks the representational power.

**TO INVESTIGATE:**
- Fine-tune `moonshine-base` on the same data and compare WER directly
- Check if base model's loss curve shows continued improvement past epoch 4 (where tiny plateaued)
- Compare tokens-per-second inference speed — base is slower but may be acceptable for the use case
- Check VRAM: base should still fit on RTX 5090 (32 GB) without gradient checkpointing

**Key detail:** `UsefulSensors/moonshine-base` is available on HuggingFace. Same architecture, same tokenizer, just more layers. The training pipeline (`train.py`) should work with zero changes — just swap `model.name` in the config.

### 3. Dataset size mismatch

Pierre used **60k samples** (carefully sized for moonshine-tiny's 27M params). We used **470k samples** — nearly 8x more. This seems like it should help, but:
- More data means more diversity, more edge cases, more speaker variation
- A 27M parameter model may not have capacity to learn 470k samples well
- Pierre's model could "memorize" patterns in 60k samples across 8.5 epochs; ours can't with 470k
- The sweet spot for moonshine-tiny might be 50k–100k high-quality samples
- **moonshine-base (61M) might handle 470k better** — more capacity means it can actually leverage the extra data instead of being overwhelmed by it

**TO INVESTIGATE:** Train both tiny and base on 60k, 100k, and full 470k subsets. Create a 2×3 experiment grid:

| Model | 60k samples | 100k samples | 470k samples |
|-------|:-----------:|:------------:|:------------:|
| tiny (27M) | ? | ? | 34.99% (done) |
| base (61M) | ? | ? | ? |

This isolates whether the issue is model size, data size, or both.

### 4. German is harder than French for English-pretrained models

- German has more complex morphology (case endings, compound words, separable verbs)
- Larger vocabulary needed for the same coverage
- Phonetic mapping from German to English pretrained features may be worse than French → English
- This compounds with the capacity issue — a harder language needs either a better tokenizer or a bigger model (or both)

### 5. We copied Pierre's approach without questioning it

Pierre's toolkit was designed for French on an RTX 2060. We adapted configs for German on RTX 5090 but didn't question:
- Whether 470k samples is appropriate for 27M params
- **Whether moonshine-tiny is the right model size for German** (Pierre never tried base for French)
- Whether the tokenizer handles German well
- Whether the training hyperparameters (LR 3e-4, batch 64) are optimal for German
- Whether the Moonshine paper's recommendations for new-language fine-tuning differ from what Pierre did
- Whether curriculum learning specifically addresses German's compound word problem

---

## What Worked

- **The training pipeline works.** Data prep, training, inference, evaluation — all solid.
- **Dependency pinning.** `datasets < 4.0`, `transformers < 4.50` — essential, well-documented.
- **Gotcha documentation.** `gradient_checkpointing` broken, dual GPU broken, bf16 config patching — all captured.
- **RTX 5090 config.** batch 16, bf16, no gradient checkpointing — proven stable.
- **The model does learn German.** Even at 35% WER, short audiobook phrases are often correct or nearly correct.
- **Live inference works.** VAD fix (512 samples) confirmed, live mode functional on Mac CPU.

## What Didn't Work

- **Only trying moonshine-tiny** — never tested whether base (61M) breaks through the 35% ceiling
- **Scaling data from 60k → 470k without adjusting approach** — diminishing returns
- **Assuming more epochs = better WER** — plateaued at epoch 4
- **Copying Pierre's hyperparameters directly** — may not be optimal for German
- **No investigation of tokenizer efficiency** — likely the core bottleneck
- **No curriculum learning** — but also unclear if it would fix the fundamental issues
- **optimum ONNX export** — incompatible with both torch 2.11 (1.x) and removed in 2.x

---

## First-Principles Questions for the Reboot

### Architecture & Model Size
1. **Does moonshine-base (61M) break through the 35% WER ceiling?** This is the single fastest experiment to run — same pipeline, just change `model.name`. If base hits <30%, the answer is clear: tiny was too small for German.
2. **How does the Moonshine tokenizer handle German?** Measure tokens-per-word for German vs English vs French. If German is 2x+ worse, the tokenizer needs adaptation regardless of model size.
3. **Should we extend/retrain the tokenizer for German?** The Moonshine paper may have guidance.
4. **What does the Moonshine architecture actually expect from fine-tuning?** Read the source code, not just Pierre's interpretation.

### Data Strategy
5. **What's the optimal dataset size for each model size?** The experiment grid (tiny vs base × 60k vs 100k vs 470k) answers this definitively.
6. **Should we curate a high-quality subset** rather than using the full 470k MLS dump?
7. **What data sources exist beyond MLS?** SWC, VoxPopuli, Bundestag, M-AILABS — different speaking styles.
8. **Should we mix audiobook + conversational data** for better real-world performance?

### Training Strategy
9. **What does the Moonshine paper recommend for new-language adaptation?** Read arXiv:2509.02523 ("Flavors of Moonshine") carefully — it discusses monolingual fine-tuning.
10. **What learning rate, batch size, and schedule work best?** Don't assume Pierre's values are optimal. Base model may need different hyperparameters than tiny.
11. **Does curriculum learning help for German specifically?** German compound words might benefit from progressive difficulty.
12. **Should we freeze the encoder and only train the decoder?** Or vice versa? Or staged unfreezing? This may differ between tiny and base.

### Community & Existing Work
13. **What has the moonshine#141 community discovered?** Check latest comments.
14. **Has anyone else fine-tuned Moonshine for German (or any other non-French/English language)?** Search HuggingFace and GitHub.
15. **What do other small ASR models (Whisper tiny, Whisper base, etc.) achieve on German?** Baseline comparison — if Whisper tiny also struggles at ~35% on German, then the problem is fundamental to 27M-param models, not Moonshine-specific.

---

## Assets Preserved

### On ziggie
- `/storage/datasets/german_combined/` — 1.1 TB decoded MLS German (train: 469,942, test: 3,394)
- `~/Projects/finetune-moonshine-asr/` — repo with scripts, configs, and all gotchas

### On HuggingFace
- `dattazigzag/moonshine-tiny-de` — 35% WER model (update or delete after reboot)

### On GitHub
- `zigzagGmbH/finetune-moonshine-asr` — fork with German pipeline, bf16 patches, dependency docs

### Key Documents
- `contexts/moonshine_de_context.md` — full technical context (server, deps, gotchas)
- `contexts/REALWORLD_TEST_RESULTS.md` — live mic + file test findings
- This file — lessons learned

---

## Recommended Next Steps

### Phase 1: Research (before any training)

1. **Read the Moonshine papers thoroughly:**
   - arXiv:2410.15608 — base architecture, training recommendations
   - arXiv:2509.02523 — "Flavors of Moonshine", monolingual fine-tuning guidance
   - arXiv:2602.12241 — streaming encoder (v2)

2. **Read the Moonshine source code:**
   - Tokenizer implementation — how it handles non-English text
   - Model architecture — what's shared between encoder/decoder, differences between tiny and base
   - Training scripts in the official repo (not Pierre's fork)

3. **Analyze tokenizer efficiency:**
   ```python
   from transformers import AutoProcessor
   processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
   
   # Compare tokenization
   de = processor.tokenizer("Geschwindigkeitsbeschränkung ist wichtig")
   en = processor.tokenizer("Speed limit is important")
   fr = processor.tokenizer("La limitation de vitesse est importante")
   print(f"German: {len(de.input_ids)} tokens")
   print(f"English: {len(en.input_ids)} tokens")
   print(f"French: {len(fr.input_ids)} tokens")
   ```

4. **Check moonshine#141** for latest community findings

5. **Benchmark against Whisper tiny AND Whisper base on German MLS test set** — establishes the realistic ceiling for small models on this exact data

### Phase 2: Experiment (small, fast, iterative)

**Priority order — fastest/highest-impact first:**

6. **Try moonshine-base (61M) on 60k subset** — quick test (~10h), answers the biggest question: is model size the bottleneck?
7. **Try moonshine-tiny on 60k subset** — match Pierre's data size, isolate language vs data size
8. **Compare tiny vs base results** — if base is significantly better, focus all subsequent work on base
9. **Try encoder-frozen training** — decoder-only adaptation (faster training, less overfitting risk)
10. **Try tokenizer extension** — add German-specific tokens (if tokenizer analysis from Phase 1 confirms the problem)

### Phase 3: Scale the winner

11. Whatever wins in Phase 2, scale it with more data/steps
12. Upload best model to HF
13. Post comprehensive results to moonshine#141
14. Request ONNX/ORT conversion guidance from Pete Warden

---

## Dependency & Infra Reference (still valid)

```
datasets >= 2.14.0, < 4.0.0    # 4.x breaks audio decoding (torchcodec)
transformers >= 4.35.0, < 4.50.0  # 4.50+ removes training params
torch >= 2.0.0                  # tested with 2.11.0+cu130
CUDA_VISIBLE_DEVICES=0          # dual GPU breaks Moonshine KV cache
gradient_checkpointing: false   # broken with Moonshine in transformers 4.49
bf16: true                      # RTX 5090 native
optimum ONNX: BLOCKED           # 1.x incompatible with torch 2.11, 2.x dropped exporter
```

**uv workflow:** ziggie leads `uv add`/`uv lock`, Mac only does `git pull` + `uv sync`.

**Model HuggingFace IDs:**
- `UsefulSensors/moonshine-tiny` — 27M params (what we trained on)
- `UsefulSensors/moonshine-base` — 61M params (TO TRY NEXT - maybe if found necessary)
