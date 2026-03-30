# Moonshine Tokenizer Analysis & Architecture Comparison

**Date:** 2026-03-30
**Purpose:** Phase 1 research for German fine-tuning reboot (see `moonshine_de_lessons_learned.md`)

---

## 1. Tokenizer Architecture

**Type:** Greedy longest-match byte tokenizer (SentencePiece convention)
**Source:** `core/bin-tokenizer/bin-tokenizer.cpp` in the moonshine repo
**Vocab size:** 32,768 tokens (32k), 1 special/empty token, 32,767 real tokens

**How it works:**
1. Replace all spaces with `▁` (U+2581 — SentencePiece word-boundary marker)
2. Convert to UTF-8 bytes
3. At each position, find the longest vocabulary entry that matches
4. Emit that token ID, advance by the matched byte length
5. Repeat until all bytes consumed

This is a flat vocabulary lookup — no BPE merge rules, no fallback to byte-level. If a byte sequence isn't in the vocabulary, encoding fails. The `tokenizer.bin` binary format stores each token as a length-prefixed byte sequence.

---

## 2. German Character Coverage

**Verdict: Good.** The tokenizer has broad coverage of German characters.

| Character | UTF-8 | Tokens containing it | Word-start (▁+char) tokens |
|-----------|-------|:--------------------:|:--------------------------:|
| ä | c3a4 | 117 | 4 |
| ö | c3b6 | 96 | 6 |
| ü | c3bc | 96 | 3 |
| ß | c39f | 32 | 0 |
| Ä | c384 | 2 | 1 |
| Ö | c396 | 3 | 2 |
| Ü | c39c | 4 | 3 |

Notable tokens: `▁über`, `▁für`, `▁Straße`, `▁Bundes`, `gericht`, `schaft`, `lichkeit`, `▁größ`, `▁Fußball`, `▁Österreich`, `▁zurück`

The vocabulary clearly was trained on multilingual text (despite Moonshine being "English-only" for ASR). German subwords exist but are less frequent than English ones — the long tail of common German word-parts is thinner.

---

## 3. Tokenization Efficiency: German vs English vs French

### Per-sentence comparison (parallel translations)

| Sentence type | DE tokens | EN tokens | FR tokens | DE/EN | DE/FR |
|---------------|:---------:|:---------:|:---------:|:-----:|:-----:|
| Simple sentence | 9 | 5 | 6 | 1.80 | 1.50 |
| Compound word (speed limit) | 19 | 10 | 13 | 1.90 | 1.46 |
| Umlauts & eszett | 13 | 10 | 16 | 1.30 | 0.81 |
| Separable verb | 10 | 10 | 10 | 1.00 | 1.00 |
| Long compound (Bundesverfassungsgericht) | 11 | 7 | 11 | 1.57 | 1.00 |
| Everyday speech | 16 | 9 | 13 | 1.78 | 1.23 |
| Technical/formal | 15 | 8 | 13 | 1.88 | 1.15 |
| Numbers & units | 17 | 9 | 15 | 1.89 | 1.13 |
| **TOTAL** | **110** | **68** | **97** | **1.62** | **1.13** |

### Tokens per word

| Language | Tokens/word | Notes |
|----------|:-----------:|-------|
| English | 1.11 | Baseline — most words are single tokens |
| French | 1.64 | ~48% overhead vs English |
| German | 2.08 | ~87% overhead vs English, ~27% vs French |

### Compound word fragmentation

| German word | Tokens | Breakdown |
|-------------|:------:|-----------|
| Geschwindigkeitsbeschränkung | 9 | Ge·sch·wind·igkeit·sb·esch·rän·ku·ng |
| Bundesverfassungsgericht | 7 | Bu·nde·sv·er·fass·ungs·gericht |
| Straßenbahnhaltestelle | 7 | Str·a·ßen·bahn·halt·este·lle |
| Kraftfahrzeughaftpflichtversicherung | 10 | K·raft·fahr·zeug·haft·pf·licht·vers·icher·ung |
| Donaudampfschifffahrtsgesellschaft | 10 | Don·aud·ampf·sch·iff·fahrt·sg·ese·ll·schaft |

Shorter common words are handled well: `über` (1 token), `Straße` (3), `schön` (2), `Brötchen` (3).

---

## 4. Interpretation for ASR Performance

### The tokenizer is NOT the primary bottleneck

The DE/FR ratio of 1.13x cannot explain a 13-point WER gap (35% vs 21.8%). If the tokenizer were the main problem, we'd expect a much larger ratio.

### What the tokenizer DOES contribute

For every German word the decoder predicts, it must output ~2.08 tokens on average (vs 1.11 for English). This means:
- **More sequential prediction steps** where errors compound
- **Longer decoder sequences** for the same audio duration — the model runs out of `max_new_tokens` sooner (current setting: 80)
- **Compound words are high-risk** — a single mistake in a 7-10 token compound corrupts the entire word, adding several wrong words to WER

### The real bottleneck: model capacity + data strategy

The tokenizer inefficiency is a multiplier on the capacity problem:
- 27M params generating 2.08 tokens/word → capacity is stretched thin
- 61M params generating 2.08 tokens/word → more headroom to maintain coherence across longer sequences
- The overfitting pattern (train loss drops, eval WER stalls) confirms the decoder can't generalize at this token-generation rate with only 27M params

---

## 5. Architecture Comparison: Tiny vs Base

From `core/moonshine-model.cpp`:

| Attribute | Tiny | Base | Ratio |
|-----------|:----:|:----:|:-----:|
| Decoder layers | 6 | 8 | 1.33× |
| KV attention heads | 8 | 8 | 1.0× |
| Head dimension | 36 | 52 | 1.44× |
| Hidden dim (heads × head_dim) | 288 | 416 | 1.44× |
| KV cache per layer | 288 | 416 | 1.44× |
| Total parameters | 27M | 61M | 2.26× |

**What base buys for German:**
- **More decoder layers (6→8):** Each layer adds another attention step where the model can refine its prediction. For 7-10 token compound words, 2 extra layers of attention over the encoder hidden states helps maintain coherence.
- **Wider hidden dim (288→416):** More representational capacity per position. German's larger effective vocabulary and compound morphology need more dimensions to disambiguate.
- **Same tokenizer, same KV heads:** No additional inference complexity from the attention pattern — base just has "deeper" and "wider" processing at each step.

**Switching to base in the training pipeline:**
```bash
# In configs/german_best.yaml, change:
model:
  name: "UsefulSensors/moonshine-base"   # was: moonshine-tiny
```
Everything else works unchanged — `train.py` uses `MoonshineForConditionalGeneration.from_pretrained()` which auto-detects the architecture. The HuggingFace ID is `UsefulSensors/moonshine-base`.

**VRAM estimate for base:**
- Tiny at batch 16: ~8-10 GB VRAM on RTX 5090
- Base at batch 16: ~18-22 GB (estimated 2.3× scaling from param count)
- RTX 5090 has 32 GB → should fit without reducing batch size
- If tight, reduce to batch 12 with gradient_accumulation_steps: 6 (still effective batch 72)

---

## 6. Recommended Config for Base Experiment

A quick sanity test on 60k samples first (matches Pierre's data size):

```yaml
model:
  name: "UsefulSensors/moonshine-base"
  freeze_encoder: false

dataset:
  type: "local"
  path: "/storage/datasets/german_combined"   # TODO: create 60k subset
  text_column: "sentence"

training:
  output_dir: "/data/results-moonshine-de-base-60k"
  per_device_train_batch_size: 16             # Try 16 first; drop to 12 if OOM
  gradient_accumulation_steps: 4
  gradient_checkpointing: false
  bf16: true
  max_steps: 8000                             # ~8.5 epochs on 60k samples
  learning_rate: 3.0e-4                       # Start with same LR; may need tuning
  warmup_steps: 200
  eval_steps: 500
  save_steps: 500
  logging_steps: 50

generation:
  max_new_tokens: 100                         # Increase for German compound words
  num_beams: 5
```

**Estimated time:** ~8k steps × ~5-6s/step (base is slower) ≈ 11-13 hours on RTX 5090.

---

## 7. Open Questions

1. **Does `gradient_checkpointing: false` still apply to base?** Likely yes (same transformers version, same Moonshine architecture class), but verify on first run.
2. **Does `CUDA_VISIBLE_DEVICES=0` still apply?** Yes — same KV cache issue exists in both model sizes.
3. **Is 60k subset creation straightforward?** Need to subsample from the 470k MLS German dataset. A simple `dataset.select(range(60000))` or stratified sample by duration.
4. **Should `max_new_tokens` increase for base?** Probably — base has more capacity to generate longer sequences accurately. 100 is a safe starting point.
5. **Should the learning rate change for base?** Typically larger models benefit from slightly lower LR. Try 3e-4 first (same as tiny); if loss is unstable, drop to 1e-4.

---

## 8. Analysis Script

The tokenizer analysis script is saved at `scripts/tokenizer_analysis.py`. It parses `tokenizer.bin` directly (same format as C++ `BinTokenizer`), runs without any ML dependencies — pure Python. To run:

```bash
python scripts/tokenizer_analysis.py
```

Requires the tokenizer.bin path to be updated in the script (line: `TOKENIZER_PATH = ...`).
