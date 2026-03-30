# Moonshine German Reboot: Issue #141 Findings & Streaming Model Analysis

**Date:** 2026-03-30
**Source:** https://github.com/moonshine-ai/moonshine/issues/141
**Purpose:** New data that changes the reboot strategy

---

## 1. The Result That Changes Everything

**fosple** (Mar 26, 2026) — a moonshine-ai org member — achieved:
- **13.16% WER on FLEURS DE**
- **11.95% WER on MCV DE**
- Using **moonshine-streaming-small** (v2 architecture, 123M params)

For comparison, our best was 34.99% WER on moonshine-tiny (v1, 27M params).

fosple hasn't yet shared which datasets they used or whether the model is on HuggingFace. Questions were asked (by us, dattazigzag) on Mar 27–28 — still waiting for a reply.

---

## 2. moonshine-streaming-small Architecture

**HuggingFace ID:** `UsefulSensors/moonshine-streaming-small`
**Model class:** `MoonshineStreamingForConditionalGeneration` (NOT `MoonshineForConditionalGeneration`)
**Processor:** `AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-small")`

| Attribute | v1 tiny | v1 base | v2 streaming-small |
|-----------|:-------:|:-------:|:-----------------:|
| Total params | 27M | 61M | **123M** |
| Encoder layers | 6 | ? | **10** |
| Decoder layers | 6 | 8 | **10** |
| Encoder dim | 288 | ? | **620** |
| Decoder dim | 288 | 416 | **512** |
| Attention heads | 8 | 8 | **8** |
| Head dim | 36 | 52 | **64** |
| Vocab size | 32768 | 32768 | **32768** (same tokenizer!) |
| Encoder type | Full attention | Full attention | **Sliding window** |
| Trained on | ~200K hours | ~200K hours | **~300K hours** |

The v2 streaming architecture uses:
- Sliding-window self-attention in encoder (no positional embeddings — "ergodic")
- Adapter layer that adds positional embeddings before decoder
- RoPE in decoder (partial_rotary_factor: 0.5)
- SiLU activation in decoder, GELU in encoder
- Same BOS/EOS/decoder_start tokens as v1

---

## 3. CRITICAL: Transformers Version Conflict

The streaming model's config specifies `transformers_version: "5.0.0.dev0"`.

Our current pipeline pins: `transformers >= 4.35.0, < 4.50.0`

**This means:**
- We CANNOT use moonshine-streaming-small with our current environment
- We need a **separate training environment** with transformers >= 5.0
- All the dependency pinning gotchas (datasets < 4.0, etc.) need to be re-evaluated
- The `gradient_checkpointing` bug and other v1-specific workarounds may or may not apply to v2

**However:** `MoonshineStreamingForConditionalGeneration.forward()` accepts a `labels` parameter, which means it's compatible with HuggingFace's `Seq2SeqTrainer`. The training script adaptation should be relatively straightforward — mainly swapping the model/processor classes and updating dependencies.

---

## 4. VRAM Estimates for RTX 5090 (32 GB)

| Model | Params | Est. training VRAM (batch 16) | Recommended batch |
|-------|:------:|:-----------------------------:|:-----------------:|
| v1 tiny | 27M | ~8-10 GB | 16 |
| v1 base | 61M | ~18-22 GB | 16 (maybe 12) |
| v2 streaming-small | 123M | ~28-35 GB | **8** (with grad accum 8 = effective 64) |

streaming-small at 123M params may be tight on a single RTX 5090. Options:
- batch 8 with gradient_accumulation_steps 8 → effective batch 64
- batch 4 with gradient_accumulation_steps 16 → effective batch 64
- If OOM, try `gradient_checkpointing: true` (may work in v2/transformers 5.x)
- bf16 is essential

---

## 5. Dataset Resources from Issue #141

fosple compiled an excellent list. Key datasets sorted by size:

| Dataset | Size | License | Notes |
|---------|:----:|---------|-------|
| Emilia-YODAS (DE) | ~5600h | CC BY 4.0 | Web speech (podcasts, lectures). Only YODAS subset is CC BY 4.0 |
| MLS German | ~1100h | CC BY 4.0 | LibriVox audiobooks. What we currently use |
| Bundestag | ~600h | Custom | Parliamentary speeches |
| SWC | ~386h | CC BY-SA 4.0 | Wikipedia articles read aloud |
| Common Voice (DE) | ~370h | CC0 | Crowdsourced, many speakers |
| VoxPopuli (DE) | ~282h | CC0 | EU Parliament |
| M-AILABS (DE) | ~237h | BSD-3 | Audiobook speech |

**Important data quality note from RobFlo98:**
German subtitle data often paraphrases/condenses speech (unlike English which is near-verbatim). This can teach the model to summarize rather than transcribe. Need to filter out subtitle sources or clean boilerplate patterns like "Untertitelung. BR" or "Untertitel im Auftrag des ZDF/Funk".

Also mentioned: **Nvidia Granary** — a cleaned/standardized multilingual speech corpus (~1M hours) that aggregates many of these datasets.

---

## 6. Revised Strategy

### Option A: Follow fosple's path (streaming-small, new env)

**Pros:**
- Proven to work: 13% WER on German
- 123M params = much more capacity for German compounds
- Same tokenizer — tokenizer analysis still applies
- v2 architecture is the future of Moonshine

**Cons:**
- Requires transformers >= 5.0 — completely new training environment
- All dependency pinning must be re-evaluated
- 123M params → higher VRAM, slower training
- fosple hasn't shared details — we're reverse-engineering the approach

**Steps:**
1. Create new training environment with transformers 5.x on ziggie
2. Adapt train.py: swap `MoonshineForConditionalGeneration` → `MoonshineStreamingForConditionalGeneration`
3. Test with small dataset first (60k MLS subset)
4. Ask fosple in issue #141 for dataset/config details
5. Scale up once verified

### Option B: Try v1 base first (safer, known env)

**Pros:**
- Same training environment — known dependencies, known gotchas
- Quick to test — just change `model.name` in config
- If base breaks 25% WER, may be good enough

**Cons:**
- fosple's result suggests v2 is dramatically better for non-English
- base (61M) may still plateau like tiny did — just at a lower WER
- Less future-proof

### Recommended: Option A (with B as fallback)

The 13% WER result from a moonshine-ai member is too strong to ignore. The environment setup is more work upfront, but it's the path that's proven to work.

---

## 7. Immediate Next Steps

1. **Reply to fosple in issue #141** — ask about dataset mix, training config, whether they used curriculum learning, and if the model will be uploaded to HF
2. **Set up transformers 5.x environment on ziggie** — separate venv/uv project, test that `MoonshineStreamingForConditionalGeneration` loads and runs inference
3. **Adapt train.py for streaming model** — minimal changes needed (model class swap, processor swap)
4. **Quick sanity test** — fine-tune streaming-small on 10k German samples for a few hundred steps, verify training loop works
5. **Meanwhile, try v1 base on 60k as low-effort experiment** — uses current env, no setup needed

---

## 8. Key People in Issue #141

- **fosple** — moonshine-ai org member, achieved 13% WER on German streaming-small. Active contributor to baresip, google-ai-edge/mediapipe, onnx-asr. Most contributions in private repos.
- **petewarden** — Moonshine co-creator/contributor, expressed interest in German support
- **leonbubova** — Issue author, tracking German support
- **RobFlo98** — Data quality insights about German subtitle paraphrasing
- **dattazigzag** — Us (Saurabh), reported 31.5% WER, asked fosple questions
