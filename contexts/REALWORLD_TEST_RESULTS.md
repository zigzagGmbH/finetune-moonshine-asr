# Real-World German Audio Test Results

**Date:** 2026-03-26
**Model:** [dattazigzag/moonshine-tiny-de](https://huggingface.co/dattazigzag/moonshine-tiny-de)
**Base:** UsefulSensors/moonshine-tiny (27M params)
**Training:** 10k steps on MLS German (~1,967h audiobook), 36.7% WER on test set

---

## Test 1: Live Microphone (Mac, CPU, VAD)

**Setup:** MacBook mic → Silero VAD → moonshine-tiny-de (CPU)
**Mode:** `--live` with VAD enabled

| Spoken | Transcribed | Notes |
|--------|------------|-------|
| "Hallo" | "halle hallo" / "hallo" | Good on short greetings |
| "Wie geht es dir?" | "gäbte er" / "begette es dir" | VAD cuts too aggressively, fragments hurt |
| "Das Wetter ist heute sehr schön in Berlin" | "wetter ist heute sehr schon in berlin" | Quite good! Missing capitalization (expected — no casing in training) |

**Observations:**
- VAD fragments speech into very short chunks (sub-second), which the model struggles with — it was trained on 4–20s audiobook segments
- When VAD captures a full phrase, results are decent
- RTF: 0.02–0.29x on CPU (fast)
- `--no-vad --chunk-duration 4.0` mode recommended for better results

**VAD fix required:** Silero VAD API changed — now expects exactly 512 samples at 16kHz. Fixed buffer size from 4608 (3×1536) to 512.

---

## Test 2: File-Based (2min interview excerpt, Mac, CPU)

**Setup:** 2-minute WAV extracted from a 1h46m German Mercedes interview recording (conversational, multi-speaker)
**Source:** `2025-10-23_Session-8.m4a` → ffmpeg → 16kHz mono WAV
**Mode:** `--audio german_interview_test.wav`

**Result:**
```
so der so zweckte ssergej verschen seien das wäre herr jetzt ein schwarzscharenschar's
g't'r'n h''m'h'l'd'b'st'g'in s'ch'er jesu f'est're jows john jews josef james junges
johns jane josu
```

**Observations:**
- Very poor on conversational multi-speaker audio — expected, given training data
- Degenerates into character-level gibberish after initial words
- The model was trained on clean read-speech audiobooks (MLS), not conversational speech
- 120s audio processed in 2.78s (RTF 0.02x) — speed is not the issue, quality is
- The inference script treats the entire 120s as one chunk, which exceeds the 4–20s training range

**Root cause:** The inference.py script doesn't segment long files. It passes the entire 120s audio to `model.generate()` with `max_new_tokens=150`, which is far too few tokens for 2 minutes of speech and far beyond the model's training distribution.

---

## Recommendations for Improvement

### Short-term (model usage)
1. **Segment long audio before inference** — split into 4–15s chunks using VAD or silence detection
2. **Use `--no-vad --chunk-duration 4.0`** for live mode — better than fragmented VAD
3. **Don't use for multi-speaker conversational audio** — model was trained on single-speaker audiobook data

### Medium-term (model quality)
1. Resume training to 20k+ steps (loss was still dropping at 10k)
2. Add curriculum learning
3. Add diverse data: SWC, VoxPopuli, Bundestag (non-audiobook sources)

### For ONNX/ORT export
- `optimum` 1.x incompatible with `torch` 2.11 (removed `torch.onnx` internal symbols)
- `optimum` 2.x dropped the ONNX exporter entirely
- Need guidance from Moonshine team on ORT conversion path
- Custom `torch.onnx.export` is possible but requires careful encoder/decoder separation + KV cache handling

---

## Environment

- **Mac:** MacBook (Apple Silicon), CPU inference, Python 3.12
- **Ziggie:** Ubuntu 24.04, RTX 5090, CUDA, Python 3.12
- **torch:** 2.11.0
- **transformers:** 4.49.0
- **datasets:** 3.6.0
