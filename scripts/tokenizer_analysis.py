#!/usr/bin/env python3
"""
Moonshine Tokenizer Efficiency Analysis
Parses tokenizer.bin (same format as C++ BinTokenizer) and compares
tokenization of German, English, and French text.

Usage:
    python scripts/tokenizer_analysis.py /path/to/tokenizer.bin

The tokenizer.bin can be found in the moonshine repo at:
    python/src/moonshine_voice/assets/tiny-en/tokenizer.bin

No ML dependencies required — pure Python.
"""

import sys
from typing import List

SPACE_STRING = "▁"  # U+2581 - SentencePiece convention


def load_tokenizer(path: str) -> List[bytes]:
    """Parse tokenizer.bin into a list of byte sequences (vocabulary)."""
    vocab = []
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    while offset < len(data):
        first_byte = data[offset]
        offset += 1

        if first_byte == 0:
            vocab.append(b"")  # empty/special token
            continue

        if first_byte < 128:
            byte_count = first_byte
        else:
            second_byte = data[offset]
            offset += 1
            byte_count = (second_byte * 128) + first_byte - 128

        token_bytes = data[offset : offset + byte_count]
        offset += byte_count
        vocab.append(token_bytes)

    return vocab


def tokenize(text: str, vocab: List[bytes]) -> List[int]:
    """Greedy longest-match tokenization (mirrors C++ text_to_tokens)."""
    replaced = text.replace(" ", SPACE_STRING)
    remaining = replaced.encode("utf-8")
    tokens = []

    while remaining:
        longest_match_len = 0
        longest_match_id = -1

        for i, token_bytes in enumerate(vocab):
            if not token_bytes:
                continue
            if len(token_bytes) > len(remaining):
                continue
            if remaining[: len(token_bytes)] == token_bytes:
                if len(token_bytes) > longest_match_len:
                    longest_match_len = len(token_bytes)
                    longest_match_id = i

        if longest_match_id == -1:
            print(f"  WARNING: No match for byte 0x{remaining[0]:02X} ('{chr(remaining[0]) if remaining[0] < 128 else '?'}')")
            remaining = remaining[1:]  # skip unknown byte
            continue

        tokens.append(longest_match_id)
        remaining = remaining[longest_match_len:]

    return tokens


def decode_tokens(token_ids: List[int], vocab: List[bytes]) -> str:
    """Decode token IDs back to text."""
    result = b""
    for tid in token_ids:
        result += vocab[tid]
    text = result.decode("utf-8", errors="replace")
    text = text.replace(SPACE_STRING, " ").strip()
    return text


def analyze_token(token_bytes: bytes) -> str:
    """Pretty-print a token's content."""
    try:
        return token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return repr(token_bytes)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokenizer_analysis.py /path/to/tokenizer.bin")
        print("\nThe tokenizer.bin can be found in the moonshine repo at:")
        print("  python/src/moonshine_voice/assets/tiny-en/tokenizer.bin")
        sys.exit(1)

    TOKENIZER_PATH = sys.argv[1]
    vocab = load_tokenizer(TOKENIZER_PATH)

    print(f"{'='*70}")
    print(f"MOONSHINE TOKENIZER ANALYSIS")
    print(f"{'='*70}")
    print(f"\nVocabulary size: {len(vocab)} tokens")

    # Count special (empty) tokens
    special_count = sum(1 for t in vocab if not t)
    print(f"Special (empty) tokens: {special_count}")
    print(f"Real tokens: {len(vocab) - special_count}")

    # ── Vocabulary character coverage ────────────────────────────────

    print(f"\n{'─'*70}")
    print("GERMAN CHARACTER COVERAGE IN VOCABULARY")
    print(f"{'─'*70}")

    german_chars = {
        "ä": "a-umlaut", "ö": "o-umlaut", "ü": "u-umlaut",
        "Ä": "A-umlaut", "Ö": "O-umlaut", "Ü": "U-umlaut",
        "ß": "eszett",
    }

    for char, name in german_chars.items():
        char_bytes = char.encode("utf-8")
        containing = []
        for i, t in enumerate(vocab):
            if char_bytes in t:
                containing.append((i, analyze_token(t)))

        print(f"\n  '{char}' ({name}, UTF-8: {char_bytes.hex()}):")
        if containing:
            print(f"    Found in {len(containing)} token(s)")
            for tid, text in containing[:15]:
                print(f"      [{tid:5d}] '{text}'")
            if len(containing) > 15:
                print(f"      ... and {len(containing) - 15} more")
        else:
            print(f"    NOT FOUND in vocabulary!")

    # ── Check for ▁ (space marker) tokens with German chars ─────────

    print(f"\n{'─'*70}")
    print("GERMAN WORD-START TOKENS (▁ + German char)")
    print(f"{'─'*70}")

    for char, name in german_chars.items():
        space_char = (SPACE_STRING + char).encode("utf-8")
        containing = []
        for i, t in enumerate(vocab):
            if t.startswith(space_char):
                containing.append((i, analyze_token(t)))

        if containing:
            print(f"\n  '▁{char}' tokens: {len(containing)}")
            for tid, text in containing[:10]:
                print(f"    [{tid:5d}] '{text}'")
            if len(containing) > 10:
                print(f"    ... and {len(containing) - 10} more")
        else:
            print(f"\n  '▁{char}' tokens: NONE")

    # ── Tokenization comparison ──────────────────────────────────────

    print(f"\n{'='*70}")
    print("TOKENIZATION EFFICIENCY: GERMAN vs ENGLISH vs FRENCH")
    print(f"{'='*70}")

    test_sentences = [
        {
            "label": "Simple sentence",
            "de": "Das ist ein einfacher Satz",
            "en": "This is a simple sentence",
            "fr": "C'est une phrase simple",
        },
        {
            "label": "Compound word (speed limit)",
            "de": "Die Geschwindigkeitsbeschränkung beträgt hundert Kilometer pro Stunde",
            "en": "The speed limit is one hundred kilometers per hour",
            "fr": "La limitation de vitesse est de cent kilomètres par heure",
        },
        {
            "label": "Umlauts & eszett",
            "de": "Die Straße führt über den Fluss zur schönen Brücke",
            "en": "The road leads over the river to the beautiful bridge",
            "fr": "La route mène au-dessus de la rivière vers le beau pont",
        },
        {
            "label": "Separable verb",
            "de": "Er steht jeden Morgen um sechs Uhr auf",
            "en": "He gets up every morning at six o'clock",
            "fr": "Il se lève chaque matin à six heures",
        },
        {
            "label": "Long compound",
            "de": "Das Bundesverfassungsgericht hat entschieden",
            "en": "The Federal Constitutional Court has decided",
            "fr": "La Cour constitutionnelle fédérale a décidé",
        },
        {
            "label": "Everyday speech",
            "de": "Ich möchte gerne eine Tasse Kaffee bestellen",
            "en": "I would like to order a cup of coffee",
            "fr": "Je voudrais commander une tasse de café",
        },
        {
            "label": "Technical/formal",
            "de": "Die Entwicklung der künstlichen Intelligenz schreitet voran",
            "en": "The development of artificial intelligence is progressing",
            "fr": "Le développement de l'intelligence artificielle progresse",
        },
        {
            "label": "Numbers & units",
            "de": "Die Temperatur beträgt fünfundzwanzig Grad Celsius",
            "en": "The temperature is twenty five degrees Celsius",
            "fr": "La température est de vingt-cinq degrés Celsius",
        },
    ]

    print(f"\n{'Label':<30s} {'DE tok':>7s} {'EN tok':>7s} {'FR tok':>7s} {'DE/EN':>7s} {'DE/FR':>7s} {'DE words':>9s} {'tok/word DE':>11s} {'tok/word EN':>11s}")
    print("─" * 120)

    total_de, total_en, total_fr = 0, 0, 0
    total_de_words, total_en_words, total_fr_words = 0, 0, 0

    for sent in test_sentences:
        de_tok = tokenize(sent["de"], vocab)
        en_tok = tokenize(sent["en"], vocab)
        fr_tok = tokenize(sent["fr"], vocab)

        de_words = len(sent["de"].split())
        en_words = len(sent["en"].split())
        fr_words = len(sent["fr"].split())

        total_de += len(de_tok)
        total_en += len(en_tok)
        total_fr += len(fr_tok)
        total_de_words += de_words
        total_en_words += en_words
        total_fr_words += fr_words

        ratio_de_en = len(de_tok) / len(en_tok) if en_tok else 0
        ratio_de_fr = len(de_tok) / len(fr_tok) if fr_tok else 0
        tpw_de = len(de_tok) / de_words
        tpw_en = len(en_tok) / en_words

        print(f"  {sent['label']:<28s} {len(de_tok):>7d} {len(en_tok):>7d} {len(fr_tok):>7d} {ratio_de_en:>7.2f} {ratio_de_fr:>7.2f} {de_words:>9d} {tpw_de:>11.2f} {tpw_en:>11.2f}")

    print("─" * 120)
    print(f"  {'TOTAL':<28s} {total_de:>7d} {total_en:>7d} {total_fr:>7d} {total_de/total_en:>7.2f} {total_de/total_fr:>7.2f} {total_de_words:>9d} {total_de/total_de_words:>11.2f} {total_en/total_en_words:>11.2f}")

    # ── Detailed token-by-token breakdown ────────────────────────────

    print(f"\n{'='*70}")
    print("DETAILED TOKEN BREAKDOWN (selected sentences)")
    print(f"{'='*70}")

    detail_indices = [1, 2, 4]  # compound word, umlauts, long compound

    for idx in detail_indices:
        sent = test_sentences[idx]
        print(f"\n── {sent['label']} ──")

        for lang in ["de", "en", "fr"]:
            text = sent[lang]
            tokens = tokenize(text, vocab)
            token_strs = [analyze_token(vocab[t]) for t in tokens]
            print(f"\n  [{lang.upper()}] \"{text}\"")
            print(f"       tokens ({len(tokens)}): {' | '.join(token_strs)}")

    # ── Worst-case German words ──────────────────────────────────────

    print(f"\n{'='*70}")
    print("WORST-CASE GERMAN WORDS (long compounds & special chars)")
    print(f"{'='*70}")

    worst_case_words = [
        "Geschwindigkeitsbeschränkung",
        "Bundesverfassungsgericht",
        "Straßenbahnhaltestelle",
        "Kraftfahrzeughaftpflichtversicherung",
        "Donaudampfschifffahrtsgesellschaft",
        "Streichholzschächtelchen",
        "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
        "Grundstücksverkehrsgenehmigungszuständigkeitsübertragungsverordnung",
        "über",
        "Straße",
        "schön",
        "größer",
        "Gemütlichkeit",
        "Fußball",
        "Brötchen",
    ]

    print(f"\n  {'Word':<65s} {'Tokens':>7s} {'Tok/Char':>9s}")
    print("  " + "─" * 83)

    for word in worst_case_words:
        tokens = tokenize(word, vocab)
        token_strs = [analyze_token(vocab[t]) for t in tokens]
        ratio = len(tokens) / len(word)
        print(f"  {word:<65s} {len(tokens):>7d} {ratio:>9.3f}")
        print(f"    → {' | '.join(token_strs)}")

    # ── Summary ──────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Vocab size:                  {len(vocab)} tokens
  German tokens per word:      {total_de/total_de_words:.2f} (across test set)
  English tokens per word:     {total_en/total_en_words:.2f} (across test set)
  French tokens per word:      {total_fr/total_fr_words:.2f} (across test set)
  German/English token ratio:  {total_de/total_en:.2f}x  (>1.0 = German needs more tokens)
  German/French token ratio:   {total_de/total_fr:.2f}x

  INTERPRETATION:
  - If DE/EN ratio > 1.3: Tokenizer is significantly penalizing German
  - If DE/FR ratio > 1.2: German is harder for this tokenizer than French
  - Tokens per word > 3.0: Severe fragmentation (compound words being shredded)
  - For ASR, more tokens per word = more sequential predictions = more error compounding
""")
