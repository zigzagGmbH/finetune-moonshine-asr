#!/usr/bin/env python3
"""
Prepare combined German ASR dataset for Moonshine fine-tuning.

Downloads MLS German + Common Voice DE, normalises columns,
computes duration, filters by Moonshine paper recommendations [4-20s],
and saves as a local DatasetDict.

Usage:
    uv run python prepare_german_dataset.py \
        --output ./data/german_combined \
        --cv-version 17_0 \
        --min-duration 4.0 \
        --max-duration 20.0

Requirements:
    datasets, soundfile, librosa, tqdm
"""

import argparse
import os
from pathlib import Path

from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare combined German dataset for Moonshine fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/german_combined",
        help="Output directory for the prepared dataset",
    )
    parser.add_argument(
        "--cv-version",
        type=str,
        default="17_0",
        help="Common Voice version (default: 17_0)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Minimum audio duration in seconds (Moonshine paper: 4.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target sampling rate (default: 16000)",
    )
    parser.add_argument(
        "--skip-mls",
        action="store_true",
        help="Skip MLS German (for testing with CV only)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip Common Voice (for testing with MLS only)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use only 500 samples per source for quick pipeline test",
    )
    return parser.parse_args()


def compute_duration(example):
    """Compute audio duration in seconds."""
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    example["duration"] = duration
    return example


def load_mls_german(sampling_rate: int, test_mode: bool = False):
    """Load MLS German dataset."""
    print("\n" + "=" * 60)
    print("LOADING: Multilingual LibriSpeech — German")
    print("=" * 60)

    mls_train = load_dataset(
        "facebook/multilingual_librispeech",
        "german",
        split="train",
    )
    mls_test = load_dataset(
        "facebook/multilingual_librispeech",
        "german",
        split="test",
    )

    if test_mode:
        mls_train = mls_train.select(range(min(500, len(mls_train))))
        mls_test = mls_test.select(range(min(100, len(mls_test))))

    print(f"  Raw train: {len(mls_train):,}")
    print(f"  Raw test:  {len(mls_test):,}")

    # MLS uses "transcript" → rename to "sentence"
    mls_train = mls_train.rename_column("transcript", "sentence")
    mls_test = mls_test.rename_column("transcript", "sentence")

    # Keep only audio + sentence
    keep_cols = {"audio", "sentence"}
    mls_train = mls_train.remove_columns(
        [c for c in mls_train.column_names if c not in keep_cols]
    )
    mls_test = mls_test.remove_columns(
        [c for c in mls_test.column_names if c not in keep_cols]
    )

    # Add source tag
    mls_train = mls_train.map(lambda x: {"source": "mls"})
    mls_test = mls_test.map(lambda x: {"source": "mls"})

    # Cast audio to target sampling rate
    mls_train = mls_train.cast_column("audio", Audio(sampling_rate=sampling_rate))
    mls_test = mls_test.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return mls_train, mls_test


def load_common_voice_german(
    version: str, sampling_rate: int, test_mode: bool = False
):
    """Load Common Voice German dataset."""
    print("\n" + "=" * 60)
    print(f"LOADING: Common Voice {version} — German (de)")
    print("=" * 60)

    dataset_name = f"mozilla-foundation/common_voice_{version}"

    cv_train = load_dataset(
        dataset_name,
        "de",
        split="train+validation",  # combine train+val for more data
    )
    cv_test = load_dataset(
        dataset_name,
        "de",
        split="test",
    )

    if test_mode:
        cv_train = cv_train.select(range(min(500, len(cv_train))))
        cv_test = cv_test.select(range(min(100, len(cv_test))))

    print(f"  Raw train+val: {len(cv_train):,}")
    print(f"  Raw test:      {len(cv_test):,}")

    # CV already has "sentence" — keep only audio + sentence
    keep_cols = {"audio", "sentence"}
    cv_train = cv_train.remove_columns(
        [c for c in cv_train.column_names if c not in keep_cols]
    )
    cv_test = cv_test.remove_columns(
        [c for c in cv_test.column_names if c not in keep_cols]
    )

    # Add source tag
    cv_train = cv_train.map(lambda x: {"source": "common_voice"})
    cv_test = cv_test.map(lambda x: {"source": "common_voice"})

    # Cast audio to target sampling rate
    cv_train = cv_train.cast_column("audio", Audio(sampling_rate=sampling_rate))
    cv_test = cv_test.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return cv_train, cv_test


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("GERMAN DATASET PREPARATION FOR MOONSHINE FINE-TUNING")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Duration filter: [{args.min_duration}s, {args.max_duration}s]")
    print(f"  Sampling rate: {args.sampling_rate}")
    if args.test_mode:
        print("  ⚠️  TEST MODE: using 500 samples per source")

    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------
    train_parts = []
    test_parts = []

    if not args.skip_mls:
        mls_train, mls_test = load_mls_german(args.sampling_rate, args.test_mode)
        train_parts.append(mls_train)
        test_parts.append(mls_test)

    if not args.skip_cv:
        cv_train, cv_test = load_common_voice_german(
            args.cv_version, args.sampling_rate, args.test_mode
        )
        train_parts.append(cv_train)
        test_parts.append(cv_test)

    if not train_parts:
        print("ERROR: No datasets loaded! Remove --skip flags.")
        return

    # --------------------------------------------------
    # 2. Combine datasets
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("COMBINING DATASETS")
    print("=" * 60)

    if len(train_parts) > 1:
        combined_train = concatenate_datasets(train_parts)
        combined_test = concatenate_datasets(test_parts)
    else:
        combined_train = train_parts[0]
        combined_test = test_parts[0]

    print(f"  Combined train: {len(combined_train):,}")
    print(f"  Combined test:  {len(combined_test):,}")

    # --------------------------------------------------
    # 3. Compute durations
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPUTING DURATIONS")
    print("=" * 60)

    combined_train = combined_train.map(
        compute_duration,
        desc="Computing train durations",
        num_proc=8,
    )
    combined_test = combined_test.map(
        compute_duration,
        desc="Computing test durations",
        num_proc=8,
    )

    # --------------------------------------------------
    # 4. Filter by duration (Moonshine paper: [4, 30]s)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FILTERING BY DURATION [{args.min_duration}s, {args.max_duration}s]")
    print("=" * 60)

    before_train = len(combined_train)
    before_test = len(combined_test)

    combined_train = combined_train.filter(
        lambda d: args.min_duration <= d <= args.max_duration,
        input_columns=["duration"],
    )
    combined_test = combined_test.filter(
        lambda d: args.min_duration <= d <= args.max_duration,
        input_columns=["duration"],
    )

    print(f"  Train: {before_train:,} → {len(combined_train):,} "
          f"(removed {before_train - len(combined_train):,})")
    print(f"  Test:  {before_test:,} → {len(combined_test):,} "
          f"(removed {before_test - len(combined_test):,})")

    # --------------------------------------------------
    # 5. Filter empty or whitespace-only transcripts
    # --------------------------------------------------
    combined_train = combined_train.filter(
        lambda s: s is not None and s.strip() != "",
        input_columns=["sentence"],
    )
    combined_test = combined_test.filter(
        lambda s: s is not None and s.strip() != "",
        input_columns=["sentence"],
    )

    # --------------------------------------------------
    # 6. Print statistics
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    for split_name, ds in [("Train", combined_train), ("Test", combined_test)]:
        durations = ds["duration"]
        total_hours = sum(durations) / 3600
        avg_dur = sum(durations) / len(durations) if durations else 0
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0

        # Source breakdown
        sources = ds["source"]
        from collections import Counter
        source_counts = Counter(sources)

        print(f"\n  {split_name}: {len(ds):,} samples ({total_hours:.1f} hours)")
        print(f"    Duration: min={min_dur:.1f}s, max={max_dur:.1f}s, avg={avg_dur:.1f}s")
        for src, count in source_counts.items():
            print(f"    Source '{src}': {count:,}")

    # --------------------------------------------------
    # 7. Remove source column (not needed for training)
    # --------------------------------------------------
    combined_train = combined_train.remove_columns(["source"])
    combined_test = combined_test.remove_columns(["source"])

    # --------------------------------------------------
    # 8. Save to disk
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SAVING TO: {args.output}")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        "train": combined_train,
        "test": combined_test,
    })

    dataset_dict.save_to_disk(str(output_path))

    print(f"\n  ✅ Saved {len(combined_train):,} train + {len(combined_test):,} test samples")
    print(f"  Path: {output_path.resolve()}")
    print(f"\n  Next step: train with")
    print(f"    uv run python train.py --config configs/mls_cv_german_no_curriculum.yaml")


if __name__ == "__main__":
    main()


