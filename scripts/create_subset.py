#!/usr/bin/env python3
"""
Create a stratified subset of a local HuggingFace dataset.

Stratifies by audio duration so the subset has the same duration
distribution as the full dataset (short, medium, and long clips
all represented proportionally).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/create_subset.py \
        --source /storage/datasets/german_combined \
        --output /storage/datasets/german_100k \
        --size 100000 --stratify-by-duration

    # Quick version (random, no stratification):
    uv run python scripts/create_subset.py \
        --source /storage/datasets/german_combined \
        --output /storage/datasets/german_60k \
        --size 60000
"""

import argparse
import numpy as np
from datasets import load_from_disk, DatasetDict


def create_subset(source_path: str, output_path: str, size: int, stratify: bool = False, seed: int = 42):
    print(f"Loading dataset from: {source_path}")
    dataset = load_from_disk(source_path)

    if isinstance(dataset, DatasetDict):
        train = dataset["train"]
        test = dataset["test"]
    else:
        raise ValueError(f"Expected DatasetDict, got {type(dataset)}")

    print(f"Full train set: {len(train):,} samples")
    print(f"Full test set:  {len(test):,} samples")

    if size >= len(train):
        print(f"Requested size ({size:,}) >= train set ({len(train):,}). Using full dataset.")
        dataset.save_to_disk(output_path)
        print(f"Saved to: {output_path}")
        return

    rng = np.random.default_rng(seed)

    if stratify and "duration" in train.column_names:
        print(f"\nStratifying by duration...")
        durations = np.array(train["duration"])

        # Create 5 bins by duration
        bins = np.quantile(durations, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_indices = np.digitize(durations, bins[1:-1])  # 0-4

        selected = []
        for b in range(5):
            bin_mask = bin_indices == b
            bin_indices_arr = np.where(bin_mask)[0]
            n_from_bin = int(size * bin_mask.sum() / len(train))
            n_from_bin = min(n_from_bin, len(bin_indices_arr))
            chosen = rng.choice(bin_indices_arr, size=n_from_bin, replace=False)
            selected.extend(chosen.tolist())
            print(f"  Bin {b} ({bins[b]:.1f}s - {bins[b+1]:.1f}s): "
                  f"{bin_mask.sum():,} total → {n_from_bin:,} selected")

        # If rounding left us short, fill randomly from remaining
        remaining = size - len(selected)
        if remaining > 0:
            all_indices = set(range(len(train)))
            used = set(selected)
            available = list(all_indices - used)
            extra = rng.choice(available, size=remaining, replace=False)
            selected.extend(extra.tolist())
            print(f"  Filled {remaining} extra samples to reach target")

        selected = sorted(selected[:size])
    elif "audio_duration" in train.column_names and stratify:
        # Same but with audio_duration column name
        print("Found 'audio_duration' column, using that for stratification...")
        durations = np.array(train["audio_duration"])
        bins = np.quantile(durations, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_indices = np.digitize(durations, bins[1:-1])

        selected = []
        for b in range(5):
            bin_mask = bin_indices == b
            bin_indices_arr = np.where(bin_mask)[0]
            n_from_bin = int(size * bin_mask.sum() / len(train))
            n_from_bin = min(n_from_bin, len(bin_indices_arr))
            chosen = rng.choice(bin_indices_arr, size=n_from_bin, replace=False)
            selected.extend(chosen.tolist())

        remaining = size - len(selected)
        if remaining > 0:
            all_indices = set(range(len(train)))
            used = set(selected)
            available = list(all_indices - used)
            extra = rng.choice(available, size=remaining, replace=False)
            selected.extend(extra.tolist())

        selected = sorted(selected[:size])
    else:
        if stratify:
            print("No duration column found — falling back to random selection")
        print(f"\nRandom selection of {size:,} samples...")
        selected = sorted(rng.choice(len(train), size=size, replace=False).tolist())

    subset_train = train.select(selected)

    subset = DatasetDict({
        "train": subset_train,
        "test": test,  # Keep full test set for comparable evaluation
    })

    print(f"\nSubset created:")
    print(f"  Train: {len(subset['train']):,} samples")
    print(f"  Test:  {len(subset['test']):,} samples (full — for comparable eval)")

    subset.save_to_disk(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified dataset subset")
    parser.add_argument("--source", required=True, help="Path to source dataset")
    parser.add_argument("--output", required=True, help="Path to save subset")
    parser.add_argument("--size", type=int, required=True, help="Number of train samples")
    parser.add_argument("--stratify-by-duration", action="store_true",
                        help="Stratify by audio duration (needs 'duration' column)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_subset(args.source, args.output, args.size, args.stratify_by_duration, args.seed)
